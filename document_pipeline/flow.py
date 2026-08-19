"""Prefect tasks and flows for the mail, scan and enrich pipelines."""

from __future__ import annotations

import os
import time

from prefect import flow, get_run_logger, task
from prefect.concurrency.sync import concurrency

from document_pipeline import enrich, extract, imap_client, metrics, scan, webdav


@task(name="process-mail", log_prints=True)
def process_mail_task() -> tuple[int, int]:
    """Fetch unprocessed messages from IMAP, submit PDFs, mark processed.

    Returns (messages_processed, pdfs_submitted).
    """
    logger = get_run_logger()
    started = time.perf_counter()

    with imap_client.open_inbox() as conn:
        messages = imap_client.fetch_unprocessed(conn)
        logger.info("process-mail: %d unprocessed message(s)", len(messages))
        pdfs_submitted = 0
        for uid, msg in messages:
            if extract.submit_message_pdfs(
                msg,
                paperless_url=os.environ["PAPERLESS_URL"],
                paperless_token=os.environ["PAPERLESS_API_TOKEN"],
            ):
                pdfs_submitted += 1
            imap_client.mark_processed(conn, uid)

    logger.info(
        "process-mail complete in %.2fs: %d message(s), %d PDF(s) submitted",
        time.perf_counter() - started, len(messages), pdfs_submitted,
    )
    return len(messages), pdfs_submitted


@task(name="push-metrics", log_prints=True)
def push_metrics_task(messages_processed: int, pdfs_submitted: int, duration_seconds: float) -> None:
    metrics.push_run_metrics(messages_processed, pdfs_submitted, duration_seconds)


@flow(name="mail", log_prints=True)
def mail_flow() -> None:
    logger = get_run_logger()
    flow_started = time.perf_counter()
    slot_acquired = False
    try:
        with concurrency("mail-pipeline", occupy=1, timeout_seconds=10):
            slot_acquired = True
            messages_processed, pdfs_submitted = process_mail_task()
            push_metrics_task(messages_processed, pdfs_submitted, time.perf_counter() - flow_started)
        logger.info("mail flow complete in %.2fs", time.perf_counter() - flow_started)
    except TimeoutError:
        if not slot_acquired:
            logger.info("Skipped — mail pipeline already running")
        else:
            raise


@task(name="process-scans", log_prints=True)
def process_scans_task() -> scan.ScanResult:
    """Drain the scanner's WebDAV directory into Paperless."""
    logger = get_run_logger()
    started = time.perf_counter()

    client = webdav.WebDAVClient(
        base_url=os.environ["WEBDAV_URL"],
        username=os.environ["WEBDAV_USERNAME"],
        password=os.environ["WEBDAV_PASSWORD"],
    )
    result = scan.ingest_scans(
        client,
        scan_path=os.environ.get("WEBDAV_SCAN_PATH", "/"),
        paperless_url=os.environ["PAPERLESS_URL"],
        paperless_token=os.environ["PAPERLESS_API_TOKEN"],
    )

    logger.info(
        "process-scans complete in %.2fs: %d ingested, %d failed, %d ignored",
        time.perf_counter() - started, result.ingested, result.failed, result.ignored,
    )
    return result


@task(name="push-scan-metrics", log_prints=True)
def push_scan_metrics_task(result: scan.ScanResult, duration_seconds: float) -> None:
    metrics.push_scan_metrics(
        result.ingested, result.failed, result.pending, result.oldest_pending_age_seconds, duration_seconds
    )


@flow(name="scan", log_prints=True)
def scan_flow() -> None:
    logger = get_run_logger()
    flow_started = time.perf_counter()
    slot_acquired = False
    try:
        # Its own slot, not the mail one: a long OCR wait must not block mail
        # ingestion, but two scan runs draining the same directory would race
        # each other into duplicate submissions.
        with concurrency("scan-pipeline", occupy=1, timeout_seconds=10):
            slot_acquired = True
            result = process_scans_task()
            push_scan_metrics_task(result, time.perf_counter() - flow_started)
        logger.info("scan flow complete in %.2fs", time.perf_counter() - flow_started)
    except TimeoutError:
        if not slot_acquired:
            logger.info("Skipped — scan pipeline already running")
        else:
            raise


@task(name="enrich-document", retries=2, retry_delay_seconds=[60, 300], log_prints=True)
def enrich_document_task(document_id: int, marker_id: int | None = None) -> enrich.EnrichResult:
    """Retitle and tag one consumed document.

    Retries cover the 503 `ai_suggestions` returns when Ollama is saturated —
    the failure mode that used to cost a document its title permanently, because
    the shell hook had nothing to re-queue it. Re-running is idempotent (same
    title, union of tags) and paperless caches LLM suggestions per document, so
    a retry inside the cache window costs no further inference.
    """
    logger = get_run_logger()
    paperless_url = os.environ["PAPERLESS_URL"]
    with enrich.open_client(_paperless_admin_token()) as client:
        if marker_id is None:
            marker_id = enrich.resolve_marker_tag(client, paperless_url)
        result = enrich.enrich_document(client, paperless_url, document_id, marker_id)

    enrich.append_result(result)
    logger.info(
        "enrich-document %s complete in %.2fs: %s", document_id, result.duration_seconds, result.outcome
    )
    return result


def _paperless_admin_token() -> str:
    """The superuser token, falling back to the ingest token when unset.

    API-uploaded documents are owned by the uploading user and paperless applies
    object-level permissions, so enriching a document someone else uploaded needs
    a superuser. The fallback keeps a Renovate digest bump that lands before the
    manifest configuring it from turning into a startup failure — same reasoning
    as the opt-in scan config in cli.py.
    """
    return os.environ.get("PAPERLESS_ADMIN_TOKEN") or os.environ["PAPERLESS_API_TOKEN"]


@task(name="push-enrich-metrics", log_prints=True)
def push_enrich_metrics_task(document_id: int, succeeded: bool) -> None:
    metrics.push_enrich_metrics(document_id, succeeded)


@flow(name="enrich", log_prints=True)
def enrich_flow(document_id: int) -> None:
    flow_started = time.perf_counter()
    logger = get_run_logger()
    # No `timeout_seconds`, deliberately unlike mail_flow and scan_flow. Those
    # treat a busy slot as "skip, the next cron covers it"; this is per-document
    # work, so a skip would silently lose that document. It queues instead.
    try:
        with concurrency("ollama", occupy=1):
            result = enrich_document_task(document_id)
    except Exception:
        push_enrich_metrics_task(document_id, succeeded=False)
        raise
    # A skip is neither success nor failure: a document with no OCR text has
    # nothing to title from, and pushing either series would misreport it. The
    # shell hook's `skip()` pushed nothing for the same reason.
    if result.outcome == "enriched":
        push_enrich_metrics_task(document_id, succeeded=True)
    logger.info("enrich flow complete in %.2fs", time.perf_counter() - flow_started)


@flow(name="enrich-sweep", log_prints=True)
def enrich_sweep_flow(batch_size: int | None = None) -> None:
    """Enrich documents that carry no `ai-processed` marker.

    Belt and braces for a dropped trigger — and, with a large enough batch, this
    is the backfill over the pre-existing library (#1280): same code path, wider
    query, rather than a separate one-off script.
    """
    logger = get_run_logger()
    flow_started = time.perf_counter()
    if batch_size is None:
        batch_size = int(os.environ.get("ENRICH_SWEEP_BATCH_SIZE", "20"))

    paperless_url = os.environ["PAPERLESS_URL"]
    with enrich.open_client(_paperless_admin_token()) as client:
        marker_id = enrich.resolve_marker_tag(client, paperless_url)
        document_ids = enrich.find_unenriched(client, paperless_url, marker_id, batch_size)

    logger.info("enrich-sweep: %d unenriched document(s), batch size %d", len(document_ids), batch_size)
    enriched = failed = 0
    for document_id in document_ids:
        try:
            with concurrency("ollama", occupy=1):
                enrich_document_task(document_id, marker_id)
            enriched += 1
        except Exception as exc:
            # Per-document isolation: one document the LLM cannot handle must
            # not strand the rest of the batch behind it.
            logger.error("enrich-sweep: document %s failed: %s", document_id, exc)
            failed += 1
            push_enrich_metrics_task(document_id, succeeded=False)

    logger.info(
        "enrich-sweep complete in %.2fs: %d enriched, %d failed",
        time.perf_counter() - flow_started, enriched, failed,
    )
