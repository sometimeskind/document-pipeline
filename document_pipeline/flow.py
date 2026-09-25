"""Prefect tasks and flows for the mail, scan and enrich pipelines."""

from __future__ import annotations

import os
import time

from prefect import flow, get_run_logger, task
from prefect.concurrency.sync import concurrency

from document_pipeline import (
    enrich, extract, imap_client, metrics, paperless_health, prefect_client, scan, webdav,
)


@task(name="process-mail", log_prints=True)
def process_mail_task() -> tuple[int, int]:
    """Fetch unprocessed messages from IMAP, queue their PDFs, mark processed.

    PDFs go into the `mail/` directory of the WebDAV scan queue, not to
    Paperless directly: the scan flow drains that queue and is what follows
    each file to a terminal consume state (homelab#1590). A message is flagged
    `$Processed` only once every PUT for it succeeded; a failed PUT leaves it
    unflagged and it is retried next run, where the UID-prefixed object names
    make the retry overwrite rather than duplicate.

    Returns (messages_processed, pdfs_queued).
    """
    logger = get_run_logger()
    started = time.perf_counter()

    queue_path = scan.source_path(os.environ.get("WEBDAV_SCAN_PATH", "/"), "mail")
    with imap_client.open_inbox() as conn:
        messages = imap_client.fetch_unprocessed(conn)
        logger.info("process-mail: %d unprocessed message(s)", len(messages))
        pdfs_queued = 0
        failed = 0
        queue = _webdav_client()
        if messages:
            queue.mkcol(queue_path)
        for uid, msg in messages:
            try:
                pdfs_queued += extract.queue_message_pdfs(msg, uid.decode(), queue, queue_path)
            except Exception as exc:
                # Per-message isolation, and no flag: the next run retries it.
                logger.error("process-mail: message %s left unflagged, PUT failed: %s", uid.decode(), exc)
                failed += 1
                continue
            imap_client.mark_processed(conn, uid)

    if pdfs_queued:
        # The hop into Paperless is the scan flow's; trigger it now rather than
        # waiting for its hourly sweep. Coalesced exactly like /trigger-scan.
        if not prefect_client.has_active_scan_run() and not prefect_client.trigger_scan():
            logger.warning("process-mail: could not trigger a scan run — the hourly sweep will drain the queue")

    logger.info(
        "process-mail complete in %.2fs: %d message(s), %d PDF(s) queued, %d message(s) failed",
        time.perf_counter() - started, len(messages), pdfs_queued, failed,
    )
    if failed:
        raise RuntimeError(f"process-mail: {failed} message(s) could not be queued and stay unflagged")
    return len(messages), pdfs_queued


def _webdav_client() -> webdav.WebDAVClient:
    return webdav.WebDAVClient(
        base_url=os.environ["WEBDAV_URL"],
        username=os.environ["WEBDAV_USERNAME"],
        password=os.environ["WEBDAV_PASSWORD"],
    )


@task(name="push-metrics", log_prints=True)
def push_metrics_task(messages_processed: int, pdfs_submitted: int, duration_seconds: float) -> None:
    metrics.push_run_metrics(messages_processed, pdfs_submitted, duration_seconds)


@task(name="push-failure-metrics", log_prints=True)
def push_failure_metrics_task() -> None:
    metrics.push_failure_metrics()


@flow(name="mail", log_prints=True)
def mail_flow() -> None:
    logger = get_run_logger()
    flow_started = time.perf_counter()
    slot_acquired = False
    try:
        with concurrency("mail-pipeline", occupy=1, timeout_seconds=10):
            slot_acquired = True
            try:
                messages_processed, pdfs_submitted = process_mail_task()
            except Exception:
                # A Failed run must still say so in Prometheus. Pushing nothing
                # freezes `document_pipeline_last_success_timestamp` and leaves
                # `..._prefect_failures_24h` at 0, so a pipeline failing every
                # run reads exactly like an idle one — which is how #58 ran 43
                # hours unnoticed (#60). Shaped like enrich_flow's failure push.
                push_failure_metrics_task()
                raise
            push_metrics_task(messages_processed, pdfs_submitted, time.perf_counter() - flow_started)
        logger.info("mail flow complete in %.2fs", time.perf_counter() - flow_started)
    except TimeoutError:
        if not slot_acquired:
            logger.info("Skipped — mail pipeline already running")
        else:
            raise


@task(name="process-scans", log_prints=True)
def process_scans_task() -> scan.ScanResult:
    """Drain the WebDAV scan queue — scanner root and `mail/` — into Paperless."""
    logger = get_run_logger()
    started = time.perf_counter()

    result = scan.ingest_scans(
        _webdav_client(),
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
    metrics.push_scan_metrics(result.sources, duration_seconds)


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
def enrich_document_task(
    document_id: int,
    queue_id: int | None = None,
    dry_run: bool = False,
    mode: str | None = None,
    tag_vocabulary: dict[str, int] | None = None,
    sample: bool = False,
) -> enrich.EnrichResult:
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
        if queue_id is None:
            queue_id = enrich.resolve_queue_tag(client, paperless_url)
        result = enrich.enrich_document(
            client, paperless_url, document_id, queue_id, dry_run=dry_run,
            mode=mode, tag_vocabulary=tag_vocabulary, sample=sample,
        )

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
def enrich_sweep_flow(
    batch_size: int | None = None,
    dry_run: bool = False,
    mode: str | None = None,
    document_ids: list[int] | None = None,
) -> None:
    """Enrich documents still carrying the `queue` tag.

    Belt and braces for a dropped trigger — and, run on a cron, this is the
    backfill over the pre-existing library (#1280): same code path, repeated,
    rather than a separate one-off script.

    `dry_run` reports what it would do and writes nothing, which is how a sample
    gets reviewed before 2000-odd documents are retitled and renamed for real.
    Because a dry run leaves `queue` in place, it re-reads the same documents every
    time — it is a sample, not a pass over the library.

    `mode` overrides ENRICH_MODE for this run, and `document_ids` replaces the
    `queue` query with a fixed list, enriched whether or not it still carries
    `queue` and even past a curated title (dry-run only). Together they are the homelab#1563 gate: the same
    sample dry-run once per mode, then `python -m document_pipeline compare`.
    """
    logger = get_run_logger()
    flow_started = time.perf_counter()
    slot_acquired = False
    if document_ids and not dry_run:
        raise ValueError("document_ids re-enriches processed documents and needs dry_run=true")
    mode = enrich.resolve_mode(mode)
    if batch_size is None:
        batch_size = int(os.environ.get("ENRICH_SWEEP_BATCH_SIZE", "20"))

    try:
        # An hourly cron whose batch runs long would otherwise overlap the next
        # run, and both would query the same unenriched set and do the same work
        # twice. Skipping is right here for the same reason it is in mail_flow:
        # whatever this run does not reach, the next hour picks up.
        with concurrency("enrich-sweep", occupy=1, timeout_seconds=10):
            slot_acquired = True
            _run_sweep(batch_size, dry_run, mode, document_ids)
        logger.info("enrich-sweep complete in %.2fs", time.perf_counter() - flow_started)
    except TimeoutError:
        if not slot_acquired:
            logger.info("Skipped — enrich sweep already running")
        else:
            raise


def _run_sweep(
    batch_size: int, dry_run: bool, mode: str, sample_ids: list[int] | None = None
) -> None:
    logger = get_run_logger()
    paperless_url = os.environ["PAPERLESS_URL"]
    tag_vocabulary = None
    with enrich.open_client(_paperless_admin_token()) as client:
        queue_id = enrich.resolve_queue_tag(client, paperless_url)
        if sample_ids:
            document_ids = [int(d) for d in sample_ids]
        else:
            document_ids = enrich.find_unenriched(client, paperless_url, queue_id, batch_size)
        if mode == "extract":
            # Once per run, not per document (homelab#1563).
            tag_vocabulary = enrich.fetch_tag_vocabulary(client, paperless_url)

    logger.info(
        "enrich-sweep: %d %s document(s), batch size %d, mode %s%s",
        len(document_ids), "sampled" if sample_ids else "unenriched", batch_size, mode,
        " (DRY RUN — nothing will be written)" if dry_run else "",
    )
    enriched = failed = 0
    for document_id in document_ids:
        try:
            with concurrency("ollama", occupy=1):
                enrich_document_task(
                    document_id, queue_id, dry_run,
                    mode=mode, tag_vocabulary=tag_vocabulary, sample=bool(sample_ids),
                )
            enriched += 1
        except Exception as exc:
            # Per-document isolation: one document the LLM cannot handle must
            # not strand the rest of the batch behind it.
            logger.error("enrich-sweep: document %s failed: %s", document_id, exc)
            failed += 1
            push_enrich_metrics_task(document_id, succeeded=False)

    logger.info("enrich-sweep: %d processed, %d failed", enriched, failed)


@task(
    name="backfill-correspondent", retries=2, retry_delay_seconds=[60, 300], log_prints=True
)
def backfill_correspondent_task(
    document_id: int, declined_id: int, dry_run: bool = False
) -> enrich.EnrichResult:
    """Assign a correspondent to one already-enriched document, or mark it declined.

    The retries are what make the terminal marker safe: the Ollama query raises
    on failure rather than reporting "no correspondent", so a timeout gets
    re-asked instead of tagging the document `no-correspondent` for good.
    """
    logger = get_run_logger()
    paperless_url = os.environ["PAPERLESS_URL"]
    with enrich.open_client(_paperless_admin_token()) as client:
        result = enrich.backfill_correspondent(
            client, paperless_url, document_id, declined_id, dry_run=dry_run
        )

    enrich.append_result(result)
    logger.info(
        "backfill-correspondent %s complete in %.2fs: %s",
        document_id, result.duration_seconds, result.outcome,
    )
    return result


@flow(name="correspondent-backfill", log_prints=True)
def correspondent_backfill_flow(batch_size: int | None = None, dry_run: bool = False) -> None:
    """Assign correspondents to enriched documents that have none (#1373).

    Shaped like enrich_sweep_flow, over the complementary set: documents that
    no longer carry `queue` but have no correspondent — everything
    enriched before the #1366 fallback, plus every document the sweep's
    fallback has declined since. Each is asked once more; a decline here is
    terminal (`no-correspondent` tag), so the set drains instead of cycling.

    Same memory budget as the sweep (see ENRICH_SWEEP_BATCH_SIZE in the cluster
    manifest): its cron is offset from the sweep's so each batch runs against a
    freshly loaded model, and the `ollama` slot serializes any overlap.
    """
    logger = get_run_logger()
    flow_started = time.perf_counter()
    slot_acquired = False
    if batch_size is None:
        batch_size = int(os.environ.get("CORRESPONDENT_BACKFILL_BATCH_SIZE", "8"))

    try:
        with concurrency("correspondent-backfill", occupy=1, timeout_seconds=10):
            slot_acquired = True
            _run_backfill(batch_size, dry_run)
        logger.info(
            "correspondent-backfill complete in %.2fs", time.perf_counter() - flow_started
        )
    except TimeoutError:
        if not slot_acquired:
            logger.info("Skipped — correspondent backfill already running")
        else:
            raise


def _run_backfill(batch_size: int, dry_run: bool) -> None:
    logger = get_run_logger()
    paperless_url = os.environ["PAPERLESS_URL"]
    with enrich.open_client(_paperless_admin_token()) as client:
        queue_id = enrich.resolve_queue_tag(client, paperless_url)
        declined_id = enrich.resolve_declined_tag(client, paperless_url)
        document_ids = enrich.find_without_correspondent(
            client, paperless_url, queue_id, declined_id, batch_size
        )

    logger.info(
        "correspondent-backfill: %d document(s) without a correspondent, batch size %d%s",
        len(document_ids), batch_size, " (DRY RUN — nothing will be written)" if dry_run else "",
    )
    assigned = declined = failed = 0
    for document_id in document_ids:
        try:
            with concurrency("ollama", occupy=1):
                result = backfill_correspondent_task(document_id, declined_id, dry_run)
        except Exception as exc:
            # Per-document isolation, as in the sweep — and nothing is marked
            # on failure, so the next run asks again.
            logger.error("correspondent-backfill: document %s failed: %s", document_id, exc)
            failed += 1
            continue
        if result.correspondent:
            assigned += 1
        elif result.outcome != "already-has-correspondent":
            declined += 1

    logger.info(
        "correspondent-backfill: %d assigned, %d declined, %d failed", assigned, declined, failed
    )
    if document_ids and failed == len(document_ids):
        # Nothing here pushes a metric series, so a run that lost every
        # document must not finish Completed and look like progress.
        raise RuntimeError(f"correspondent-backfill: all {failed} document(s) failed")


@task(name="probe-paperless-health", log_prints=True)
def probe_paperless_health_task() -> paperless_health.TaskQueueHealth:
    return paperless_health.probe(os.environ["PAPERLESS_URL"], _paperless_admin_token())


@task(name="push-paperless-health-metrics", log_prints=True)
def push_paperless_health_metrics_task(health: paperless_health.TaskQueueHealth) -> None:
    metrics.push_paperless_health_metrics(health.failed, health.oldest_unfinished_seconds)


@flow(name="paperless-health", log_prints=True)
def paperless_health_flow() -> None:
    """Two reads of the tasks API and a push (homelab#1589).

    No concurrency slot and no retries, deliberately: a run that fails leaves
    `paperless_health_last_success_timestamp` where it was, and the
    PaperlessHealthProbeDead rule is what turns that into a page.
    """
    logger = get_run_logger()
    health = probe_paperless_health_task()
    push_paperless_health_metrics_task(health)
    logger.info(
        "paperless-health: %d unacknowledged failed consume task(s), oldest unfinished task %.0fs",
        health.failed, health.oldest_unfinished_seconds,
    )
