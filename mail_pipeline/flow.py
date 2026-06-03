"""Prefect tasks and flow for the mail pipeline."""

from __future__ import annotations

import os
import time

from prefect import flow, get_run_logger, task
from prefect.concurrency.sync import concurrency

from mail_pipeline import extract, imap_client, metrics


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
