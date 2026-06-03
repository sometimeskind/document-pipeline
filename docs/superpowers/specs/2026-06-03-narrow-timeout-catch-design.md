# Narrow TimeoutError catch in mail_flow

**Issue:** #13

## Problem

`mail_flow` wraps both concurrency slot acquisition and the entire flow body in a single `try/except TimeoutError`. `imaplib.IMAP4_SSL` raises the same `TimeoutError` on TCP connection timeout, so IMAP failures are silently swallowed and runs report `Completed` instead of `Failed`.

## Design

Add a `slot_acquired: bool = False` flag. Flip it to `True` as the first statement inside the `with concurrency(...)` block. In the `except TimeoutError` handler, re-raise if `slot_acquired` is `True` — meaning the error came from within the flow body, not from slot acquisition.

```python
slot_acquired = False
try:
    with concurrency("mail-pipeline", occupy=1, timeout_seconds=10):
        slot_acquired = True
        messages_processed, pdfs_submitted = process_mail_task()
        push_metrics_task(...)
    logger.info(...)
except TimeoutError:
    if not slot_acquired:
        logger.info("Skipped — mail pipeline already running")
    else:
        raise
```

## Testing

- Existing test `test_mail_flow_skipped_when_pipeline_busy` covers the `slot_acquired = False` path (TimeoutError on `__enter__`).
- New test: mock `concurrency` to succeed (enter returns normally), mock `imap_client.open_inbox` to raise `TimeoutError`, assert the flow raises rather than swallowing.
