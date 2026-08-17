"""Ingest scanned documents from a WebDAV share into Paperless.

The scanner (a Brother MFC) writes straight to a WebDAV home over the LAN; this
module drains that directory. The contract that makes it safe to delete the
only copy of a document: a 2xx from `post_document` means *queued*, not
*ingested*, so every file is followed through to a terminal Paperless task
state and only removed once that state says the document actually landed.
"""

from __future__ import annotations

import datetime
import logging
import mimetypes
import time
from dataclasses import dataclass

import httpx

from mail_pipeline.webdav import WebDAVClient, WebDAVEntry

logger = logging.getLogger(__name__)

# Formats the printer can emit. Anything else is left untouched — and, crucially,
# excluded from the pending-file metric, so an unrelated file dropped into the
# share can never hold the staleness alert on forever.
SCAN_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff")

DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_POLL_TIMEOUT_SECONDS = 300.0

# PaperlessTask.COMPLETE_STATUSES — the states the consume task stops moving in.
_TERMINAL_STATUSES = frozenset({"success", "failure", "revoked"})


@dataclass
class ScanResult:
    """Outcome of one drain of the scan directory."""

    ingested: int = 0
    failed: int = 0
    ignored: int = 0
    pending: int = 0
    oldest_pending_age_seconds: float = 0.0


def is_eligible(name: str) -> bool:
    """True for files the pipeline should try to ingest.

    Excludes dotfiles, which is how partial uploads and editor droppings show
    up, and anything outside the printer's own output formats.
    """
    return not name.startswith(".") and name.lower().endswith(SCAN_EXTENSIONS)


def ingest_scans(
    webdav: WebDAVClient,
    scan_path: str,
    paperless_url: str,
    paperless_token: str,
    tag: str = "scanner",
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_timeout: float = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> ScanResult:
    """Ingest every eligible scan under `scan_path`, deleting each on success."""
    entries = webdav.list(scan_path)
    files = [e for e in entries if not e.is_collection]
    eligible = [e for e in files if is_eligible(e.name)]

    for collection in (e for e in entries if e.is_collection):
        # Depth-1 only: some printer firmwares create date subfolders. Surface
        # them rather than silently descending into an unknown layout.
        logger.warning("Ignoring subdirectory %r under %s — scans there are not ingested", collection.name, scan_path)

    result = ScanResult(ignored=len(files) - len(eligible))
    if result.ignored:
        logger.info("Ignoring %d non-scan file(s) in %s", result.ignored, scan_path)
    if not eligible:
        logger.info("No eligible scans in %s", scan_path)
        return result

    remaining: list[WebDAVEntry] = []
    with httpx.Client(headers={"Authorization": f"Token {paperless_token}"}, timeout=60.0) as client:
        tag_ids = [_resolve_tag(client, paperless_url, tag)] if tag else []
        for entry in eligible:
            try:
                if not _ingest_one(entry, webdav, client, paperless_url, tag_ids, poll_interval, poll_timeout):
                    result.failed += 1
                    remaining.append(entry)
                    continue
                result.ingested += 1
            except Exception as exc:
                # Per-file isolation: one malformed scan must never strand the
                # rest of the batch behind it.
                logger.error("Failed to ingest %r: %s", entry.name, exc)
                result.failed += 1
                remaining.append(entry)

    result.pending = len(remaining)
    result.oldest_pending_age_seconds = _oldest_age_seconds(remaining)
    logger.info(
        "Scan ingest complete: %d ingested, %d failed, %d ignored", result.ingested, result.failed, result.ignored
    )
    return result


def _ingest_one(
    entry: WebDAVEntry,
    webdav: WebDAVClient,
    client: httpx.Client,
    paperless_url: str,
    tag_ids: list[int],
    poll_interval: float,
    poll_timeout: float,
) -> bool:
    """Submit one scan and delete it once Paperless confirms ingestion."""
    payload = webdav.get(entry.href)
    if payload is None:
        # Another run got there first. Expected under the sweep/trigger overlap
        # and deliberately not a failure — nothing is lost.
        logger.info("%r vanished before download, skipping", entry.name)
        return True
    if not payload:
        logger.warning("%r is empty, leaving it in place", entry.name)
        return False

    task_id = _post_document(client, paperless_url, entry.name, payload, tag_ids)
    logger.info("Submitted %r (%d bytes) to Paperless as task %s", entry.name, len(payload), task_id)

    status = _await_task(client, paperless_url, task_id, poll_interval, poll_timeout)
    if status not in ("success", "duplicate"):
        # Left in place on purpose: the next sweep retries it, and the staleness
        # alert fires if it never clears. Deleting here would destroy the only copy.
        logger.error("Paperless task %s for %r ended as %s — leaving file in place", task_id, entry.name, status)
        return False

    webdav.delete(entry.href)
    logger.info("Ingested %r (task %s: %s), removed from WebDAV", entry.name, task_id, status)
    return True


def _resolve_tag(client: httpx.Client, paperless_url: str, name: str) -> int:
    """Return the id of the named tag, creating it if it does not exist."""
    resp = client.get(f"{paperless_url}/api/tags/", params={"name__iexact": name})
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if results:
        return results[0]["id"]

    resp = client.post(f"{paperless_url}/api/tags/", json={"name": name})
    resp.raise_for_status()
    tag_id = resp.json()["id"]
    logger.info("Created Paperless tag %r (id %s)", name, tag_id)
    return tag_id


def _post_document(
    client: httpx.Client, paperless_url: str, filename: str, payload: bytes, tag_ids: list[int]
) -> str:
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    resp = client.post(
        f"{paperless_url}/api/documents/post_document/",
        files={"document": (filename, payload, content_type)},
        # Repeated `tags` parts, one per id — the shape post_document expects.
        data={"tags": [str(t) for t in tag_ids]},
    )
    resp.raise_for_status()
    body = resp.json()
    # Paperless returns the consume task UUID, historically as a bare JSON
    # string and as {"task_id": ...} in current releases.
    return str(body["task_id"] if isinstance(body, dict) else body)


def _await_task(
    client: httpx.Client, paperless_url: str, task_id: str, poll_interval: float, poll_timeout: float
) -> str:
    """Poll a consume task to a terminal state.

    Returns `success`, `duplicate`, `failure`, `revoked`, or `timeout`.
    """
    deadline = time.monotonic() + poll_timeout
    while True:
        task = _read_task(client, paperless_url, task_id)
        status = str((task or {}).get("status") or "").lower()
        if status in _TERMINAL_STATUSES:
            # A re-POST after a poll timeout hits PAPERLESS_CONSUMER_DELETE_DUPLICATES,
            # which fails the task rather than succeeding it. The document is in
            # Paperless either way, so this counts as ingested — otherwise the
            # file would sit in WebDAV forever, failing on every future sweep.
            if status == "failure" and _is_duplicate(task or {}):
                return "duplicate"
            return status
        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(poll_interval)


def _read_task(client: httpx.Client, paperless_url: str, task_id: str) -> dict | None:
    resp = client.get(f"{paperless_url}/api/tasks/", params={"task_id": task_id})
    resp.raise_for_status()
    body = resp.json()
    tasks = body.get("results", []) if isinstance(body, dict) else body
    return tasks[0] if tasks else None


def _is_duplicate(task: dict) -> bool:
    result_data = task.get("result_data")
    if isinstance(result_data, dict) and result_data.get("duplicate_of"):
        return True
    return "duplicate" in str(task.get("result") or "").lower()


def _oldest_age_seconds(entries: list[WebDAVEntry]) -> float:
    """Age of the oldest entry, or 0.0 when there are none or no timestamps.

    Reported as an age rather than a timestamp so the metric self-clears: an
    empty directory pushes 0 and the staleness alert resolves on its own.
    """
    timestamps = [e.last_modified for e in entries if e.last_modified]
    if not timestamps:
        return 0.0
    now = datetime.datetime.now(datetime.timezone.utc)
    return max((now - ts).total_seconds() for ts in timestamps)
