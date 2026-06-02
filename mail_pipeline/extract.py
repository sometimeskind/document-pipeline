"""Extract PDF attachments from mail messages and submit them to Paperless."""

from __future__ import annotations

import logging
import time
from email.message import Message

import httpx

logger = logging.getLogger(__name__)


def submit_message_pdfs(msg: Message, paperless_url: str, paperless_token: str) -> bool:
    """Submit PDF attachments from msg to Paperless. Returns True if any were submitted."""
    with httpx.Client(
        headers={"Authorization": f"Token {paperless_token}"},
        timeout=30.0,
    ) as client:
        count, _ = _submit_pdfs(msg, client, paperless_url)
    return count > 0


def _submit_pdfs(msg: Message, client: httpx.Client, paperless_url: str) -> tuple[int, int]:
    """Return (number of PDFs submitted, total bytes)."""
    count = 0
    total = 0
    for part in msg.walk():
        if part.get_content_type() != "application/pdf":
            continue
        filename = part.get_filename() or "attachment.pdf"
        payload = part.get_payload(decode=True)
        if not payload:
            logger.warning("  PDF part %r had empty payload, skipping", filename)
            continue

        size = len(payload)
        logger.info("  -> submitting PDF %r (%s) to Paperless", filename, _human_size(size))
        started = time.perf_counter()
        resp = client.post(
            f"{paperless_url}/api/documents/post_document/",
            files={"document": (filename, payload, "application/pdf")},
        )
        resp.raise_for_status()
        logger.info(
            "     paperless accepted %r: HTTP %d in %.2fs",
            filename, resp.status_code, time.perf_counter() - started,
        )
        count += 1
        total += size

    return count, total


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"
