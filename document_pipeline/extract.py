"""Extract PDF attachments from mail messages and submit them to Paperless."""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from email.header import decode_header, make_header
from email.message import Message

import httpx

logger = logging.getLogger(__name__)

# Paperless derives the title from the filename stem and Document.title is
# max_length=128, so keep the stem comfortably under that.
_MAX_STEM_CHARS = 120
# Senders that fumble RFC 2231 wrap the whole `charset''value` in an RFC 2047
# word, so the charset tag survives decoding as literal text (e.g.
# "utf-8''Leistungsübersicht.pdf"). Observed on 4 of the 47 documents this bug
# produced.
_RFC2231_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9_.:+-]*''")


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
        filename = _attachment_filename(part)
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


def _attachment_filename(part: Message) -> str:
    """Decoded, sanitised filename for a PDF part."""
    raw = part.get_filename()
    if not raw:
        return "attachment.pdf"
    try:
        # get_filename() collapses RFC 2231 continuations but leaves RFC 2047
        # encoded-words verbatim — illegal inside a Content-Disposition
        # parameter, emitted by plenty of MUAs anyway (#1297).
        decoded = str(make_header(decode_header(raw)))
    except Exception:
        # LookupError on an unknown charset, HeaderParseError on bad base64.
        # A mangled name is worth far more than a dropped document.
        logger.warning("  could not decode attachment filename %r, using it as-is", raw)
        decoded = raw
    return _sanitise(_RFC2231_PREFIX.sub("", decoded))


def _sanitise(name: str) -> str:
    """Make a decoded filename safe for the multipart filename field."""
    # Sender-controlled text now heading for a form field and a filesystem.
    name = name.replace("/", "_").replace("\\", "_")
    # Category C* is control/format/surrogate/unassigned — nothing that belongs
    # in a filename, and CR/LF would otherwise ride into a multipart header.
    name = "".join(c for c in name if unicodedata.category(c)[0] != "C")
    name = name.strip(" .")
    if not name:
        return "attachment.pdf"
    stem, dot, ext = name.rpartition(".")
    if not dot or ext.lower() != "pdf":
        stem, ext = name, "pdf"
    return f"{stem[:_MAX_STEM_CHARS]}.{ext}"


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"
