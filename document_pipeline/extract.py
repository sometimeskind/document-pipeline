"""Extract PDF attachments from mail messages into the WebDAV scan queue.

The queue is drained by `scan.ingest_scans`, which is what follows each file to
a terminal Paperless task state — the mail flow itself never talks to Paperless
(homelab#1590)."""

from __future__ import annotations

import logging
import re
import unicodedata
from email.header import decode_header, make_header
from email.message import Message

from document_pipeline.webdav import WebDAVClient

logger = logging.getLogger(__name__)

# Paperless derives the title from the filename stem and Document.title is
# max_length=128, so keep the stem comfortably under that.
_MAX_STEM_CHARS = 120
# Senders that fumble RFC 2231 wrap the whole `charset''value` in an RFC 2047
# word, so the charset tag survives decoding as literal text (e.g.
# "utf-8''Leistungsübersicht.pdf"). Observed on 4 of the 47 documents this bug
# produced.
_RFC2231_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9_.:+-]*''")


def queue_message_pdfs(msg: Message, uid: str, queue: WebDAVClient, queue_path: str) -> int:
    """PUT every PDF attachment of `msg` into the scan queue. Returns how many.

    Objects are named `<uid>-<filename>` so a message retried after a failed
    PUT overwrites its own earlier objects rather than queueing duplicates.
    Nothing is caught here on purpose: a failed PUT must propagate so the
    caller leaves the message unflagged for the next run.
    """
    count = 0
    names: set[str] = set()
    for part in msg.walk():
        if part.get_content_type() != "application/pdf":
            continue
        filename = _attachment_filename(part)
        payload = part.get_payload(decode=True)
        if not payload:
            logger.warning("  PDF part %r had empty payload, skipping", filename)
            continue

        name = _unique(f"{uid}-{filename}", names)
        names.add(name)
        queue.put(f"{queue_path}/{name}", payload)
        logger.info("  -> queued PDF %r (%d bytes) as %r", filename, len(payload), name)
        count += 1

    return count


def _unique(name: str, taken: set[str]) -> str:
    """Suffix the stem when a message carries two attachments with one name."""
    if name not in taken:
        return name
    stem, _, ext = name.rpartition(".")
    n = 2
    while f"{stem}-{n}.{ext}" in taken:
        n += 1
    return f"{stem}-{n}.{ext}"


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
    """Make a decoded filename safe as a WebDAV object name and a Paperless filename."""
    # Sender-controlled text now heading for a URL path and a filesystem.
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
