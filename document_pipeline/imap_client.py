"""IMAP client for fetching and flagging mail from maddy."""

from __future__ import annotations

import email
import imaplib
import os
from contextlib import contextmanager
from email.message import Message
from typing import Iterator

_PROCESSED_KEYWORD = "$Processed"


@contextmanager
def open_inbox() -> Iterator[imaplib.IMAP4_SSL]:
    host = os.environ.get("IMAP_HOST", "maddy.mail.svc.cluster.local")
    port = int(os.environ.get("IMAP_PORT", "993"))
    user = os.environ.get("IMAP_USER", "tom@prins.id")
    password = os.environ["IMAP_PASSWORD"]
    conn = imaplib.IMAP4_SSL(host, port)
    conn.login(user, password)
    conn.select("INBOX")
    try:
        yield conn
    finally:
        conn.logout()


def fetch_unprocessed(conn: imaplib.IMAP4_SSL) -> list[tuple[bytes, Message]]:
    """Return (uid, message) pairs for messages not yet marked $Processed."""
    _, data = conn.uid("SEARCH", f"UNKEYWORD {_PROCESSED_KEYWORD}")
    uid_list = data[0].split() if data[0] else []
    messages = []
    for uid in uid_list:
        _, msg_data = conn.uid("FETCH", uid, "(RFC822)")
        raw = msg_data[0][1]
        messages.append((uid, email.message_from_bytes(raw)))
    return messages


def mark_processed(conn: imaplib.IMAP4_SSL, uid: bytes) -> None:
    """Flag a message with the $Processed keyword."""
    conn.uid("STORE", uid, "+FLAGS", f"({_PROCESSED_KEYWORD})")
