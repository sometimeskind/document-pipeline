"""Tests for mail_pipeline.imap_client — IMAP fetch and flag operations."""

from __future__ import annotations

from email.message import EmailMessage
from unittest.mock import MagicMock, call, patch

from mail_pipeline import imap_client

_PROCESSED = "$Processed"


def _make_conn(uids: list[bytes] = (), messages: dict[bytes, bytes] | None = None) -> MagicMock:
    """Return a mock IMAP connection with canned SEARCH and FETCH responses."""
    conn = MagicMock()
    msg_map = messages or {}

    def handle_uid(command, *args):
        if command == "SEARCH":
            return ("OK", [b" ".join(uids) if uids else b""])
        if command == "FETCH":
            raw = msg_map.get(args[0], b"")
            return ("OK", [(b"header", raw)])
        return ("OK", [None])

    conn.uid.side_effect = handle_uid
    return conn


def _raw_message(subject: str = "Test") -> bytes:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg.set_content("body")
    return bytes(msg)


def test_fetch_unprocessed_searches_unkeyword_processed():
    conn = _make_conn()
    imap_client.fetch_unprocessed(conn)
    conn.uid.assert_any_call("SEARCH", f"UNKEYWORD {_PROCESSED}")


def test_fetch_unprocessed_returns_empty_when_no_messages():
    conn = _make_conn(uids=[])
    assert imap_client.fetch_unprocessed(conn) == []


def test_fetch_unprocessed_returns_parsed_messages():
    raw = _raw_message("Hello")
    conn = _make_conn(uids=[b"7"], messages={b"7": raw})

    result = imap_client.fetch_unprocessed(conn)

    assert len(result) == 1
    uid, msg = result[0]
    assert uid == b"7"
    assert msg["Subject"] == "Hello"


def test_mark_processed_stores_processed_keyword():
    conn = MagicMock()
    imap_client.mark_processed(conn, b"7")
    conn.uid.assert_called_once_with("STORE", b"7", "+FLAGS", f"({_PROCESSED})")


def test_open_inbox_connects_logs_out(monkeypatch):
    monkeypatch.setenv("IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("IMAP_PORT", "993")
    monkeypatch.setenv("IMAP_USER", "user@example.com")
    monkeypatch.setenv("IMAP_PASSWORD", "secret")

    mock_conn = MagicMock()
    with patch("mail_pipeline.imap_client.imaplib.IMAP4_SSL", return_value=mock_conn):
        with imap_client.open_inbox() as conn:
            assert conn is mock_conn

    mock_conn.login.assert_called_once_with("user@example.com", "secret")
    mock_conn.select.assert_called_once_with("INBOX")
    mock_conn.logout.assert_called_once()
