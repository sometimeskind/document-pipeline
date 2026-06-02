"""Tests for mail_pipeline.flow — task wiring and concurrency coalescing."""

from __future__ import annotations

from email.message import EmailMessage
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True, scope="module")
def prefect_test_env():
    from prefect.testing.utilities import prefect_test_harness
    with prefect_test_harness():
        yield


def _message(subject: str = "Test") -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg.set_content("body")
    return msg


def test_mail_flow_processes_messages_and_pushes_metrics(monkeypatch):
    from mail_pipeline.flow import mail_flow
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless")
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "tok")
    monkeypatch.setenv("IMAP_PASSWORD", "secret")

    mock_conn = MagicMock()
    msg = _message()

    with patch("mail_pipeline.flow.imap_client") as mock_imap, \
         patch("mail_pipeline.flow.extract") as mock_extract, \
         patch("mail_pipeline.flow.metrics") as mock_metrics, \
         patch("mail_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_imap.open_inbox.return_value.__enter__.return_value = mock_conn
        mock_imap.open_inbox.return_value.__exit__.return_value = False
        mock_imap.fetch_unprocessed.return_value = [(b"1", msg)]
        mock_extract.submit_message_pdfs.return_value = True

        mail_flow()

    mock_extract.submit_message_pdfs.assert_called_once()
    mock_imap.mark_processed.assert_called_once_with(mock_conn, b"1")
    mock_metrics.push_run_metrics.assert_called_once()


def test_mail_flow_marks_message_processed_even_without_pdf(monkeypatch):
    from mail_pipeline.flow import mail_flow
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless")
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "tok")
    monkeypatch.setenv("IMAP_PASSWORD", "secret")

    mock_conn = MagicMock()

    with patch("mail_pipeline.flow.imap_client") as mock_imap, \
         patch("mail_pipeline.flow.extract") as mock_extract, \
         patch("mail_pipeline.flow.metrics"), \
         patch("mail_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_imap.open_inbox.return_value.__enter__.return_value = mock_conn
        mock_imap.open_inbox.return_value.__exit__.return_value = False
        mock_imap.fetch_unprocessed.return_value = [(b"5", _message())]
        mock_extract.submit_message_pdfs.return_value = False  # no PDF

        mail_flow()

    mock_imap.mark_processed.assert_called_once_with(mock_conn, b"5")


def test_mail_flow_skipped_when_pipeline_busy():
    from mail_pipeline.flow import mail_flow
    with patch("mail_pipeline.flow.concurrency") as mock_concurrency, \
         patch("mail_pipeline.flow.imap_client") as mock_imap, \
         patch("mail_pipeline.flow.extract") as mock_extract, \
         patch("mail_pipeline.flow.metrics") as mock_metrics:
        mock_concurrency.return_value.__enter__.side_effect = TimeoutError
        mock_concurrency.return_value.__exit__.return_value = False

        mail_flow()

        mock_imap.fetch_unprocessed.assert_not_called()
        mock_extract.submit_message_pdfs.assert_not_called()
        mock_metrics.push_run_metrics.assert_not_called()
