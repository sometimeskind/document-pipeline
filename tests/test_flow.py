"""Tests for document_pipeline.flow — task wiring and concurrency coalescing."""

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
    from document_pipeline.flow import mail_flow
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless")
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "tok")
    monkeypatch.setenv("IMAP_PASSWORD", "secret")

    mock_conn = MagicMock()
    msg = _message()

    with patch("document_pipeline.flow.imap_client") as mock_imap, \
         patch("document_pipeline.flow.extract") as mock_extract, \
         patch("document_pipeline.flow.metrics") as mock_metrics, \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
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
    from document_pipeline.flow import mail_flow
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless")
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "tok")
    monkeypatch.setenv("IMAP_PASSWORD", "secret")

    mock_conn = MagicMock()

    with patch("document_pipeline.flow.imap_client") as mock_imap, \
         patch("document_pipeline.flow.extract") as mock_extract, \
         patch("document_pipeline.flow.metrics"), \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_imap.open_inbox.return_value.__enter__.return_value = mock_conn
        mock_imap.open_inbox.return_value.__exit__.return_value = False
        mock_imap.fetch_unprocessed.return_value = [(b"5", _message())]
        mock_extract.submit_message_pdfs.return_value = False  # no PDF

        mail_flow()

    mock_imap.mark_processed.assert_called_once_with(mock_conn, b"5")


def test_mail_flow_propagates_imap_timeout():
    """An IMAP connection timeout inside the flow body must not be silently swallowed."""
    from document_pipeline.flow import mail_flow
    import pytest
    with patch("document_pipeline.flow.concurrency") as mock_concurrency, \
         patch("document_pipeline.flow.imap_client") as mock_imap, \
         patch("document_pipeline.flow.extract"), \
         patch("document_pipeline.flow.metrics"):
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_imap.open_inbox.return_value.__enter__.side_effect = TimeoutError(110, "Connection timed out")

        with pytest.raises(TimeoutError):
            mail_flow()


def test_mail_flow_skipped_when_pipeline_busy():
    from document_pipeline.flow import mail_flow
    with patch("document_pipeline.flow.concurrency") as mock_concurrency, \
         patch("document_pipeline.flow.imap_client") as mock_imap, \
         patch("document_pipeline.flow.extract") as mock_extract, \
         patch("document_pipeline.flow.metrics") as mock_metrics:
        mock_concurrency.return_value.__enter__.side_effect = TimeoutError
        mock_concurrency.return_value.__exit__.return_value = False

        mail_flow()

        mock_imap.fetch_unprocessed.assert_not_called()
        mock_extract.submit_message_pdfs.assert_not_called()
        mock_metrics.push_run_metrics.assert_not_called()


def _enrich_env(monkeypatch):
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless")
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "ingest-tok")


def _result(document_id: int = 42, outcome: str = "enriched"):
    from document_pipeline.enrich import EnrichResult
    return EnrichResult(document_id=document_id, outcome=outcome)


# The enrich task carries real retry delays (60s, 300s), so these exercise its
# body via `.fn` and patch the task itself when driving the flow — otherwise a
# failure test would sit out the backoff.

def test_enrich_task_records_every_result_to_the_jsonl(monkeypatch):
    from document_pipeline.flow import enrich_document_task
    _enrich_env(monkeypatch)

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.get_run_logger"):
        mock_enrich.resolve_marker_tag.return_value = 9
        mock_enrich.enrich_document.return_value = _result()

        enrich_document_task.fn(42)

        mock_enrich.enrich_document.assert_called_once()
        assert mock_enrich.enrich_document.call_args.args[2:] == (42, 9)
        mock_enrich.append_result.assert_called_once()


def test_enrich_task_prefers_the_admin_token_over_the_ingest_token(monkeypatch):
    from document_pipeline.flow import enrich_document_task
    _enrich_env(monkeypatch)
    monkeypatch.setenv("PAPERLESS_ADMIN_TOKEN", "superuser-tok")

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.get_run_logger"):
        mock_enrich.enrich_document.return_value = _result()
        enrich_document_task.fn(42, marker_id=9)
        mock_enrich.open_client.assert_called_once_with("superuser-tok")


def test_enrich_task_falls_back_to_the_ingest_token_when_unset(monkeypatch):
    """A Renovate digest bump can land before the manifest that configures it."""
    from document_pipeline.flow import enrich_document_task
    _enrich_env(monkeypatch)
    monkeypatch.delenv("PAPERLESS_ADMIN_TOKEN", raising=False)

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.get_run_logger"):
        mock_enrich.enrich_document.return_value = _result()
        enrich_document_task.fn(42, marker_id=9)
        mock_enrich.open_client.assert_called_once_with("ingest-tok")


def test_enrich_flow_queues_on_the_ollama_slot_rather_than_skipping(monkeypatch):
    _enrich_env(monkeypatch)
    from document_pipeline.flow import enrich_flow

    with patch("document_pipeline.flow.enrich_document_task", return_value=_result()), \
         patch("document_pipeline.flow.metrics") as mock_metrics, \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False

        enrich_flow(42)

        # No `timeout_seconds`, unlike mail/scan: per-document work, so a busy
        # slot must queue. A skip would silently lose this document.
        mock_concurrency.assert_called_once_with("ollama", occupy=1)
        mock_metrics.push_enrich_metrics.assert_called_once_with(42, True)


def test_enrich_flow_pushes_neither_series_for_a_skip(monkeypatch):
    _enrich_env(monkeypatch)
    from document_pipeline.flow import enrich_flow

    with patch("document_pipeline.flow.enrich_document_task",
               return_value=_result(outcome="skipped-short-content")), \
         patch("document_pipeline.flow.metrics") as mock_metrics, \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False

        enrich_flow(42)

        mock_metrics.push_enrich_metrics.assert_not_called()


def test_enrich_flow_pushes_the_failure_series_and_reraises(monkeypatch):
    """PaperlessAutoTitleFailing latches on this series — it must still fire."""
    _enrich_env(monkeypatch)
    from document_pipeline.flow import enrich_flow

    with patch("document_pipeline.flow.enrich_document_task", side_effect=ValueError("boom")), \
         patch("document_pipeline.flow.metrics") as mock_metrics, \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False

        with pytest.raises(ValueError):
            enrich_flow(42)

        mock_metrics.push_enrich_metrics.assert_called_once_with(42, False)


def test_enrich_sweep_continues_past_a_failing_document(monkeypatch):
    _enrich_env(monkeypatch)
    from document_pipeline.flow import enrich_sweep_flow

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.enrich_document_task") as mock_task, \
         patch("document_pipeline.flow.metrics") as mock_metrics, \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_enrich.resolve_marker_tag.return_value = 9
        mock_enrich.find_unenriched.return_value = [1, 2, 3]
        mock_task.side_effect = [_result(1), ValueError("boom"), _result(3)]

        enrich_sweep_flow(batch_size=3)

        # One bad document must not strand the rest of the batch behind it.
        assert mock_task.call_count == 3
        assert mock_enrich.find_unenriched.call_args.args[-1] == 3
        mock_metrics.push_enrich_metrics.assert_called_once_with(2, False)


def test_enrich_sweep_batch_size_defaults_from_the_environment(monkeypatch):
    _enrich_env(monkeypatch)
    monkeypatch.setenv("ENRICH_SWEEP_BATCH_SIZE", "7")
    from document_pipeline.flow import enrich_sweep_flow

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.enrich_document_task"), \
         patch("document_pipeline.flow.metrics"), \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_enrich.find_unenriched.return_value = []

        enrich_sweep_flow()

        assert mock_enrich.find_unenriched.call_args.args[-1] == 7
