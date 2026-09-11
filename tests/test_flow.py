"""Tests for document_pipeline.flow — task wiring and concurrency coalescing."""

from __future__ import annotations

from email.message import EmailMessage
from unittest.mock import ANY, MagicMock, patch

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


def _mail_env(monkeypatch):
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless")
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "tok")
    monkeypatch.setenv("IMAP_PASSWORD", "secret")
    monkeypatch.setenv("WEBDAV_URL", "http://stalwart:8080/dav")
    monkeypatch.setenv("WEBDAV_USERNAME", "scanner")
    monkeypatch.setenv("WEBDAV_PASSWORD", "hunter2")
    monkeypatch.setenv("WEBDAV_SCAN_PATH", "/file/scanner@prins.id")


class _MailRun:
    """Every collaborator of the mail flow patched, for one run."""

    def __enter__(self):
        self._patches = [
            patch("document_pipeline.flow.imap_client"),
            patch("document_pipeline.flow.extract"),
            patch("document_pipeline.flow.webdav"),
            patch("document_pipeline.flow.prefect_client"),
            patch("document_pipeline.flow.metrics"),
            patch("document_pipeline.flow.concurrency"),
        ]
        self.imap, self.extract, self.webdav, self.prefect, self.metrics, self.concurrency = (
            p.__enter__() for p in self._patches
        )
        self.concurrency.return_value.__enter__.return_value = None
        self.concurrency.return_value.__exit__.return_value = False
        self.conn = MagicMock()
        self.imap.open_inbox.return_value.__enter__.return_value = self.conn
        self.imap.open_inbox.return_value.__exit__.return_value = False
        self.queue = self.webdav.WebDAVClient.return_value
        self.prefect.has_active_scan_run.return_value = False
        self.prefect.trigger_scan.return_value = True
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)
        return False


def test_mail_flow_queues_pdfs_flags_the_message_and_triggers_a_scan_run(monkeypatch):
    from document_pipeline.flow import mail_flow
    _mail_env(monkeypatch)
    msg = _message()

    with _MailRun() as run:
        run.imap.fetch_unprocessed.return_value = [(b"1", msg)]
        run.extract.queue_message_pdfs.return_value = 1

        mail_flow()

        run.queue.mkcol.assert_called_once_with("/file/scanner@prins.id/mail")
        run.extract.queue_message_pdfs.assert_called_once_with(msg, "1", run.queue, "/file/scanner@prins.id/mail")
        run.imap.mark_processed.assert_called_once_with(run.conn, b"1")
        # The hop into Paperless is the scan flow's — kicked in-process, as /trigger-scan does.
        run.prefect.trigger_scan.assert_called_once()
        run.metrics.push_run_metrics.assert_called_once_with(1, 1, ANY)


def test_mail_flow_does_not_trigger_a_scan_run_when_nothing_was_queued(monkeypatch):
    from document_pipeline.flow import mail_flow
    _mail_env(monkeypatch)

    with _MailRun() as run:
        run.imap.fetch_unprocessed.return_value = [(b"5", _message())]
        run.extract.queue_message_pdfs.return_value = 0  # no PDF

        mail_flow()

        run.imap.mark_processed.assert_called_once_with(run.conn, b"5")
        run.prefect.trigger_scan.assert_not_called()


def test_mail_flow_coalesces_onto_an_in_flight_scan_run(monkeypatch):
    from document_pipeline.flow import mail_flow
    _mail_env(monkeypatch)

    with _MailRun() as run:
        run.imap.fetch_unprocessed.return_value = [(b"1", _message())]
        run.extract.queue_message_pdfs.return_value = 1
        run.prefect.has_active_scan_run.return_value = True

        mail_flow()

        run.prefect.trigger_scan.assert_not_called()


def test_mail_flow_leaves_a_message_unflagged_when_its_put_fails_and_fails_the_run(monkeypatch):
    """A failed PUT must be retried next run — flagging would lose the PDF; a
    Completed run would hide that it happened. Other messages still go through."""
    from document_pipeline.flow import mail_flow
    _mail_env(monkeypatch)

    with _MailRun() as run:
        run.imap.fetch_unprocessed.return_value = [(b"1", _message()), (b"2", _message())]
        run.extract.queue_message_pdfs.side_effect = [ConnectionError("stalwart down"), 1]

        with pytest.raises(RuntimeError, match="unflagged"):
            mail_flow()

        run.imap.mark_processed.assert_called_once_with(run.conn, b"2")
        run.prefect.trigger_scan.assert_called_once()
        run.metrics.push_run_metrics.assert_not_called()


def test_mail_flow_flags_nothing_when_the_queue_directory_cannot_be_created(monkeypatch):
    from document_pipeline.flow import mail_flow
    _mail_env(monkeypatch)

    with _MailRun() as run:
        run.imap.fetch_unprocessed.return_value = [(b"1", _message())]
        run.queue.mkcol.side_effect = ConnectionError("stalwart down")

        with pytest.raises(ConnectionError):
            mail_flow()

        run.extract.queue_message_pdfs.assert_not_called()
        run.imap.mark_processed.assert_not_called()


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
        mock_extract.queue_message_pdfs.assert_not_called()
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


def test_enrich_sweep_skipped_when_a_previous_sweep_is_still_running(monkeypatch):
    """An hourly cron over a long batch overlaps itself; both runs would then
    query the same unenriched set and do every document twice."""
    _enrich_env(monkeypatch)
    from document_pipeline.flow import enrich_sweep_flow

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.enrich_document_task") as mock_task, \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.side_effect = TimeoutError
        mock_concurrency.return_value.__exit__.return_value = False

        enrich_sweep_flow(batch_size=3)

        mock_enrich.find_unenriched.assert_not_called()
        mock_task.assert_not_called()


def test_enrich_sweep_passes_dry_run_through_to_every_document(monkeypatch):
    _enrich_env(monkeypatch)
    from document_pipeline.flow import enrich_sweep_flow

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.enrich_document_task") as mock_task, \
         patch("document_pipeline.flow.metrics"), \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_enrich.resolve_marker_tag.return_value = 9
        mock_enrich.find_unenriched.return_value = [1, 2]
        mock_task.side_effect = [_result(1), _result(2)]

        enrich_sweep_flow(batch_size=2, dry_run=True)

        assert [c.args for c in mock_task.call_args_list] == [(1, 9, True), (2, 9, True)]


def test_enrich_sweep_is_live_by_default(monkeypatch):
    """The sweep's steady-state job is catching dropped triggers — defaulting it
    to dry-run would silently disable that (#1280)."""
    _enrich_env(monkeypatch)
    from document_pipeline.flow import enrich_sweep_flow

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.enrich_document_task") as mock_task, \
         patch("document_pipeline.flow.metrics"), \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_enrich.resolve_marker_tag.return_value = 9
        mock_enrich.find_unenriched.return_value = [1]
        mock_task.side_effect = [_result(1)]

        enrich_sweep_flow(batch_size=1)

        assert mock_task.call_args.args == (1, 9, False)


# --- correspondent backfill (#1373) ---

def _backfill_result(document_id: int, correspondent: str | None = "Symbox"):
    from document_pipeline.enrich import EnrichResult
    return EnrichResult(
        document_id=document_id,
        outcome="backfilled" if correspondent else "declined",
        correspondent=correspondent,
    )


def test_backfill_task_records_every_result_to_the_jsonl(monkeypatch):
    from document_pipeline.flow import backfill_correspondent_task
    _enrich_env(monkeypatch)
    monkeypatch.setenv("PAPERLESS_ADMIN_TOKEN", "superuser-tok")

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.get_run_logger"):
        mock_enrich.backfill_correspondent.return_value = _backfill_result(42)

        backfill_correspondent_task.fn(42, 11)

        mock_enrich.open_client.assert_called_once_with("superuser-tok")
        assert mock_enrich.backfill_correspondent.call_args.args[2:] == (42, 11)
        mock_enrich.append_result.assert_called_once()


def test_backfill_continues_past_a_failing_document_and_counts_outcomes(monkeypatch):
    _enrich_env(monkeypatch)
    from document_pipeline.flow import correspondent_backfill_flow

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.backfill_correspondent_task") as mock_task, \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_enrich.resolve_marker_tag.return_value = 9
        mock_enrich.resolve_declined_tag.return_value = 11
        mock_enrich.find_without_correspondent.return_value = [1, 2, 3]
        mock_task.side_effect = [
            _backfill_result(1), ValueError("boom"), _backfill_result(3, correspondent=None)
        ]

        correspondent_backfill_flow(batch_size=3)

        # One bad document must not strand the rest of the batch behind it.
        assert [c.args for c in mock_task.call_args_list] == [
            (1, 11, False), (2, 11, False), (3, 11, False)
        ]
        assert mock_enrich.find_without_correspondent.call_args.args[2:] == (9, 11, 3)


def test_backfill_batch_size_defaults_from_the_environment(monkeypatch):
    _enrich_env(monkeypatch)
    monkeypatch.setenv("CORRESPONDENT_BACKFILL_BATCH_SIZE", "5")
    from document_pipeline.flow import correspondent_backfill_flow

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.backfill_correspondent_task"), \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_enrich.find_without_correspondent.return_value = []

        correspondent_backfill_flow()

        assert mock_enrich.find_without_correspondent.call_args.args[-1] == 5


def test_backfill_fails_the_run_when_every_document_failed(monkeypatch):
    """Nothing here pushes a metric series, so a run that lost its whole batch
    must not finish Completed and read as progress."""
    _enrich_env(monkeypatch)
    from document_pipeline.flow import correspondent_backfill_flow

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.backfill_correspondent_task") as mock_task, \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_enrich.find_without_correspondent.return_value = [1, 2]
        mock_task.side_effect = ValueError("ollama down")

        with pytest.raises(RuntimeError):
            correspondent_backfill_flow(batch_size=2)

        assert mock_task.call_count == 2


def test_backfill_skipped_when_a_previous_run_is_still_running(monkeypatch):
    _enrich_env(monkeypatch)
    from document_pipeline.flow import correspondent_backfill_flow

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.backfill_correspondent_task") as mock_task, \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.side_effect = TimeoutError
        mock_concurrency.return_value.__exit__.return_value = False

        correspondent_backfill_flow(batch_size=3)

        mock_concurrency.assert_called_once_with(
            "correspondent-backfill", occupy=1, timeout_seconds=10
        )
        mock_enrich.find_without_correspondent.assert_not_called()
        mock_task.assert_not_called()


def test_backfill_passes_dry_run_through_to_every_document(monkeypatch):
    _enrich_env(monkeypatch)
    from document_pipeline.flow import correspondent_backfill_flow

    with patch("document_pipeline.flow.enrich") as mock_enrich, \
         patch("document_pipeline.flow.backfill_correspondent_task") as mock_task, \
         patch("document_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_enrich.resolve_declined_tag.return_value = 11
        mock_enrich.find_without_correspondent.return_value = [1, 2]
        mock_task.side_effect = [_backfill_result(1), _backfill_result(2)]

        correspondent_backfill_flow(batch_size=2, dry_run=True)

        assert [c.args for c in mock_task.call_args_list] == [(1, 11, True), (2, 11, True)]


def test_paperless_health_flow_probes_with_the_admin_token_and_pushes(monkeypatch):
    from document_pipeline.flow import paperless_health_flow
    from document_pipeline.paperless_health import TaskQueueHealth
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless")
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "ingest-tok")
    monkeypatch.setenv("PAPERLESS_ADMIN_TOKEN", "superuser-tok")

    with patch("document_pipeline.flow.paperless_health") as mock_probe, \
         patch("document_pipeline.flow.metrics") as mock_metrics:
        mock_probe.probe.return_value = TaskQueueHealth(failed=2, oldest_unfinished_seconds=901.0)

        paperless_health_flow()

    mock_probe.probe.assert_called_once_with("http://paperless", "superuser-tok")
    mock_metrics.push_paperless_health_metrics.assert_called_once_with(2, 901.0)
