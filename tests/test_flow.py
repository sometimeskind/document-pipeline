"""Tests for mail_pipeline.flow — task wiring and concurrency coalescing."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True, scope="module")
def prefect_test_env():
    from prefect.testing.utilities import prefect_test_harness
    with prefect_test_harness():
        yield


def test_mail_flow_runs_all_tasks_in_order(monkeypatch):
    from mail_pipeline.flow import mail_flow
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless")
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "tok")

    call_order: list[str] = []

    with patch("mail_pipeline.flow.mbsync") as mock_mbsync, \
         patch("mail_pipeline.flow.notmuch") as mock_notmuch, \
         patch("mail_pipeline.flow.extract") as mock_extract, \
         patch("mail_pipeline.flow.metrics") as mock_metrics, \
         patch("mail_pipeline.flow.concurrency") as mock_concurrency:
        mock_concurrency.return_value.__enter__.return_value = None
        mock_concurrency.return_value.__exit__.return_value = False
        mock_mbsync.run_mbsync.side_effect = lambda *a, **kw: call_order.append("sync")
        mock_notmuch.index_mail.side_effect = lambda *a, **kw: (call_order.append("index"), 3)[1]
        mock_extract.extract_pdfs.side_effect = lambda **kw: (call_order.append("extract"), 1)[1]
        mock_metrics.push_run_metrics.side_effect = lambda *a, **kw: call_order.append("push")

        mail_flow()

    assert call_order == ["sync", "index", "extract", "push"]


def test_mail_flow_skipped_when_pipeline_busy():
    from mail_pipeline.flow import mail_flow
    with patch("mail_pipeline.flow.concurrency") as mock_concurrency, \
         patch("mail_pipeline.flow.mbsync") as mock_mbsync, \
         patch("mail_pipeline.flow.notmuch") as mock_notmuch, \
         patch("mail_pipeline.flow.extract") as mock_extract, \
         patch("mail_pipeline.flow.metrics") as mock_metrics:
        mock_concurrency.return_value.__enter__.side_effect = TimeoutError
        mock_concurrency.return_value.__exit__.return_value = False

        mail_flow()

        mock_mbsync.run_mbsync.assert_not_called()
        mock_notmuch.index_mail.assert_not_called()
        mock_extract.extract_pdfs.assert_not_called()
        mock_metrics.push_run_metrics.assert_not_called()
