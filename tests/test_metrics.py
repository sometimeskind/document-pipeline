"""Tests for document_pipeline.metrics — the per-source scan series (homelab#1590)
and the mail group's success/failure split (#60)."""

from __future__ import annotations

from unittest.mock import patch

from document_pipeline import metrics
from document_pipeline.scan import SourceResult


def _pushed(monkeypatch, sources):
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    monkeypatch.delenv("PREFECT_API_URL", raising=False)
    with patch("document_pipeline.metrics.push_to_gateway") as push:
        metrics.push_scan_metrics(sources, duration_seconds=1.5)
    assert push.call_args.kwargs["job"] == "scan-pipeline"
    return push.call_args.kwargs["registry"]


def test_scan_file_gauges_carry_one_series_per_source(monkeypatch):
    registry = _pushed(monkeypatch, {
        "scanner": SourceResult(ingested=2),
        "mail": SourceResult(failed=1, pending=1, oldest_pending_age_seconds=90.0),
    })

    assert registry.get_sample_value("scan_pipeline_files_ingested", {"source": "scanner"}) == 2
    assert registry.get_sample_value("scan_pipeline_files_failed", {"source": "mail"}) == 1
    assert registry.get_sample_value("scan_pipeline_files_pending", {"source": "mail"}) == 1
    assert registry.get_sample_value("scan_pipeline_oldest_pending_file_age_seconds", {"source": "mail"}) == 90.0


def test_a_drained_source_reads_zero_rather_than_disappearing(monkeypatch):
    """The alerts on these series must resolve, so the series must exist at 0."""
    registry = _pushed(monkeypatch, {"scanner": SourceResult(), "mail": SourceResult()})

    for source in ("scanner", "mail"):
        assert registry.get_sample_value("scan_pipeline_files_failed", {"source": source}) == 0
        assert registry.get_sample_value("scan_pipeline_oldest_pending_file_age_seconds", {"source": source}) == 0


def test_run_level_gauges_stay_unlabelled(monkeypatch):
    registry = _pushed(monkeypatch, {"scanner": SourceResult(), "mail": SourceResult()})

    assert registry.get_sample_value("scan_pipeline_run_duration_seconds") == 1.5
    assert registry.get_sample_value("scan_pipeline_last_success_timestamp") > 0


def _pushed_failure(monkeypatch, prefect_failures=None):
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    if prefect_failures is None:
        monkeypatch.delenv("PREFECT_API_URL", raising=False)
    else:
        monkeypatch.setenv("PREFECT_API_URL", "http://prefect:4200/api")
    with patch("document_pipeline.metrics.pushadd_to_gateway") as pushadd, \
         patch("document_pipeline.metrics._prefect_failures_24h", return_value=prefect_failures):
        metrics.push_failure_metrics()
    assert pushadd.call_args.kwargs["job"] == "mail-pipeline"
    return pushadd.call_args.kwargs["registry"]


def test_a_failed_run_publishes_a_failure_timestamp_and_no_success_timestamp(monkeypatch):
    """The pair is what tells "runs are failing" from "no new mail": a stale
    success next to a fresh failure, rather than metrics that simply stop."""
    registry = _pushed_failure(monkeypatch)

    assert registry.get_sample_value("document_pipeline_last_failure_timestamp") > 0
    assert registry.get_sample_value("document_pipeline_last_success_timestamp") is None


def test_the_failure_push_adds_to_the_group_rather_than_replacing_it(monkeypatch):
    """A PUT here would wipe `last_success_timestamp` — the very gauge that
    says how long the pipeline has been stalled."""
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    monkeypatch.delenv("PREFECT_API_URL", raising=False)

    with patch("document_pipeline.metrics.pushadd_to_gateway") as pushadd, \
         patch("document_pipeline.metrics.push_to_gateway") as push:
        metrics.push_failure_metrics()

    pushadd.assert_called_once()
    push.assert_not_called()


def test_the_failure_count_includes_the_run_that_is_pushing_it(monkeypatch):
    """That run is still Running, so the FAILED/CRASHED query cannot see it —
    without the increment the first failure of the day would publish 0."""
    registry = _pushed_failure(monkeypatch, prefect_failures=2)

    assert registry.get_sample_value("document_pipeline_prefect_failures_24h") == 3


def test_a_successful_run_publishes_the_queried_failure_count_unchanged(monkeypatch):
    """The success path keeps replacing the whole group with the queried value,
    which is also what clears the failure gauges once a run recovers."""
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    monkeypatch.setenv("PREFECT_API_URL", "http://prefect:4200/api")

    with patch("document_pipeline.metrics.push_to_gateway") as push, \
         patch("document_pipeline.metrics._prefect_failures_24h", return_value=2):
        metrics.push_run_metrics(3, 4, duration_seconds=1.5)

    registry = push.call_args.kwargs["registry"]
    assert push.call_args.kwargs["job"] == "mail-pipeline"
    assert registry.get_sample_value("document_pipeline_prefect_failures_24h") == 2
    assert registry.get_sample_value("document_pipeline_emails_synced") == 3
    assert registry.get_sample_value("document_pipeline_last_failure_timestamp") is None
