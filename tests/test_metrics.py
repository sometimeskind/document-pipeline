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


def _pushed_failure(monkeypatch):
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    with patch("document_pipeline.metrics.pushadd_to_gateway") as pushadd:
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

    with patch("document_pipeline.metrics.pushadd_to_gateway") as pushadd, \
         patch("document_pipeline.metrics.push_to_gateway") as push:
        metrics.push_failure_metrics()

    pushadd.assert_called_once()
    push.assert_not_called()


def test_a_successful_run_replaces_the_group_without_a_failure_timestamp(monkeypatch):
    """The success path replaces the whole group, which is what clears the
    failure gauge once a run recovers. Failed-run counts are the Prefect
    exporter's job (homelab#1838), so the mail group carries none and the
    push makes no Prefect API call."""
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    monkeypatch.setenv("PREFECT_API_URL", "http://prefect:4200/api")

    with patch("document_pipeline.metrics.push_to_gateway") as push, \
         patch("document_pipeline.metrics._prefect_failures_24h") as query:
        metrics.push_run_metrics(3, 4, duration_seconds=1.5)

    registry = push.call_args.kwargs["registry"]
    assert push.call_args.kwargs["job"] == "mail-pipeline"
    assert registry.get_sample_value("document_pipeline_prefect_failures_24h") is None
    query.assert_not_called()
    assert registry.get_sample_value("document_pipeline_emails_synced") == 3
    assert registry.get_sample_value("document_pipeline_last_failure_timestamp") is None


def _pushed_health(monkeypatch, **counts):
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    with patch("document_pipeline.metrics.push_to_gateway") as push:
        metrics.push_paperless_health_metrics(1, 30.0, **counts)
    assert push.call_args.kwargs["job"] == "paperless-health"
    return push.call_args.kwargs["registry"]


def test_health_push_carries_the_24h_document_counts(monkeypatch):
    registry = _pushed_health(monkeypatch, consumed_24h=5, enriched_24h=0)

    assert registry.get_sample_value("paperless_tasks_failed") == 1
    assert registry.get_sample_value("paperless_documents_consumed_24h") == 5
    assert registry.get_sample_value("paperless_documents_enriched_24h") == 0


def test_an_unreadable_count_is_left_out_rather_than_pushed_as_zero(monkeypatch):
    """0 would read as a quiet day in the digest; absent reads as "not available"."""
    registry = _pushed_health(monkeypatch)

    assert registry.get_sample_value("paperless_tasks_failed") == 1
    assert registry.get_sample_value("paperless_documents_consumed_24h") is None
    assert registry.get_sample_value("paperless_documents_enriched_24h") is None
