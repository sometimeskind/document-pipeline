"""Tests for document_pipeline.metrics — the per-source scan series (homelab#1590)."""

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
