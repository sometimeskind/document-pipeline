"""Tests for the read-only subcommands in document_pipeline.cli."""

from __future__ import annotations

import json
import sys

import pytest

from document_pipeline import cli


def _results(tmp_path, monkeypatch, records):
    target = tmp_path / "results.jsonl"
    target.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    monkeypatch.setenv("ENRICH_RESULTS_PATH", str(target))


def test_compare_prints_both_modes_and_the_gate_totals(tmp_path, monkeypatch, capsys):
    _results(tmp_path, monkeypatch, [
        {"document_id": 1, "outcome": "dry-run", "mode": "suggest", "title": "Invoice",
         "correspondent": "Hermes", "matched_tags": [5], "suggested_tags": [],
         "model_passes": 4, "duration_seconds": 100.0},
        {"document_id": 1, "outcome": "dry-run", "mode": "extract", "title": "Factuur",
         "correspondent": "Hermes", "matched_tags": [], "suggested_tags": ["x"],
         "model_passes": 1, "duration_seconds": 30.0, "created": "2026-09-01"},
    ])
    monkeypatch.setattr(sys, "argv", ["document_pipeline", "compare"])

    cli.main()

    out = capsys.readouterr().out
    assert "'Invoice'" in out and "'Factuur'" in out
    assert "suggest 4  extract 1" in out
    assert "1/1 agree" in out
    assert "suggest 1/1  extract 0/1" in out


def test_compare_exits_when_no_document_has_both_modes(tmp_path, monkeypatch):
    _results(tmp_path, monkeypatch, [
        {"document_id": 1, "outcome": "dry-run", "mode": "suggest", "title": "Invoice"},
    ])
    monkeypatch.setattr(sys, "argv", ["document_pipeline", "compare"])

    with pytest.raises(SystemExit):
        cli.main()
