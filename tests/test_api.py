"""Tests for document_pipeline.api — Flask routes."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from document_pipeline.api import create_app


AUTH = {"Authorization": "Bearer test-secret"}


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("API_BEARER_TOKEN", "test-secret")
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_sync_trigger_returns_202_when_accepted(client):
    with patch("document_pipeline.prefect_client.trigger_sync", return_value=True):
        resp = client.post("/sync/trigger", headers=AUTH)
    assert resp.status_code == 202


def test_sync_trigger_returns_503_when_prefect_unreachable(client):
    with patch("document_pipeline.prefect_client.trigger_sync", return_value=False):
        resp = client.post("/sync/trigger", headers=AUTH)
    assert resp.status_code == 503


def test_trigger_flow_returns_200_when_no_active_run(client):
    with patch("document_pipeline.prefect_client.has_active_run", return_value=False), \
         patch("document_pipeline.prefect_client.trigger_sync", return_value=True):
        resp = client.post("/trigger-flow", headers=AUTH)
    assert resp.status_code == 200


def test_trigger_flow_returns_202_when_run_already_in_flight(client):
    with patch("document_pipeline.prefect_client.has_active_run", return_value=True):
        resp = client.post("/trigger-flow", headers=AUTH)
    assert resp.status_code == 202


def test_trigger_flow_returns_401_without_auth(client):
    resp = client.post("/trigger-flow")
    assert resp.status_code == 401


def test_trigger_flow_returns_503_when_prefect_fails(client):
    with patch("document_pipeline.prefect_client.has_active_run", return_value=False), \
         patch("document_pipeline.prefect_client.trigger_sync", return_value=False):
        resp = client.post("/trigger-flow", headers=AUTH)
    assert resp.status_code == 503


def test_trigger_scan_returns_200_when_no_active_run(client):
    with patch("document_pipeline.prefect_client.has_active_scan_run", return_value=False), \
         patch("document_pipeline.prefect_client.trigger_scan", return_value=True):
        resp = client.post("/trigger-scan", headers=AUTH)
    assert resp.status_code == 200


def test_trigger_scan_coalesces_onto_an_in_flight_run(client):
    """One inotify event per page must not queue one Prefect run per page."""
    with patch("document_pipeline.prefect_client.has_active_scan_run", return_value=True), \
         patch("document_pipeline.prefect_client.trigger_scan") as trigger:
        resp = client.post("/trigger-scan", headers=AUTH)
    assert resp.status_code == 202
    trigger.assert_not_called()


def test_trigger_scan_returns_401_without_auth(client):
    resp = client.post("/trigger-scan")
    assert resp.status_code == 401


def test_trigger_scan_returns_503_when_prefect_fails(client):
    with patch("document_pipeline.prefect_client.has_active_scan_run", return_value=False), \
         patch("document_pipeline.prefect_client.trigger_scan", return_value=False):
        resp = client.post("/trigger-scan", headers=AUTH)
    assert resp.status_code == 503


def test_trigger_enrich_submits_a_run_for_the_document(client):
    with patch("document_pipeline.prefect_client.trigger_enrich", return_value=True) as trigger:
        resp = client.post("/trigger-enrich", headers=AUTH, json={"document_id": 42})
    assert resp.status_code == 200
    trigger.assert_called_once_with(42)


def test_trigger_enrich_does_not_coalesce(client):
    """Unlike /trigger-scan: two document ids are not interchangeable work."""
    with patch("document_pipeline.prefect_client.trigger_enrich", return_value=True) as trigger:
        client.post("/trigger-enrich", headers=AUTH, json={"document_id": 42})
        client.post("/trigger-enrich", headers=AUTH, json={"document_id": 43})
    assert [c.args[0] for c in trigger.call_args_list] == [42, 43]


@pytest.mark.parametrize("body", [None, {}, {"document_id": "42"}, {"document_id": 0}, {"document_id": True}])
def test_trigger_enrich_rejects_a_bad_document_id(client, body):
    with patch("document_pipeline.prefect_client.trigger_enrich") as trigger:
        resp = client.post("/trigger-enrich", headers=AUTH, json=body)
    assert resp.status_code == 400
    trigger.assert_not_called()


def test_trigger_enrich_returns_401_without_auth(client):
    resp = client.post("/trigger-enrich", json={"document_id": 42})
    assert resp.status_code == 401


def test_trigger_enrich_returns_503_when_prefect_fails(client):
    with patch("document_pipeline.prefect_client.trigger_enrich", return_value=False):
        resp = client.post("/trigger-enrich", headers=AUTH, json={"document_id": 42})
    assert resp.status_code == 503
