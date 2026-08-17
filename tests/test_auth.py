"""Tests for document_pipeline.auth — bearer-token middleware."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from document_pipeline.api import create_app


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("API_BEARER_TOKEN", "test-secret")
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_health_requires_no_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_sync_trigger_without_token_returns_401(client):
    resp = client.post("/sync/trigger")
    assert resp.status_code == 401


def test_sync_trigger_wrong_token_returns_401(client):
    resp = client.post("/sync/trigger", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_sync_trigger_correct_token_returns_202(client):
    with patch("document_pipeline.prefect_client.trigger_sync", return_value=True):
        resp = client.post(
            "/sync/trigger",
            headers={"Authorization": "Bearer test-secret"},
        )
    assert resp.status_code == 202
