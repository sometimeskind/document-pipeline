"""Tests for document_pipeline.prefect_client — deployment trigger dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True, scope="module")
def prefect_test_env():
    # The harness default is 30s, which is enough on CI but not on a slow box —
    # the ephemeral server startup is the only thing this waits on.
    from prefect.testing.utilities import prefect_test_harness
    with prefect_test_harness(server_startup_timeout=180):
        yield


def test_trigger_from_inside_a_sync_task_submits_the_run():
    """The scan trigger fired by `process-mail` runs inside a sync task.

    `run_deployment` is `@async_dispatch(arun_deployment)`, and the dispatcher
    keys off the Prefect *run context*, not the event loop: inside a sync task
    `is_in_async_context()` returns `parent.isasync`, i.e. False, so the sync
    implementation runs and hands back a `FlowRun` — awaiting that raises
    `'FlowRun' object can't be awaited`. The run is created either way, so the
    only symptom was `_trigger` reporting failure for work it had just queued
    (2026-09-16 08:20Z). Awaiting `arun_deployment` directly is unconditional.

    This has to drive a real sync task: test_flow patches `prefect_client`
    wholesale, so the call shape never runs there.
    """
    from prefect import task

    from document_pipeline import prefect_client

    calls = []

    async def fake_arun_deployment(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(id="00000000-0000-0000-0000-000000000000")

    @task
    def trigger_scan_in_a_sync_task() -> bool:
        return prefect_client.trigger_scan()

    with patch("prefect.deployments.arun_deployment", fake_arun_deployment):
        assert trigger_scan_in_a_sync_task() is True

    assert calls == [("scan/scan", {"parameters": None, "timeout": 0})]


def test_trigger_outside_a_run_context_submits_the_run():
    """The same call from the Flask thread — no run context, plain asyncio.run."""
    from document_pipeline import prefect_client

    calls = []

    async def fake_arun_deployment(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(id="00000000-0000-0000-0000-000000000000")

    with patch("prefect.deployments.arun_deployment", fake_arun_deployment):
        assert prefect_client.trigger_enrich(42) is True

    assert calls == [("enrich/enrich", {"parameters": {"document_id": 42}, "timeout": 0})]


def test_trigger_reports_failure_when_the_submit_raises():
    """A genuine submit failure still has to come back as False."""
    from document_pipeline import prefect_client

    async def boom(name, **kwargs):
        raise RuntimeError("prefect server unreachable")

    with patch("prefect.deployments.arun_deployment", boom):
        assert prefect_client.trigger_scan() is False
