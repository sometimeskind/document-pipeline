"""Tests for document_pipeline.paperless_health — the tasks-API probe.

The fake below applies the same filters the paperless v10 tasks endpoint does
(`status`, `acknowledged`, `task_type`, `ordering`, `page_size`), so the
tests pin the query the probe sends as much as the arithmetic it does on the
answer.
"""

from __future__ import annotations

import datetime

import httpx
import pytest
import respx

from document_pipeline import paperless_health


PAPERLESS = "http://paperless"


def _task(status, *, created_ago=0, acknowledged=False, task_type="consume_file"):
    created = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=created_ago)
    return {
        "id": 1,
        "task_type": task_type,
        "status": status,
        "acknowledged": acknowledged,
        "date_created": created.isoformat(),
    }


def _fake_tasks_api(tasks):
    def respond(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        rows = list(tasks)
        if statuses := params.get_list("status"):
            rows = [t for t in rows if t["status"] in statuses]
        if (ack := params.get("acknowledged")) is not None:
            rows = [t for t in rows if t["acknowledged"] is (ack == "true")]
        if task_type := params.get("task_type"):
            rows = [t for t in rows if t["task_type"] == task_type]
        if params.get("ordering") == "date_created":
            rows.sort(key=lambda t: t["date_created"])
        page = rows[: int(params.get("page_size", 25))]
        return httpx.Response(200, json={"count": len(rows), "next": None, "results": page})

    return respx.get(f"{PAPERLESS}/api/tasks/").mock(side_effect=respond)


@respx.mock
def test_no_tasks_is_healthy():
    _fake_tasks_api([])
    assert paperless_health.probe(PAPERLESS, "tok") == paperless_health.TaskQueueHealth(0, 0.0)


@respx.mock
def test_only_unacknowledged_failures_count():
    _fake_tasks_api([_task("failure"), _task("failure", acknowledged=True)])
    assert paperless_health.probe(PAPERLESS, "tok").failed == 1


@respx.mock
def test_failures_of_other_task_types_do_not_count():
    _fake_tasks_api([_task("failure", task_type="train_classifier")])
    assert paperless_health.probe(PAPERLESS, "tok").failed == 0


@respx.mock
def test_oldest_unfinished_age_comes_from_the_oldest_pending_or_started_task():
    _fake_tasks_api([
        _task("started", created_ago=60),
        _task("pending", created_ago=900),
        _task("success", created_ago=5000),
    ])
    age = paperless_health.probe(PAPERLESS, "tok").oldest_unfinished_seconds
    assert 899 <= age <= 905


@respx.mock
def test_finished_tasks_are_ignored():
    _fake_tasks_api([_task("success", created_ago=3000), _task("revoked", created_ago=3000)])
    assert paperless_health.probe(PAPERLESS, "tok") == paperless_health.TaskQueueHealth(0, 0.0)


@respx.mock
def test_probe_sends_the_token_and_pages_of_one():
    route = _fake_tasks_api([])
    paperless_health.probe(PAPERLESS, "tok")
    assert route.call_count == 2
    for call in route.calls:
        assert call.request.headers["Authorization"] == "Token tok"
        assert call.request.url.params["page_size"] == "1"


@respx.mock
def test_api_errors_raise():
    """A 403 must fail the probe, not report a healthy queue."""
    respx.get(f"{PAPERLESS}/api/tasks/").mock(return_value=httpx.Response(403))
    with pytest.raises(httpx.HTTPStatusError):
        paperless_health.probe(PAPERLESS, "tok")
