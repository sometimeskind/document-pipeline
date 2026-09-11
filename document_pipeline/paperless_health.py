"""Paperless task-queue health: what the tasks API says about consumes.

Paperless exposes no metrics, but ``/api/tasks/`` is the source of truth for
the two faults that were invisible in homelab#1589: a consume that failed
(silently, on the mail path, which only POSTs) and a starved celery worker
leaving tasks queued for hours.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import httpx

TIMEOUT = 30.0


@dataclass(frozen=True)
class TaskQueueHealth:
    failed: int
    oldest_unfinished_seconds: float


def probe(paperless_url: str, paperless_token: str) -> TaskQueueHealth:
    """Read the tasks API (v10, the default: paginated, lowercase statuses).

    Two cheap queries rather than a walk over the task list: ``count`` from a
    filtered page of one is the failure total, and the first row of an
    ascending ``date_created`` order is the oldest unfinished task.

    Failures count ``consume_file`` tasks only — the alert on this is about
    documents that did not land, and acknowledging the task in the Paperless
    UI is what clears it. The unfinished age spans every task type: a starved
    worker stalls whatever is queued, so a scheduled task sitting in PENDING
    is the same signal as a consume doing so.
    """
    with httpx.Client(
        headers={"Authorization": f"Token {paperless_token}"}, timeout=TIMEOUT
    ) as client:
        failed = _tasks(
            client, paperless_url,
            task_type="consume_file", status="failure", acknowledged="false",
        )["count"]
        unfinished = _tasks(
            client, paperless_url,
            status=["pending", "started"], ordering="date_created",
        )["results"]

    oldest = 0.0
    if unfinished:
        created = datetime.datetime.fromisoformat(unfinished[0]["date_created"])
        now = datetime.datetime.now(datetime.timezone.utc)
        oldest = max(0.0, (now - created).total_seconds())
    return TaskQueueHealth(failed=int(failed), oldest_unfinished_seconds=oldest)


def _tasks(client: httpx.Client, paperless_url: str, **params) -> dict:
    resp = client.get(f"{paperless_url}/api/tasks/", params={"page_size": 1, **params})
    resp.raise_for_status()
    return resp.json()
