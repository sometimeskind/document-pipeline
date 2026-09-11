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
    UI is what clears it. Duplicate rejections are excluded: since 3.1
    Paperless records "already have this document" as a *failed* task with
    ``duplicate_of`` in its result data (the scan flow treats that as
    success and deletes the file), so every resubmit of an already-ingested
    file would otherwise page — 67 of the 78 tasks the first live probe
    found were exactly that. The unfinished age spans every task type: a starved
    worker stalls whatever is queued, so a scheduled task sitting in PENDING
    is the same signal as a consume doing so.

    Both queries skip acknowledged tasks. That is the operator's escape hatch
    in either direction: the failed backlog is dismissed in the Paperless UI
    once handled, and a task record left in PENDING/STARTED forever by a pod
    killed mid-run (the first live probe found one 110 days old) is dismissed
    the same way — Paperless never finalises it, so the age would otherwise
    ratchet the stalled-queue alert on for good.
    """
    with httpx.Client(
        headers={"Authorization": f"Token {paperless_token}"}, timeout=TIMEOUT
    ) as client:
        failed = _failed_consumes(client, paperless_url)
        unfinished = _tasks(
            client, paperless_url,
            status=["pending", "started"], acknowledged="false", ordering="date_created",
        )["results"]

    oldest = 0.0
    if unfinished:
        created = datetime.datetime.fromisoformat(unfinished[0]["date_created"])
        now = datetime.datetime.now(datetime.timezone.utc)
        oldest = max(0.0, (now - created).total_seconds())
    return TaskQueueHealth(failed=int(failed), oldest_unfinished_seconds=oldest)


def _failed_consumes(client: httpx.Client, paperless_url: str) -> int:
    """Unacknowledged failed consume tasks that are not duplicate rejections.

    The API cannot filter on result data, so this walks the (paginated)
    failed list; it is short in steady state because dismissal empties it.
    """
    failed = 0
    page = _tasks(
        client, paperless_url,
        task_type="consume_file", status="failure", acknowledged="false", page_size=100,
    )
    while True:
        failed += sum(1 for task in page["results"] if not _is_duplicate(task))
        if not page.get("next"):
            return failed
        resp = client.get(page["next"])
        resp.raise_for_status()
        page = resp.json()


def _is_duplicate(task: dict) -> bool:
    result = task.get("result_data")
    return isinstance(result, dict) and bool(result.get("duplicate_of"))


def _tasks(client: httpx.Client, paperless_url: str, page_size: int = 1, **params) -> dict:
    resp = client.get(f"{paperless_url}/api/tasks/", params={"page_size": page_size, **params})
    resp.raise_for_status()
    return resp.json()
