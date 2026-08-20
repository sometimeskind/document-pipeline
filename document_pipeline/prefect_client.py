"""Helpers for talking to the Prefect server."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def _submit(name: str, parameters: dict | None = None) -> None:
    from prefect.deployments import run_deployment
    await run_deployment(f"{name}/{name}", parameters=parameters, timeout=0)


def _trigger(name: str, parameters: dict | None = None) -> bool:
    try:
        asyncio.run(_submit(name, parameters))
        return True
    except Exception as exc:
        logger.error("Failed to submit %s run: %s", name, exc)
        return False


def trigger_sync() -> bool:
    """Submit a `mail` deployment run. Returns True if accepted."""
    return _trigger("mail")


def trigger_scan() -> bool:
    """Submit a `scan` deployment run. Returns True if accepted."""
    return _trigger("scan")


async def _check_active_runs(name: str) -> bool:
    from prefect import get_client
    from prefect.client.schemas.filters import (
        DeploymentFilter,
        DeploymentFilterName,
        FlowRunFilter,
        FlowRunFilterState,
        FlowRunFilterStateType,
    )
    from prefect.client.schemas.objects import StateType

    # Scope by deployment name so unrelated deployments (e.g. a daily-cron
    # deployment that always has SCHEDULED future runs queued) don't block
    # our trigger. RUNNING/PENDING only — SCHEDULED runs are by definition
    # future and don't conflict with starting one now; Prefect's own
    # concurrency limit on the flow is the last line of defense.
    async with get_client() as client:
        runs = await client.read_flow_runs(
            deployment_filter=DeploymentFilter(name=DeploymentFilterName(any_=[name])),
            flow_run_filter=FlowRunFilter(
                state=FlowRunFilterState(
                    type=FlowRunFilterStateType(
                        any_=[StateType.RUNNING, StateType.PENDING]
                    )
                )
            ),
        )
        return bool(runs)


def _has_active_run(name: str) -> bool:
    try:
        return asyncio.run(_check_active_runs(name))
    except Exception as exc:
        logger.warning("Could not check active %s runs: %s", name, exc)
        return False


def has_active_run() -> bool:
    """Return True if a mail flow run is already running or queued."""
    return _has_active_run("mail")


def has_active_scan_run() -> bool:
    """Return True if a scan flow run is already running or queued.

    This is what coalesces a burst of triggers: a multi-page batch fires one
    inotify event per file, and every trigger after the first is answered 202
    instead of queueing another Prefect run that would only block on the
    concurrency slot. Mirrors mail-sync's `/trigger-flow` path.
    """
    return _has_active_run("scan")


def trigger_enrich(document_id: int) -> bool:
    """Submit an `enrich` deployment run for one document. Returns True if accepted.

    Note there is no `has_active_enrich_run()` companion, unlike every other
    trigger here. Coalescing is right for scan and mail because those runs drain
    a source wholesale, so an in-flight run covers later arrivals. Two different
    document ids are not interchangeable work — coalescing would silently drop
    documents. One run per document instead, which also gives a Prefect run page
    per document.
    """
    return _trigger("enrich", {"document_id": document_id})


async def _clear_schedules(name: str) -> None:
    import os
    import httpx
    api = os.environ.get("PREFECT_API_URL", "http://prefect-server.prefect.svc.cluster.local:4200/api")
    async with httpx.AsyncClient() as http:
        resp = await http.get(f"{api}/deployments/name/{name}/{name}")
        resp.raise_for_status()
        dep = resp.json()
        dep_id = dep["id"]
        # Clear new-style schedules list
        for s in dep.get("schedules") or []:
            await http.delete(f"{api}/deployments/{dep_id}/schedules/{s['id']}")
            logger.info("Cleared stale deployment schedule: %s", s["id"])
        # Clear legacy single-schedule field if present
        if dep.get("schedule"):
            await http.patch(f"{api}/deployments/{dep_id}", json={"schedule": None})
            logger.info("Cleared stale legacy deployment schedule")


def clear_deployment_schedules(name: str = "mail") -> None:
    """Remove any cron schedules persisted in the Prefect DB for this deployment."""
    try:
        asyncio.run(_clear_schedules(name))
    except Exception as exc:
        logger.warning("Could not clear %s deployment schedules: %s", name, exc)


async def _upsert_limits() -> None:
    from prefect import get_client
    async with get_client() as client:
        # `ollama` is not a pipeline slot: it guards the shared single-GPU
        # model. OLLAMA_NUM_PARALLEL=1 already serializes inference; this makes
        # that explicit and queued rather than a pile of timeouts, and is the
        # prerequisite for ever giving Ollama a second consumer.
        # `enrich-sweep` keeps a long hourly batch from overlapping the next
        # cron firing and doing the same documents twice (#1280).
        for name in ("mail-pipeline", "scan-pipeline", "ollama", "enrich-sweep"):
            await client.upsert_global_concurrency_limit_by_name(name=name, limit=1)


def ensure_concurrency_limits() -> None:
    """Create/update the Prefect global concurrency limits used by the flows."""
    try:
        asyncio.run(_upsert_limits())
        logger.info(
            "Prefect concurrency limits ensured: "
            "mail-pipeline=1, scan-pipeline=1, ollama=1, enrich-sweep=1"
        )
    except Exception as exc:
        logger.warning("Could not upsert Prefect concurrency limit: %s", exc)
