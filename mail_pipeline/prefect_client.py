"""Helpers for talking to the Prefect server."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def _submit() -> None:
    from prefect.deployments import run_deployment
    await run_deployment("mail/mail", timeout=0)


def trigger_sync() -> bool:
    """Submit a `mail` deployment run. Returns True if accepted."""
    try:
        asyncio.run(_submit())
        return True
    except Exception as exc:
        logger.error("Failed to submit mail run: %s", exc)
        return False


async def _check_active_runs() -> bool:
    from prefect import get_client
    from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterState, FlowRunFilterStateType
    from prefect.client.schemas.objects import StateType

    async with get_client() as client:
        runs = await client.read_flow_runs(
            flow_run_filter=FlowRunFilter(
                state=FlowRunFilterState(
                    type=FlowRunFilterStateType(
                        any_=[StateType.RUNNING, StateType.SCHEDULED, StateType.PENDING]
                    )
                )
            )
        )
        return bool(runs)


def has_active_run() -> bool:
    """Return True if a mail flow run is already running or queued."""
    try:
        return asyncio.run(_check_active_runs())
    except Exception as exc:
        logger.warning("Could not check active runs: %s", exc)
        return False


async def _upsert_limits() -> None:
    from prefect import get_client
    async with get_client() as client:
        await client.upsert_global_concurrency_limit_by_name(
            name="mail-pipeline",
            limit=1,
        )


def ensure_concurrency_limits() -> None:
    """Create/update the Prefect global concurrency limit used by the flow."""
    try:
        asyncio.run(_upsert_limits())
        logger.info("Prefect concurrency limits ensured: mail-pipeline=1")
    except Exception as exc:
        logger.warning("Could not upsert Prefect concurrency limit: %s", exc)
