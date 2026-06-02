"""Pushgateway metric reporter."""

from __future__ import annotations

import logging
import os
import time

import httpx
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)

_PREFECT_FILTER_URL = "{url}/flow_runs/filter"
_PREFECT_FILTER_BODY = {
    "flow_runs": {
        "state": {"type": {"any_": ["FAILED", "CRASHED"]}},
        "start_time": {"after_": ""},
    },
    "flows": {"name": {"like_": "%mail%"}},
    "limit": 200,
}


def _prefect_failures_24h(prefect_url: str) -> int | None:
    """Return count of failed/crashed mail flow runs in the last 24 h, or None on error."""
    import datetime

    cutoff = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = _PREFECT_FILTER_BODY.copy()
    body["flow_runs"] = {**body["flow_runs"], "start_time": {"after_": cutoff}}

    try:
        resp = httpx.post(
            _PREFECT_FILTER_URL.format(url=prefect_url),
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        return len(resp.json())
    except Exception as exc:
        logger.warning("Prefect failure query failed: %s", exc)
        return None


def push_run_metrics(
    emails_synced: int,
    pdfs_submitted: int,
    duration_seconds: float,
) -> None:
    """Push per-run metrics to Pushgateway. No-op when PUSHGATEWAY_URL is unset."""
    url = os.environ.get("PUSHGATEWAY_URL", "")
    if not url:
        return

    registry = CollectorRegistry()

    Gauge(
        "mail_pipeline_last_success_timestamp",
        "Unix timestamp of the last successful mail-pipeline run",
        registry=registry,
    ).set(time.time())
    Gauge(
        "mail_pipeline_emails_synced",
        "Number of new messages processed from IMAP in the last run",
        registry=registry,
    ).set(emails_synced)
    Gauge(
        "mail_pipeline_pdfs_submitted",
        "Number of PDFs submitted to Paperless in the last run",
        registry=registry,
    ).set(pdfs_submitted)
    Gauge(
        "mail_pipeline_run_duration_seconds",
        "Total duration of the last successful mail-pipeline run in seconds",
        registry=registry,
    ).set(duration_seconds)

    prefect_url = os.environ.get("PREFECT_API_URL", "")
    if prefect_url:
        failures = _prefect_failures_24h(prefect_url)
        if failures is not None:
            Gauge(
                "mail_pipeline_prefect_failures_24h",
                "Number of failed/crashed mail Prefect flow runs in the last 24 hours",
                registry=registry,
            ).set(failures)

    try:
        push_to_gateway(url, job="mail-pipeline", registry=registry, timeout=10)
    except Exception as exc:
        logger.warning("Pushgateway push failed: %s", exc)
