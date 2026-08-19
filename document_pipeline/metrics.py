"""Pushgateway metric reporter."""

from __future__ import annotations

import logging
import os
import time

import httpx
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, pushadd_to_gateway

logger = logging.getLogger(__name__)

_PREFECT_FILTER_URL = "{url}/flow_runs/filter"


def _prefect_failures_24h(prefect_url: str, flow_name: str = "mail") -> int | None:
    """Return count of failed/crashed runs of the named flow in the last 24 h, or None on error."""
    import datetime

    cutoff = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {
        "flow_runs": {
            "state": {"type": {"any_": ["FAILED", "CRASHED"]}},
            "start_time": {"after_": cutoff},
        },
        "flows": {"name": {"like_": f"%{flow_name}%"}},
        "limit": 200,
    }

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
        "document_pipeline_last_success_timestamp",
        "Unix timestamp of the last successful mail-pipeline run",
        registry=registry,
    ).set(time.time())
    Gauge(
        "document_pipeline_emails_synced",
        "Number of new messages processed from IMAP in the last run",
        registry=registry,
    ).set(emails_synced)
    Gauge(
        "document_pipeline_pdfs_submitted",
        "Number of PDFs submitted to Paperless in the last run",
        registry=registry,
    ).set(pdfs_submitted)
    Gauge(
        "document_pipeline_run_duration_seconds",
        "Total duration of the last successful mail-pipeline run in seconds",
        registry=registry,
    ).set(duration_seconds)

    prefect_url = os.environ.get("PREFECT_API_URL", "")
    if prefect_url:
        failures = _prefect_failures_24h(prefect_url)
        if failures is not None:
            Gauge(
                "document_pipeline_prefect_failures_24h",
                "Number of failed/crashed mail Prefect flow runs in the last 24 hours",
                registry=registry,
            ).set(failures)

    _push(url, "mail-pipeline", registry)


def push_scan_metrics(
    files_ingested: int,
    files_failed: int,
    files_pending: int,
    oldest_pending_age_seconds: float,
    duration_seconds: float,
) -> None:
    """Push per-run scan metrics to Pushgateway. No-op when PUSHGATEWAY_URL is unset.

    Pushed under its own job name: a push replaces every metric in a job's
    group, so sharing `mail-pipeline` would have each flow wipe the other's
    gauges on every run.
    """
    url = os.environ.get("PUSHGATEWAY_URL", "")
    if not url:
        return

    registry = CollectorRegistry()

    Gauge(
        "scan_pipeline_last_success_timestamp",
        "Unix timestamp of the last successful scan-pipeline run",
        registry=registry,
    ).set(time.time())
    Gauge(
        "scan_pipeline_files_ingested",
        "Number of scans accepted by Paperless and removed from WebDAV in the last run",
        registry=registry,
    ).set(files_ingested)
    Gauge(
        "scan_pipeline_files_failed",
        "Number of scans that failed to ingest in the last run",
        registry=registry,
    ).set(files_failed)
    Gauge(
        "scan_pipeline_files_pending",
        "Number of eligible scans still sitting in WebDAV after the last run",
        registry=registry,
    ).set(files_pending)
    # An age rather than a timestamp so an empty directory pushes 0 and the
    # staleness alert resolves itself.
    Gauge(
        "scan_pipeline_oldest_pending_file_age_seconds",
        "Age of the oldest eligible scan still in WebDAV, 0 when none remain",
        registry=registry,
    ).set(oldest_pending_age_seconds)
    Gauge(
        "scan_pipeline_run_duration_seconds",
        "Total duration of the last successful scan-pipeline run in seconds",
        registry=registry,
    ).set(duration_seconds)

    prefect_url = os.environ.get("PREFECT_API_URL", "")
    if prefect_url:
        failures = _prefect_failures_24h(prefect_url, flow_name="scan")
        if failures is not None:
            Gauge(
                "scan_pipeline_prefect_failures_24h",
                "Number of failed/crashed scan Prefect flow runs in the last 24 hours",
                registry=registry,
            ).set(failures)

    _push(url, "scan-pipeline", registry)


def push_enrich_metrics(document_id: int, succeeded: bool) -> None:
    """Push auto-title health under the job the PaperlessAutoTitleFailing rule reads.

    Deliberately the same job name and the same series the shell hook used, so
    the alert in kubernetes/paperless/prometheus-rules.yaml needs no change.

    Success and failure are separate series pushed with POST rather than one
    shared result gauge replaced with PUT. That is load-bearing: with a shared
    gauge, document N+1 succeeding overwrote document N failing, which is how
    the alert stayed green while 3 of 8 documents lost their titles (#1295).
    """
    url = os.environ.get("PUSHGATEWAY_URL", "")
    if not url:
        return

    registry = CollectorRegistry()
    if succeeded:
        Gauge(
            "paperless_autotitle_last_success_timestamp",
            "Unix timestamp of the last successful document enrichment",
            registry=registry,
        ).set(time.time())
    else:
        Gauge(
            "paperless_autotitle_last_failure_timestamp",
            "Unix timestamp of the last failed document enrichment",
            registry=registry,
        ).set(time.time())
        # The document id rides along so the alert can name the document that
        # lost its title instead of sending the operator to grep the logs.
        Gauge(
            "paperless_autotitle_last_failure_document",
            "Paperless document id of the last failed enrichment",
            registry=registry,
        ).set(document_id)

    _pushadd(url, "paperless_autotitle", registry)


def _push(url: str, job: str, registry: CollectorRegistry) -> None:
    try:
        push_to_gateway(url, job=job, registry=registry, timeout=10)
    except Exception as exc:
        logger.warning("Pushgateway push failed: %s", exc)


def _pushadd(url: str, job: str, registry: CollectorRegistry) -> None:
    """POST rather than PUT: replaces only the metrics named in the payload.

    `_push` replaces the whole group, which is what the mail and scan flows want
    — each pushes its complete set every run. Enrichment pushes one half of its
    group at a time, so it must leave the other half standing.
    """
    try:
        pushadd_to_gateway(url, job=job, registry=registry, timeout=10)
    except Exception as exc:
        logger.warning("Pushgateway push failed: %s", exc)
