"""Entry point: document-pipeline service."""

from __future__ import annotations

import logging
import os
import sys
import threading

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger(__name__)


_REQUIRED = ("PREFECT_API_URL", "PAPERLESS_URL", "PAPERLESS_API_TOKEN", "API_BEARER_TOKEN", "IMAP_PASSWORD")
# Scan ingestion is opt-in on WEBDAV_URL. The deployment pins `:latest` by
# digest and Renovate bumps that digest automatically, so a new image can land
# before the manifest that configures it — making scan config mandatory would
# turn that ordering into a mail-ingestion outage.
_REQUIRED_SCAN = ("WEBDAV_USERNAME", "WEBDAV_PASSWORD")

# Hourly sweep. inotify only reports live events, so this is what picks up
# anything that arrived while the watcher sidecar was down, and bounds that
# worst case at an hour.
_DEFAULT_SCAN_CRON = "0 * * * *"


def main() -> None:
    missing = [v for v in _REQUIRED if not os.environ.get(v)]
    if os.environ.get("WEBDAV_URL"):
        missing += [v for v in _REQUIRED_SCAN if not os.environ.get(v)]
    if missing:
        for var in missing:
            logger.error("Required environment variable not set: %s", var)
        sys.exit(1)

    from prefect import serve as prefect_serve

    import waitress
    from document_pipeline.api import create_app
    from document_pipeline.flow import enrich_flow, enrich_sweep_flow, mail_flow, scan_flow
    from document_pipeline.prefect_client import ensure_concurrency_limits

    fetch_cron = os.environ.get("FETCH_CRON")
    scan_enabled = bool(os.environ.get("WEBDAV_URL"))

    # Start Flask first so /health responds immediately, even while Prefect
    # init below is still retrying against a slow or starting server.
    app = create_app()
    flask_thread = threading.Thread(
        target=lambda: waitress.serve(app, host="0.0.0.0", port=8080),
        daemon=True,
    )
    flask_thread.start()
    logger.info("Flask API started on 0.0.0.0:8080")

    ensure_concurrency_limits()

    if not fetch_cron:
        from document_pipeline.prefect_client import clear_deployment_schedules
        clear_deployment_schedules("mail")

    deployments = [mail_flow.to_deployment(name="mail", cron=fetch_cron)]
    scan_cron = os.environ.get("SCAN_CRON", _DEFAULT_SCAN_CRON)
    if scan_enabled:
        deployments.append(scan_flow.to_deployment(name="scan", cron=scan_cron))

    # Trigger-driven, so no cron. `concurrency_limit` is not the Ollama slot the
    # flow itself takes — it is what keeps queued runs out of the serve() runner.
    # The in-flow slot blocks *inside* a run, so without this a burst of consumed
    # documents fills every runner slot with runs waiting on Ollama and the mail
    # cron cannot start. Here they wait in AwaitingConcurrencySlot instead.
    deployments.append(enrich_flow.to_deployment(name="enrich", concurrency_limit=1))
    enrich_sweep_cron = os.environ.get("ENRICH_SWEEP_CRON") or None
    deployments.append(enrich_sweep_flow.to_deployment(name="enrich-sweep", cron=enrich_sweep_cron))

    logger.info(
        "Starting Prefect runner (FETCH_CRON=%s, SCAN_CRON=%s, ENRICH_SWEEP_CRON=%s)",
        fetch_cron or "disabled",
        scan_cron if scan_enabled else "disabled",
        enrich_sweep_cron or "disabled",
    )
    prefect_serve(*deployments)
