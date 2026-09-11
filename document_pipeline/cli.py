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


# WebDAV is no longer opt-in: the mail flow queues its PDFs through it
# (homelab#1590), so without it neither ingest path works.
_REQUIRED = (
    "PREFECT_API_URL", "PAPERLESS_URL", "PAPERLESS_API_TOKEN", "API_BEARER_TOKEN", "IMAP_PASSWORD",
    "WEBDAV_URL", "WEBDAV_USERNAME", "WEBDAV_PASSWORD",
)

# Hourly sweep. inotify only reports live events, so this is what picks up
# anything that arrived while the watcher sidecar was down, and bounds that
# worst case at an hour.
_DEFAULT_SCAN_CRON = "0 * * * *"

# Task-queue probe (homelab#1589). Five minutes bounds how late a failed
# consume or a starved worker is noticed; the alert on the pushed timestamp
# expects a push at least every 15.
_DEFAULT_PAPERLESS_HEALTH_CRON = "*/5 * * * *"


def _print_vocab() -> None:
    """Frequency-ranked tag names the model proposed that matched nothing.

    The one read-only subcommand this image has, because the data it reports on
    lives on a PVC only this pod mounts. It is the #1280 harvesting step: tags
    cannot bootstrap themselves, so the vocabulary has to be created before
    matching can ever fire, and this is the only evidence of which names are
    worth creating.

        kubectl exec -n mail deploy/document-pipeline -- python -m document_pipeline vocab
    """
    from document_pipeline import enrich

    try:
        documents, ranked = enrich.rank_suggested_tags()
    except FileNotFoundError:
        logger.error("No enrich results yet — nothing has run.")
        sys.exit(1)

    print(f"{documents} document(s), {len(ranked)} distinct unmatched tag name(s)")
    for name, count in ranked:
        print(f"{count:6d}  {name}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "vocab":
        _print_vocab()
        return

    missing = [v for v in _REQUIRED if not os.environ.get(v)]
    if missing:
        for var in missing:
            logger.error("Required environment variable not set: %s", var)
        sys.exit(1)

    from prefect import serve as prefect_serve

    import waitress
    from document_pipeline.api import create_app
    from document_pipeline.flow import (
        correspondent_backfill_flow,
        enrich_flow,
        enrich_sweep_flow,
        mail_flow,
        paperless_health_flow,
        scan_flow,
    )
    from document_pipeline.prefect_client import ensure_concurrency_limits

    fetch_cron = os.environ.get("FETCH_CRON")

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
    deployments.append(scan_flow.to_deployment(name="scan", cron=scan_cron))

    # Trigger-driven, so no cron. `concurrency_limit` is not the Ollama slot the
    # flow itself takes — it is what keeps queued runs out of the serve() runner.
    # The in-flow slot blocks *inside* a run, so without this a burst of consumed
    # documents fills every runner slot with runs waiting on Ollama and the mail
    # cron cannot start. Here they wait in AwaitingConcurrencySlot instead.
    deployments.append(enrich_flow.to_deployment(name="enrich", concurrency_limit=1))
    enrich_sweep_cron = os.environ.get("ENRICH_SWEEP_CRON") or None
    deployments.append(enrich_sweep_flow.to_deployment(name="enrich-sweep", cron=enrich_sweep_cron))

    # Correspondent backfill (#1373): gated on the same env as the Ollama query
    # it runs, so an image that lands ahead of its manifest registers nothing.
    # Like the sweep, registered without a schedule when the cron is unset so a
    # dry-run sample can still be started from the Prefect UI.
    backfill_enabled = bool(
        os.environ.get("ENRICH_OLLAMA_URL") and os.environ.get("ENRICH_OLLAMA_MODEL")
    )
    backfill_cron = os.environ.get("CORRESPONDENT_BACKFILL_CRON") or None
    if backfill_enabled:
        deployments.append(
            correspondent_backfill_flow.to_deployment(
                name="correspondent-backfill", cron=backfill_cron
            )
        )

    paperless_health_cron = os.environ.get("PAPERLESS_HEALTH_CRON", _DEFAULT_PAPERLESS_HEALTH_CRON)
    deployments.append(
        paperless_health_flow.to_deployment(name="paperless-health", cron=paperless_health_cron)
    )

    logger.info(
        "Starting Prefect runner (FETCH_CRON=%s, SCAN_CRON=%s, ENRICH_SWEEP_CRON=%s, "
        "CORRESPONDENT_BACKFILL_CRON=%s, PAPERLESS_HEALTH_CRON=%s)",
        fetch_cron or "disabled",
        scan_cron,
        enrich_sweep_cron or "disabled",
        (backfill_cron or "unscheduled") if backfill_enabled else "disabled",
        paperless_health_cron,
    )
    prefect_serve(*deployments)
