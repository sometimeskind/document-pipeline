"""Entry point: mail-pipeline service."""

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


def main() -> None:
    missing = [v for v in _REQUIRED if not os.environ.get(v)]
    if missing:
        for var in missing:
            logger.error("Required environment variable not set: %s", var)
        sys.exit(1)

    from prefect import serve as prefect_serve

    import waitress
    from mail_pipeline.api import create_app
    from mail_pipeline.flow import mail_flow
    from mail_pipeline.prefect_client import ensure_concurrency_limits

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

    deployment = mail_flow.to_deployment(name="mail", cron=fetch_cron)

    logger.info("Starting Prefect runner (FETCH_CRON=%s)", fetch_cron or "disabled")
    prefect_serve(deployment)
