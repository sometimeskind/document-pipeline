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

    A subcommand (like `rollback`) because the data it reports on lives on a
    PVC only this pod mounts. It is the #1280 harvesting step: tags
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


def _rollback(argv: list[str]) -> None:
    """Replay the enrich before-state from the results JSONL (#1562).

    Lives here for the same reason as `vocab`: the JSONL is on a PVC only this
    pod mounts. Dry-run unless `--write`, like the rest of the pipeline.

        kubectl exec -n mail deploy/document-pipeline -- \\
            python -m document_pipeline rollback --since 2026-09-20T00:00:00Z
    """
    import argparse

    from document_pipeline import enrich, rollback

    parser = argparse.ArgumentParser(
        prog="python -m document_pipeline rollback",
        description="Revert enrich/backfill writes to the recorded before-state.",
    )
    parser.add_argument(
        "--since", required=True, type=rollback.parse_since,
        help="ISO date or timestamp (UTC if no offset); records before it are ignored",
    )
    parser.add_argument("--document", type=int, help="only this document id")
    parser.add_argument("--write", action="store_true", help="apply; default is a dry run")
    args = parser.parse_args(argv)

    try:
        records = rollback.load_latest(None, args.since, args.document)
    except FileNotFoundError:
        logger.error("No enrich results yet — nothing has run.")
        sys.exit(1)

    paperless_url = os.environ["PAPERLESS_URL"]
    # Same superuser-first rule as flow._paperless_admin_token, inlined so this
    # subcommand does not import Prefect.
    token = os.environ.get("PAPERLESS_ADMIN_TOKEN") or os.environ["PAPERLESS_API_TOKEN"]
    with enrich.open_client(token) as client:
        reversions = rollback.run(client, paperless_url, records, write=args.write)

    for r in reversions:
        print(f"{r.document_id:8d}  {r.status:16s}  {r.detail}")
    counts: dict[str, int] = {}
    for r in reversions:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = ", ".join(f"{n} {status}" for status, n in sorted(counts.items())) or "nothing to do"
    print(f"{len(reversions)} document(s): {summary}")
    if not args.write and counts.get("would-revert"):
        print("DRY RUN — nothing was written. Re-run with --write to apply.")


def _migrate_queue(argv: list[str]) -> None:
    """One-off: give `queue` to every document without `ai-processed` (homelab#1561).

    Run AFTER the image that queries on `queue` is deployed, then delete the
    `ai-processed` tag in the paperless UI. Dry-run unless `--write`.

        kubectl exec -n mail deploy/document-pipeline -- \\
            python -m document_pipeline migrate-queue [--write]
    """
    import argparse

    from document_pipeline import enrich, queue_migration

    parser = argparse.ArgumentParser(
        prog="python -m document_pipeline migrate-queue",
        description="Tag every document without `ai-processed` with `queue`.",
    )
    parser.add_argument("--write", action="store_true", help="apply; default is a dry run")
    args = parser.parse_args(argv)

    paperless_url = os.environ["PAPERLESS_URL"]
    token = os.environ.get("PAPERLESS_ADMIN_TOKEN") or os.environ["PAPERLESS_API_TOKEN"]
    with enrich.open_client(token) as client:
        m = queue_migration.run(client, paperless_url, write=args.write)

    marker = queue_migration.LEGACY_MARKER_TAG
    if m.marker_id is None:
        print(f"No {marker!r} tag — already migrated, nothing to do.")
        return
    queue = f"id {m.queue_id}" if m.queue_id is not None else "missing, created on --write"
    print(f"{marker!r}: id {m.marker_id}; {enrich.QUEUE_TAG!r}: {queue}")
    ids = m.document_ids
    shown = ", ".join(str(i) for i in ids[:20]) + (", ..." if len(ids) > 20 else "")
    print(f"{len(ids)} document(s) without {marker!r} and not yet queued: {shown or 'none'}")
    if not args.write:
        if ids:
            print("DRY RUN — nothing was written. Re-run with --write to apply.")
        return
    if m.written:
        print(f"Added {enrich.QUEUE_TAG!r} to {len(ids)} document(s).")
    print(f"Next: delete the {marker!r} tag in the paperless UI (Documents -> Tags).")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "vocab":
        _print_vocab()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        _rollback(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "migrate-queue":
        _migrate_queue(sys.argv[2:])
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
