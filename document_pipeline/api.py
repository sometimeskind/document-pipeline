"""Flask application factory."""

from __future__ import annotations

from flask import Flask, jsonify

from document_pipeline.auth import setup_auth


def create_app() -> Flask:
    app = Flask(__name__)
    setup_auth(app)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/sync/trigger")
    def sync_trigger():
        from document_pipeline import prefect_client
        if not prefect_client.trigger_sync():
            return jsonify({"error": "failed to submit run"}), 503
        return jsonify({}), 202

    @app.post("/trigger-flow")
    def trigger_flow():
        from document_pipeline import prefect_client
        if prefect_client.has_active_run():
            return jsonify({}), 202
        if not prefect_client.trigger_sync():
            return jsonify({"error": "failed to start run"}), 503
        return jsonify({}), 200

    @app.post("/trigger-scan")
    def trigger_scan():
        # 202 means "a run is already in flight and will pick this up" — the
        # scan directory is drained wholesale, so an in-flight run covers files
        # that landed after it started. Callers fire one trigger per inotify
        # event; coalescing here is what stops a 20-page batch queueing 20 runs.
        from document_pipeline import prefect_client
        if prefect_client.has_active_scan_run():
            return jsonify({}), 202
        if not prefect_client.trigger_scan():
            return jsonify({"error": "failed to start run"}), 503
        return jsonify({}), 200

    return app
