"""Bearer-token authentication for the Flask app."""

from __future__ import annotations

import os

from flask import Flask, jsonify, request


def setup_auth(app: Flask) -> None:
    """Register a `before_request` hook that enforces bearer-token auth on all routes except `/health`."""

    @app.before_request
    def check_auth():
        if request.path == "/health":
            return None

        token = os.environ.get("API_BEARER_TOKEN", "")
        auth_header = request.headers.get("Authorization", "")
        if not token or not auth_header.startswith("Bearer ") or auth_header[7:] != token:
            return jsonify({"error": "unauthorized"}), 401

        return None
