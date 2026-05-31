"""Subprocess wrappers around the `notmuch` CLI."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time

logger = logging.getLogger(__name__)


def _env(notmuch_config: str) -> dict[str, str]:
    return {**os.environ, "NOTMUCH_CONFIG": notmuch_config}


def index_mail(notmuch_config: str) -> int:
    """Run `notmuch new` and return the number of newly indexed messages."""
    started = time.perf_counter()
    result = subprocess.run(
        ["notmuch", "new"],
        check=True, capture_output=True, text=True,
        env=_env(notmuch_config),
    )
    elapsed = time.perf_counter() - started
    output = (result.stdout or "").strip()
    if output:
        logger.info("notmuch new (%.2fs): %s", elapsed, output.replace("\n", " | "))
    else:
        logger.info("notmuch new finished in %.2fs (no output)", elapsed)

    m = re.search(r"Added (\d+)", output)
    return int(m.group(1)) if m else 0


def search_message_ids(query: str, notmuch_config: str) -> list[str]:
    """Return notmuch message identifiers (one `id:<hash>` per match)."""
    result = subprocess.run(
        ["notmuch", "search", "--output=messages", query],
        capture_output=True, text=True, check=True, env=_env(notmuch_config),
    )
    return [line for line in result.stdout.strip().splitlines() if line]


def message_files(message_id: str, notmuch_config: str) -> list[str]:
    """Return on-disk file paths for *message_id* (`id:<hash>` form)."""
    result = subprocess.run(
        ["notmuch", "search", "--output=files", message_id],
        capture_output=True, text=True, check=True, env=_env(notmuch_config),
    )
    return [line for line in result.stdout.strip().splitlines() if line]


def tag(message_id: str, tag_expr: str, notmuch_config: str) -> None:
    """Apply *tag_expr* (e.g. `+paperless`) to *message_id*."""
    subprocess.run(
        ["notmuch", "tag", tag_expr, message_id],
        check=True, env=_env(notmuch_config),
    )
