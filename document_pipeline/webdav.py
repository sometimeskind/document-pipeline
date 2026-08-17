"""WebDAV access for the scan pipeline, on top of `webdav4`.

`webdav4` rather than a hand-rolled client: it speaks RFC 4918 over httpx —
already a dependency here, so no second HTTP stack — it is maintained, and its
own suite runs against wsgidav rather than the server we happen to deploy. That
last point is the one the pipeline actually needs: a later move off Davis
(homelab #1259, #1261) has to be a config repoint, not a code change, and a
client tested against a different server than ours is far better evidence of
that than fixtures we wrote ourselves.

This module is only the adapter. It exists for the three things `webdav4`
correctly leaves to the caller:

  * entries in the shape the scan flow wants;
  * "already gone" treated as success rather than an exception, because two
    overlapping runs racing on the same file is expected and benign;
  * a missing scan directory treated as an empty one — sabre creates a user's
    home lazily on first authenticated access, so it genuinely does not exist
    until the scanner (or an operator) first writes to it.
"""

from __future__ import annotations

import datetime
import io
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from webdav4.client import Client, ResourceNotFound

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebDAVEntry:
    """One member of a collection."""

    path: str
    """Path relative to the client's base URL — what get/delete take."""
    name: str
    """Final path segment, percent-decoded."""
    is_collection: bool
    size: int | None
    last_modified: datetime.datetime | None


def _reject_doctype(response: httpx.Response) -> None:
    """Refuse a multistatus body carrying a document type declaration.

    `webdav4` parses with stdlib ElementTree, which does not fetch external
    entities but still expands internal ones — the "billion laughs"
    amplification, live as of 3.14. Entity declarations only exist inside a
    DTD, so refusing the DOCTYPE closes that off.

    Scoped to 207 so it never touches a streamed GET: a multistatus is small
    and `webdav4` reads it in full anyway, whereas reading a download here
    would defeat chunked transfer.
    """
    if response.status_code != 207:
        return
    response.read()
    if b"<!DOCTYPE" in response.content:
        raise ValueError("WebDAV multistatus carries a DOCTYPE declaration — refusing to parse")


class WebDAVClient:
    """Scoped to one base URL and one set of Basic credentials."""

    def __init__(self, base_url: str, username: str, password: str, timeout: float = 30.0) -> None:
        # The httpx client is built here rather than passing `auth=` to Client:
        # webdav4 ignores `auth` entirely once `http_client` is supplied, and
        # the response hook above is only reachable by owning the client.
        self._client = Client(
            base_url,
            http_client=httpx.Client(
                base_url=base_url,
                auth=httpx.BasicAuth(username, password),
                timeout=timeout,
                follow_redirects=True,
                event_hooks={"response": [_reject_doctype]},
            ),
        )

    def list(self, path: str) -> list[WebDAVEntry]:
        """Return the members of `path`, or nothing if it does not exist yet."""
        try:
            items: list[Any] = self._client.ls(path, detail=True)
        except ResourceNotFound:
            # Not an error: the home is created on first authenticated write.
            # Warned rather than silent so a mistyped WEBDAV_SCAN_PATH is still
            # visible in the logs instead of looking like an idle directory.
            logger.warning("Scan directory %r does not exist (yet) — nothing to ingest", path)
            return []

        return [
            WebDAVEntry(
                path=item["name"],
                name=str(item["name"]).rstrip("/").rpartition("/")[2],
                is_collection=item["type"] == "directory",
                size=item.get("content_length"),
                last_modified=item.get("modified"),
            )
            for item in items
        ]

    def get(self, path: str) -> bytes | None:
        """Fetch a resource. Returns None if it is already gone."""
        buffer = io.BytesIO()
        try:
            self._client.download_fileobj(path, buffer)
        except ResourceNotFound:
            return None
        return buffer.getvalue()

    def delete(self, path: str) -> None:
        """Remove a resource. Already-absent counts as success — absence is the goal."""
        try:
            self._client.remove(path)
        except ResourceNotFound:
            logger.info("%r was already gone", path)
