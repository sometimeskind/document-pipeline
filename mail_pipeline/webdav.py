"""Minimal WebDAV client — the three verbs the scan pipeline needs.

Deliberately server-agnostic RFC 4918: PROPFIND (Depth: 1), GET and DELETE,
hand-rolled on httpx. Nothing here may encode knowledge of a specific server
(currently Davis/sabre-dav) — a later move to another WebDAV server has to be
an env repoint, not a code change (homelab #1259, coupling contract with
#1261). Two rules keep that true:

  * elements are matched by XML *local* name, so any namespace prefix
    (`d:response`, `D:response`, or an unprefixed default `DAV:` namespace)
    parses identically;
  * hrefs are accepted both as absolute URIs and as absolute paths, and are
    never re-encoded — the server's own href is echoed straight back.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import unquote, urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx

logger = logging.getLogger(__name__)

# Ask for the smallest set of properties the pipeline actually uses.
_PROPFIND_BODY = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<propfind xmlns="DAV:"><prop>'
    "<resourcetype/><getcontentlength/><getlastmodified/>"
    "</prop></propfind>"
)


@dataclass(frozen=True)
class WebDAVEntry:
    """One member of a collection, as reported by PROPFIND."""

    href: str
    """Server-supplied href, used verbatim for GET/DELETE."""
    name: str
    """Percent-decoded final path segment."""
    is_collection: bool
    size: int | None
    last_modified: datetime.datetime | None


def _local(tag: object) -> str:
    """Strip the `{namespace}` prefix ElementTree puts on qualified tags."""
    return str(tag).rpartition("}")[2].lower()


def _child(element: ElementTree.Element, name: str) -> ElementTree.Element | None:
    """First direct child with the given local name."""
    return next((c for c in element if _local(c.tag) == name), None)


def _prop(response: ElementTree.Element, name: str) -> ElementTree.Element | None:
    """First property with the given local name, anywhere under `response`.

    Properties live under `propstat/prop`, but servers split them across
    several `propstat` blocks by status, so this searches the whole subtree
    rather than assuming one shape.
    """
    return next((el for el in response.iter() if el is not response and _local(el.tag) == name), None)


def _text(element: ElementTree.Element | None) -> str | None:
    return element.text if element is not None else None


def _parse_xml(content: bytes) -> ElementTree.Element:
    """Parse a multistatus body, refusing any document type declaration.

    The stdlib parser does not fetch *external* entities but does expand
    internal ones — the "billion laughs" amplification, still live on 3.14.
    Entity declarations only exist inside a DTD, so rejecting the DOCTYPE
    closes that off without pulling in defusedxml for what is otherwise a
    dependency-free three-verb client. A multistatus response has no legitimate
    reason to carry one, and `<` cannot appear raw in element text, so this
    cannot trip on a filename.
    """
    if b"<!DOCTYPE" in content:
        raise ValueError("WebDAV response carries a DOCTYPE declaration — refusing to parse")
    return ElementTree.fromstring(content)


def _parse_last_modified(value: str | None) -> datetime.datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        logger.warning("Unparseable getlastmodified %r", value)
        return None


def _parse_size(value: str | None) -> int | None:
    return int(value) if value and value.strip().isdigit() else None


class WebDAVClient:
    """RFC 4918 client scoped to one base URL and one set of Basic credentials."""

    def __init__(self, base_url: str, username: str, password: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        split = urlsplit(self._base_url)
        self._scheme = split.scheme
        self._netloc = split.netloc
        self._auth = httpx.BasicAuth(username, password)
        self._timeout = timeout

    def _url(self, path: str) -> str:
        """Absolute URL for a path relative to the base URL."""
        return f"{self._base_url}/{path.strip('/')}" if path.strip("/") else self._base_url

    def _href_url(self, href: str) -> str:
        """Absolute URL for a server-supplied href.

        Servers may answer with an absolute URI or an absolute path; both are
        legal (RFC 4918 §8.3). The href is used as-is — it is already
        percent-encoded, and re-encoding it would corrupt non-ASCII filenames.
        """
        split = urlsplit(href)
        if split.scheme and split.netloc:
            return href
        return urlunsplit((self._scheme, self._netloc, split.path, split.query, ""))

    def _client(self) -> httpx.Client:
        return httpx.Client(auth=self._auth, timeout=self._timeout, follow_redirects=True)

    def list(self, path: str) -> list[WebDAVEntry]:
        """PROPFIND `path` with Depth: 1 and return its members.

        A Depth-1 multistatus includes the collection itself; that entry is
        filtered out by comparing paths.
        """
        url = self._url(path)
        with self._client() as client:
            resp = client.request(
                "PROPFIND",
                url,
                content=_PROPFIND_BODY,
                headers={"Depth": "1", "Content-Type": 'application/xml; charset="utf-8"'},
            )
        resp.raise_for_status()

        self_path = urlsplit(url).path.rstrip("/")
        entries: list[WebDAVEntry] = []
        for response in _parse_xml(resp.content).iter():
            if _local(response.tag) != "response":
                continue
            href = (_text(_child(response, "href")) or "").strip()
            href_path = urlsplit(href).path
            if not href_path or href_path.rstrip("/") == self_path:
                continue

            resourcetype = _prop(response, "resourcetype")
            entries.append(
                WebDAVEntry(
                    href=href,
                    name=unquote(href_path.rstrip("/").rpartition("/")[2]),
                    is_collection=resourcetype is not None and _child(resourcetype, "collection") is not None,
                    size=_parse_size(_text(_prop(response, "getcontentlength"))),
                    last_modified=_parse_last_modified(_text(_prop(response, "getlastmodified"))),
                )
            )
        return entries

    def get(self, href: str) -> bytes | None:
        """Fetch a resource. Returns None if it is already gone (404)."""
        with self._client() as client:
            resp = client.get(self._href_url(href))
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content

    def delete(self, href: str) -> None:
        """Remove a resource. A 404 is treated as success — the goal is absence."""
        with self._client() as client:
            resp = client.delete(self._href_url(href))
        if resp.status_code != 404:
            resp.raise_for_status()
