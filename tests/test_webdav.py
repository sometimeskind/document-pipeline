"""Tests for document_pipeline.webdav — the adapter over webdav4.

The parsing itself is webdav4's job and is its problem to regress. What these
tests hold down is the contract this pipeline depends on: that the same code
works against differently-shaped servers, and that the three cases webdav4
leaves to the caller (already-gone resources, a not-yet-created scan directory,
a hostile multistatus) behave the way the scan flow assumes.

Every parsing case therefore runs against two deliberately different multistatus
shapes — lowercase `d:`-prefixed with absolute-path hrefs (what sabre-dav/Davis
emits) and uppercase `D:`-prefixed with absolute-URI hrefs. Pinning to one
server's fixtures is how Davis-specific behaviour would creep in unnoticed, and
would only surface during a future migration (homelab #1259, #1261).
"""

from __future__ import annotations

import httpx
import pytest
import respx

from document_pipeline.webdav import WebDAVClient


SABRE_STYLE = """<?xml version="1.0" encoding="utf-8"?>
<d:multistatus xmlns:d="DAV:">
  <d:response>
    <d:href>/dav.php/homes/scanner/</d:href>
    <d:propstat>
      <d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop>
      <d:status>HTTP/1.1 200 OK</d:status>
    </d:propstat>
  </d:response>
  <d:response>
    <d:href>/dav.php/homes/scanner/scan%20001.pdf</d:href>
    <d:propstat>
      <d:prop>
        <d:resourcetype/>
        <d:getcontentlength>1234</d:getcontentlength>
        <d:getlastmodified>Sun, 16 Aug 2026 09:00:00 GMT</d:getlastmodified>
      </d:prop>
      <d:status>HTTP/1.1 200 OK</d:status>
    </d:propstat>
    <d:propstat>
      <d:prop><d:getetag/></d:prop>
      <d:status>HTTP/1.1 404 Not Found</d:status>
    </d:propstat>
  </d:response>
  <d:response>
    <d:href>/dav.php/homes/scanner/2026-08/</d:href>
    <d:propstat>
      <d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop>
      <d:status>HTTP/1.1 200 OK</d:status>
    </d:propstat>
  </d:response>
</d:multistatus>
"""

UPPERCASE_ABSOLUTE_URI_STYLE = """<?xml version="1.0" encoding="utf-8"?>
<D:multistatus xmlns:D="DAV:">
  <D:response>
    <D:href>http://dav.test/dav.php/homes/scanner/</D:href>
    <D:propstat>
      <D:prop><D:resourcetype><D:collection/></D:resourcetype></D:prop>
      <D:status>HTTP/1.1 200 OK</D:status>
    </D:propstat>
  </D:response>
  <D:response>
    <D:href>http://dav.test/dav.php/homes/scanner/scan%20001.pdf</D:href>
    <D:propstat>
      <D:prop>
        <D:resourcetype/>
        <D:getcontentlength>1234</D:getcontentlength>
        <D:getlastmodified>Sun, 16 Aug 2026 09:00:00 GMT</D:getlastmodified>
      </D:prop>
      <D:status>HTTP/1.1 200 OK</D:status>
    </D:propstat>
  </D:response>
  <D:response>
    <D:href>http://dav.test/dav.php/homes/scanner/2026-08/</D:href>
    <D:propstat>
      <D:prop><D:resourcetype><D:collection/></D:resourcetype></D:prop>
      <D:status>HTTP/1.1 200 OK</D:status>
    </D:propstat>
  </D:response>
</D:multistatus>
"""

BASE_URL = "http://dav.test/dav.php"
LIST_URL = f"{BASE_URL}/homes/scanner"
SCAN_PATH = "homes/scanner/scan 001.pdf"


def _file_multistatus(href: str) -> str:
    """Single-resource multistatus, as returned by PROPFIND on a file."""
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<d:multistatus xmlns:d="DAV:"><d:response>'
        f"<d:href>{href}</d:href>"
        "<d:propstat><d:prop><d:resourcetype/><d:getcontentlength>8</d:getcontentlength></d:prop>"
        "<d:status>HTTP/1.1 200 OK</d:status></d:propstat>"
        "</d:response></d:multistatus>"
    )


def _mock_download(url: str, response: httpx.Response):
    """Mock both halves of a webdav4 download.

    `Client.open` checks `isdir` before streaming, so every download is a
    PROPFIND followed by a GET. One extra small request per file, which is
    nothing beside the OCR wait that follows it.
    """
    respx.request("PROPFIND", url).mock(
        return_value=httpx.Response(207, text=_file_multistatus(httpx.URL(url).path))
    )
    return respx.get(url).mock(return_value=response)


@pytest.fixture
def client():
    return WebDAVClient(BASE_URL, "scanner", "hunter2")


@pytest.fixture(params=[SABRE_STYLE, UPPERCASE_ABSOLUTE_URI_STYLE], ids=["sabre", "uppercase-absolute-uri"])
def multistatus(request):
    return request.param


@respx.mock
def test_list_returns_members_excluding_the_collection_itself(client, multistatus):
    respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(207, text=multistatus))

    entries = client.list("homes/scanner")

    assert [(e.name, e.is_collection) for e in entries] == [("scan 001.pdf", False), ("2026-08", True)]


@respx.mock
def test_list_parses_size_and_last_modified(client, multistatus):
    respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(207, text=multistatus))

    scan = client.list("homes/scanner")[0]

    assert scan.size == 1234
    assert scan.last_modified is not None
    assert scan.last_modified.isoformat() == "2026-08-16T09:00:00+00:00"


@respx.mock
def test_list_yields_a_path_that_get_and_delete_accept(client, multistatus):
    """The entry path must round-trip, whichever href form the server used."""
    respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(207, text=multistatus))
    download = _mock_download(
        f"{BASE_URL}/homes/scanner/scan%20001.pdf", httpx.Response(200, content=b"%PDF-1.4")
    )

    entry = client.list("homes/scanner")[0]

    assert client.get(entry.path) == b"%PDF-1.4"
    assert download.called


@respx.mock
def test_missing_scan_directory_is_empty_not_an_error(client):
    """Sabre creates a user's home on first authenticated write, so before the
    scanner has ever run the directory genuinely does not exist. That must not
    fail every hourly sweep."""
    respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(404))

    assert client.list("homes/scanner") == []


@respx.mock
def test_list_rejects_entity_declarations(client):
    """webdav4 parses with stdlib ElementTree, which expands internal entities."""
    bomb = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE multistatus [<!ENTITY a "aaaaaaaaaa">]>'
        '<d:multistatus xmlns:d="DAV:"><d:response><d:href>/&a;</d:href></d:response></d:multistatus>'
    )
    respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(207, text=bomb))

    with pytest.raises(ValueError, match="DOCTYPE"):
        client.list("homes/scanner")


@respx.mock
def test_downloads_are_not_intercepted_by_the_doctype_guard(client):
    """The guard is scoped to 207 so a streamed GET is never read into memory
    by the hook — a PDF that happens to contain the bytes must still download."""
    _mock_download(
        f"{BASE_URL}/homes/scanner/scan%20001.pdf",
        httpx.Response(200, content=b"%PDF-1.4 <!DOCTYPE html> trailing"),
    )

    assert client.get(SCAN_PATH) == b"%PDF-1.4 <!DOCTYPE html> trailing"


@respx.mock
def test_get_returns_none_when_already_gone(client):
    respx.request("PROPFIND", f"{BASE_URL}/homes/scanner/gone.pdf").mock(return_value=httpx.Response(404))

    assert client.get("homes/scanner/gone.pdf") is None


@respx.mock
def test_get_raises_on_server_error(client):
    _mock_download(f"{BASE_URL}/homes/scanner/boom.pdf", httpx.Response(500))

    with pytest.raises(Exception):
        client.get("homes/scanner/boom.pdf")


@respx.mock
def test_delete_tolerates_a_missing_resource(client):
    respx.delete(f"{BASE_URL}/homes/scanner/gone.pdf").mock(return_value=httpx.Response(404))

    client.delete("homes/scanner/gone.pdf")


@respx.mock
def test_delete_raises_on_failure(client):
    respx.delete(f"{BASE_URL}/homes/scanner/locked.pdf").mock(return_value=httpx.Response(423))

    with pytest.raises(Exception):
        client.delete("homes/scanner/locked.pdf")


@respx.mock
def test_requests_carry_basic_auth(client):
    route = respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(207, text=SABRE_STYLE))

    client.list("homes/scanner")

    assert route.calls.last.request.headers["Authorization"].startswith("Basic ")
