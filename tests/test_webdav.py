"""Tests for mail_pipeline.webdav — RFC 4918 parsing that must not assume a server.

Every parsing test runs against two deliberately different multistatus shapes:
lowercase `d:`-prefixed with absolute-path hrefs (what sabre-dav/Davis emits)
and uppercase `D:`-prefixed with absolute-URI hrefs. Pinning the tests to one
server's fixtures is exactly how Davis-specific behaviour would get baked in
unnoticed (homelab #1259).
"""

from __future__ import annotations

import httpx
import pytest
import respx

from mail_pipeline.webdav import WebDAVClient


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


@pytest.fixture
def client():
    return WebDAVClient(BASE_URL, "scanner", "hunter2")


@pytest.fixture(params=[SABRE_STYLE, UPPERCASE_ABSOLUTE_URI_STYLE], ids=["sabre", "uppercase-absolute-uri"])
def multistatus(request):
    return request.param


@respx.mock
def test_list_returns_members_excluding_the_collection_itself(client, multistatus):
    respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(207, text=multistatus))

    entries = client.list("/homes/scanner")

    assert [(e.name, e.is_collection) for e in entries] == [("scan 001.pdf", False), ("2026-08", True)]


@respx.mock
def test_list_parses_size_and_last_modified(client, multistatus):
    respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(207, text=multistatus))

    scan = client.list("/homes/scanner")[0]

    assert scan.size == 1234
    assert scan.last_modified is not None
    assert scan.last_modified.isoformat() == "2026-08-16T09:00:00+00:00"


@respx.mock
def test_list_sends_depth_1(client):
    route = respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(207, text=SABRE_STYLE))

    client.list("/homes/scanner")

    assert route.calls.last.request.headers["Depth"] == "1"


@respx.mock
def test_list_rejects_entity_declarations(client):
    """Refuse the billion-laughs shape rather than expanding it."""
    bomb = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE multistatus [<!ENTITY a "aaaaaaaaaa">]>'
        '<d:multistatus xmlns:d="DAV:"><d:response><d:href>&a;</d:href></d:response></d:multistatus>'
    )
    respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(207, text=bomb))

    with pytest.raises(Exception):
        client.list("/homes/scanner")


@respx.mock
def test_get_uses_the_server_href_verbatim(client, multistatus):
    """The href is already percent-encoded; re-encoding it would break the fetch."""
    respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(207, text=multistatus))
    route = respx.get("http://dav.test/dav.php/homes/scanner/scan%20001.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4")
    )

    entry = client.list("/homes/scanner")[0]

    assert client.get(entry.href) == b"%PDF-1.4"
    assert route.called


@respx.mock
def test_get_returns_none_when_already_gone(client):
    respx.get(f"{LIST_URL}/gone.pdf").mock(return_value=httpx.Response(404))

    assert client.get("/dav.php/homes/scanner/gone.pdf") is None


@respx.mock
def test_get_raises_on_server_error(client):
    respx.get(f"{LIST_URL}/boom.pdf").mock(return_value=httpx.Response(500))

    with pytest.raises(httpx.HTTPStatusError):
        client.get("/dav.php/homes/scanner/boom.pdf")


@respx.mock
def test_delete_tolerates_a_missing_resource(client):
    respx.delete(f"{LIST_URL}/gone.pdf").mock(return_value=httpx.Response(404))

    client.delete("/dav.php/homes/scanner/gone.pdf")


@respx.mock
def test_delete_raises_on_failure(client):
    respx.delete(f"{LIST_URL}/locked.pdf").mock(return_value=httpx.Response(423))

    with pytest.raises(httpx.HTTPStatusError):
        client.delete("/dav.php/homes/scanner/locked.pdf")


@respx.mock
def test_requests_carry_basic_auth(client):
    route = respx.request("PROPFIND", LIST_URL).mock(return_value=httpx.Response(207, text=SABRE_STYLE))

    client.list("/homes/scanner")

    assert route.calls.last.request.headers["Authorization"].startswith("Basic ")
