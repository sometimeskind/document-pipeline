"""Tests for document_pipeline.queue_migration — the one-off move to `queue` (#1561)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from document_pipeline import enrich, queue_migration


PAPERLESS = "http://paperless"
MARKER_ID = 4
QUEUE_ID = 9


def _client():
    return enrich.open_client("tok", suggest_timeout=5.0)


def _mock_tags(*, marker=True, queue=True):
    ids = {}
    if marker:
        ids[queue_migration.LEGACY_MARKER_TAG] = MARKER_ID
    if queue:
        ids[enrich.QUEUE_TAG] = QUEUE_ID

    def respond(request):
        name = request.url.params["name__iexact"]
        results = [{"id": ids[name]}] if name in ids else []
        return httpx.Response(200, json={"results": results})

    return respx.get(f"{PAPERLESS}/api/tags/").mock(side_effect=respond)


def _mock_documents(pages):
    """`pages` is a list of id lists; `next` is set on every page but the last."""
    def respond(request):
        page = int(request.url.params["page"])
        body = {
            "results": [{"id": i} for i in pages[page - 1]],
            "next": f"{PAPERLESS}/api/documents/?page={page + 1}" if page < len(pages) else None,
        }
        return httpx.Response(200, json=body)

    return respx.get(f"{PAPERLESS}/api/documents/").mock(side_effect=respond)


def _mock_bulk_edit():
    return respx.post(f"{PAPERLESS}/api/documents/bulk_edit/").mock(
        return_value=httpx.Response(200, json={"result": "OK"})
    )


def _run(*, write=False):
    with _client() as client:
        return queue_migration.run(client, PAPERLESS, write=write)


@respx.mock
def test_dry_run_lists_the_unmarked_documents_and_writes_nothing():
    _mock_tags()
    docs = _mock_documents([[1, 2]])
    bulk = _mock_bulk_edit()

    m = _run()

    assert m.document_ids == [1, 2]
    assert not m.written
    assert not bulk.called
    params = docs.calls.last.request.url.params
    # Unmarked, and not already queued — so a re-run reports nothing left.
    assert params["tags__id__none"] == f"{MARKER_ID},{QUEUE_ID}"


@respx.mock
def test_write_adds_queue_to_every_unmarked_document_across_pages(monkeypatch):
    monkeypatch.setattr(queue_migration, "CHUNK_SIZE", 2)
    _mock_tags()
    _mock_documents([[1, 2], [3]])
    bulk = _mock_bulk_edit()

    m = _run(write=True)

    assert m.written
    bodies = [json.loads(c.request.content) for c in bulk.calls]
    assert bodies == [
        {"documents": [1, 2], "method": "add_tag", "parameters": {"tag": QUEUE_ID}},
        {"documents": [3], "method": "add_tag", "parameters": {"tag": QUEUE_ID}},
    ]


@respx.mock
def test_write_creates_queue_with_matching_off_and_no_owner():
    _mock_tags(queue=False)
    docs = _mock_documents([[1]])
    create = respx.post(f"{PAPERLESS}/api/tags/").mock(
        return_value=httpx.Response(201, json={"id": QUEUE_ID})
    )
    bulk = _mock_bulk_edit()

    m = _run(write=True)

    assert m.created_queue
    assert json.loads(create.calls.last.request.content) == {
        "name": "queue", "matching_algorithm": 0, "owner": None,
    }
    assert docs.calls.last.request.url.params["tags__id__none"] == str(MARKER_ID)
    assert json.loads(bulk.calls.last.request.content)["parameters"] == {"tag": QUEUE_ID}


@respx.mock
def test_dry_run_never_creates_queue():
    _mock_tags(queue=False)
    _mock_documents([[1]])
    create = respx.post(f"{PAPERLESS}/api/tags/").mock(
        return_value=httpx.Response(201, json={"id": QUEUE_ID})
    )

    m = _run()

    assert m.queue_id is None
    assert not create.called


@respx.mock
def test_nothing_to_do_once_the_marker_is_deleted():
    _mock_tags(marker=False)
    bulk = _mock_bulk_edit()

    m = _run(write=True)

    assert m.marker_id is None
    assert m.document_ids == []
    assert not bulk.called


@respx.mock
def test_cli_is_a_dry_run_by_default(monkeypatch, capsys):
    from document_pipeline import cli

    monkeypatch.setenv("PAPERLESS_URL", PAPERLESS)
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "tok")
    monkeypatch.setattr("sys.argv", ["document_pipeline", "migrate-queue"])
    _mock_tags()
    _mock_documents([[1, 2]])
    bulk = _mock_bulk_edit()

    cli.main()

    out = capsys.readouterr().out
    assert "2 document(s)" in out
    assert "--write" in out
    assert not bulk.called


@respx.mock
def test_cli_write_points_at_the_next_step(monkeypatch, capsys):
    from document_pipeline import cli

    monkeypatch.setenv("PAPERLESS_URL", PAPERLESS)
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "tok")
    monkeypatch.setattr("sys.argv", ["document_pipeline", "migrate-queue", "--write"])
    _mock_tags()
    _mock_documents([[1]])
    _mock_bulk_edit()

    cli.main()

    out = capsys.readouterr().out
    assert "Added 'queue' to 1 document(s)" in out
    assert "delete the 'ai-processed' tag" in out


def test_cli_rejects_unknown_flags(monkeypatch):
    from document_pipeline import cli

    monkeypatch.setattr("sys.argv", ["document_pipeline", "migrate-queue", "--yes"])
    with pytest.raises(SystemExit):
        cli.main()
