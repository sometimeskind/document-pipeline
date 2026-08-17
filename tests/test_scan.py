"""Tests for document_pipeline.scan — the delete-only-when-really-ingested contract."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from document_pipeline import scan
from document_pipeline.webdav import WebDAVClient, WebDAVEntry


PAPERLESS = "http://paperless"
TASK_ID = "a1b2c3d4-e5f6-7890-1234-567890abcdef"


def _entry(name: str, *, is_collection: bool = False, age_hours: float = 1.0) -> WebDAVEntry:
    return WebDAVEntry(
        path=f"homes/scanner/{name}",
        name=name,
        is_collection=is_collection,
        size=1024,
        last_modified=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=age_hours),
    )


@pytest.fixture
def webdav():
    client = MagicMock(spec=WebDAVClient)
    client.get.return_value = b"%PDF-1.4"
    return client


def _mock_paperless(*, tag_results=({"id": 7, "name": "scanner"},), task_status="success", result_data=None):
    """Register the three Paperless routes and hand back the post_document one."""
    respx.get(f"{PAPERLESS}/api/tags/", params={"name__iexact": "scanner"}).mock(
        return_value=httpx.Response(200, json={"results": list(tag_results)})
    )
    respx.get(f"{PAPERLESS}/api/tasks/", params={"task_id": TASK_ID}).mock(
        return_value=httpx.Response(200, json=[{"status": task_status, "result_data": result_data}])
    )
    return respx.post(f"{PAPERLESS}/api/documents/post_document/").mock(
        return_value=httpx.Response(200, json={"task_id": TASK_ID})
    )


def _ingest(webdav, **kwargs):
    return scan.ingest_scans(
        webdav,
        scan_path="/homes/scanner",
        paperless_url=PAPERLESS,
        paperless_token="tok",
        poll_interval=0,
        **kwargs,
    )


@respx.mock
def test_successful_ingest_deletes_the_source_file(webdav):
    webdav.list.return_value = [_entry("scan001.pdf")]
    _mock_paperless()

    result = _ingest(webdav)

    assert (result.ingested, result.failed, result.pending) == (1, 0, 0)
    webdav.delete.assert_called_once_with("homes/scanner/scan001.pdf")


@respx.mock
def test_document_is_tagged_on_submission(webdav):
    webdav.list.return_value = [_entry("scan001.pdf")]
    post = _mock_paperless()

    _ingest(webdav)

    assert b'name="tags"\r\n\r\n7' in post.calls.last.request.content


@respx.mock
def test_missing_tag_is_created(webdav):
    webdav.list.return_value = [_entry("scan001.pdf")]
    _mock_paperless(tag_results=())
    create = respx.post(f"{PAPERLESS}/api/tags/").mock(return_value=httpx.Response(201, json={"id": 12}))

    _ingest(webdav)

    assert create.called


@respx.mock
def test_failed_consume_task_leaves_the_file_in_place(webdav):
    """A 2xx POST only means queued — a failed consume must not cost the only copy."""
    webdav.list.return_value = [_entry("scan001.pdf")]
    _mock_paperless(task_status="failure")

    result = _ingest(webdav)

    webdav.delete.assert_not_called()
    assert (result.ingested, result.failed, result.pending) == (0, 1, 1)


@respx.mock
def test_duplicate_failure_counts_as_ingested(webdav):
    """The timeout -> re-POST -> DELETE_DUPLICATES path: the document is in
    Paperless, so the file must be cleared rather than retried forever."""
    webdav.list.return_value = [_entry("scan001.pdf")]
    _mock_paperless(task_status="failure", result_data={"duplicate_of": 42})

    result = _ingest(webdav)

    assert result.ingested == 1
    webdav.delete.assert_called_once()


@respx.mock
def test_duplicate_reported_in_legacy_result_string_counts_as_ingested(webdav):
    webdav.list.return_value = [_entry("scan001.pdf")]
    respx.get(f"{PAPERLESS}/api/tags/", params={"name__iexact": "scanner"}).mock(
        return_value=httpx.Response(200, json={"results": [{"id": 7}]})
    )
    respx.post(f"{PAPERLESS}/api/documents/post_document/").mock(
        return_value=httpx.Response(200, json={"task_id": TASK_ID})
    )
    respx.get(f"{PAPERLESS}/api/tasks/", params={"task_id": TASK_ID}).mock(
        return_value=httpx.Response(
            200, json=[{"status": "failure", "result": "scan001.pdf: Not consuming: It is a duplicate of foo (#42)"}]
        )
    )

    assert _ingest(webdav).ingested == 1


@respx.mock
def test_non_terminal_task_times_out_and_leaves_the_file(webdav):
    webdav.list.return_value = [_entry("scan001.pdf")]
    _mock_paperless(task_status="started")

    result = _ingest(webdav, poll_timeout=0)

    webdav.delete.assert_not_called()
    assert result.failed == 1


@respx.mock
def test_one_bad_file_does_not_abort_the_batch(webdav):
    webdav.list.return_value = [_entry("bad.pdf"), _entry("good.pdf")]
    webdav.get.side_effect = [httpx.ReadTimeout("boom"), b"%PDF-1.4"]
    _mock_paperless()

    result = _ingest(webdav)

    assert (result.ingested, result.failed) == (1, 1)
    webdav.delete.assert_called_once_with("homes/scanner/good.pdf")


@respx.mock
def test_a_file_that_vanished_mid_run_is_not_a_failure(webdav):
    """The sweep and a trigger-driven run can list the same file; whoever loses
    the race gets a 404 and nothing is lost."""
    webdav.list.return_value = [_entry("scan001.pdf")]
    webdav.get.return_value = None
    _mock_paperless()

    result = _ingest(webdav)

    assert (result.ingested, result.failed, result.pending) == (1, 0, 0)
    webdav.delete.assert_not_called()


@respx.mock
def test_ineligible_files_are_ignored_and_never_hold_the_alert_open(webdav):
    webdav.list.return_value = [
        _entry(".hidden.pdf"),
        _entry("notes.txt"),
        _entry("scan001.pdf.part"),
        _entry("2026-08", is_collection=True),
    ]

    result = _ingest(webdav)

    assert (result.ingested, result.failed, result.pending) == (0, 0, 0)
    assert result.ignored == 3
    assert result.oldest_pending_age_seconds == 0.0
    webdav.delete.assert_not_called()


@respx.mock
def test_oldest_pending_age_reports_the_stalest_leftover(webdav):
    webdav.list.return_value = [_entry("old.pdf", age_hours=9), _entry("new.pdf", age_hours=1)]
    _mock_paperless(task_status="failure")

    result = _ingest(webdav)

    assert result.pending == 2
    assert 9 * 3600 <= result.oldest_pending_age_seconds < 10 * 3600


@pytest.mark.parametrize(
    "name,eligible",
    [
        ("scan001.pdf", True),
        ("SCAN001.PDF", True),
        ("scan001.jpg", True),
        ("scan001.tiff", True),
        (".scan001.pdf", False),
        ("scan001.pdf.part", False),
        ("notes.txt", False),
        ("scan001", False),
    ],
)
def test_eligibility(name, eligible):
    assert scan.is_eligible(name) is eligible
