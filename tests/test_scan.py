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
SCAN_PATH = "/homes/scanner"
TASK_ID = "a1b2c3d4-e5f6-7890-1234-567890abcdef"


def _entry(name: str, *, is_collection: bool = False, age_hours: float = 1.0) -> WebDAVEntry:
    return WebDAVEntry(
        path=f"homes/scanner/{name}",
        name=name,
        is_collection=is_collection,
        size=1024,
        last_modified=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=age_hours),
    )


def _listing(webdav, root=(), mail=()):
    """One drain lists the scanner root and the mail queue, each by its own path."""
    webdav.list.side_effect = lambda path: {SCAN_PATH: list(root), f"{SCAN_PATH}/mail": list(mail)}[path]


@pytest.fixture
def webdav():
    client = MagicMock(spec=WebDAVClient)
    client.get.return_value = b"%PDF-1.4"
    _listing(client)
    return client


def _mock_paperless(*, tag_results=({"id": 7, "name": "scanner"},), task_status="success", result_data=None, tag="scanner"):
    """Register the three Paperless routes and hand back the post_document one."""
    respx.get(f"{PAPERLESS}/api/tags/", params={"name__iexact": tag}).mock(
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
        scan_path=SCAN_PATH,
        paperless_url=PAPERLESS,
        paperless_token="tok",
        poll_interval=0,
        **kwargs,
    )


@respx.mock
def test_successful_ingest_deletes_the_source_file(webdav):
    _listing(webdav, [_entry("scan001.pdf")])
    _mock_paperless()

    result = _ingest(webdav)

    assert (result.ingested, result.failed, result.pending) == (1, 0, 0)
    webdav.delete.assert_called_once_with("homes/scanner/scan001.pdf")


@respx.mock
def test_document_is_tagged_on_submission(webdav):
    _listing(webdav, [_entry("scan001.pdf")])
    post = _mock_paperless()

    _ingest(webdav)

    assert b'name="tags"\r\n\r\n7' in post.calls.last.request.content


@respx.mock
def test_missing_tag_is_created(webdav):
    _listing(webdav, [_entry("scan001.pdf")])
    _mock_paperless(tag_results=())
    create = respx.post(f"{PAPERLESS}/api/tags/").mock(return_value=httpx.Response(201, json={"id": 12}))

    _ingest(webdav)

    assert create.called


@respx.mock
def test_failed_consume_task_leaves_the_file_in_place(webdav):
    """A 2xx POST only means queued — a failed consume must not cost the only copy."""
    _listing(webdav, [_entry("scan001.pdf")])
    _mock_paperless(task_status="failure")

    result = _ingest(webdav)

    webdav.delete.assert_not_called()
    assert (result.ingested, result.failed, result.pending) == (0, 1, 1)


@respx.mock
def test_duplicate_failure_counts_as_ingested(webdav):
    """The timeout -> re-POST -> DELETE_DUPLICATES path: the document is in
    Paperless, so the file must be cleared rather than retried forever."""
    _listing(webdav, [_entry("scan001.pdf")])
    _mock_paperless(task_status="failure", result_data={"duplicate_of": 42})

    result = _ingest(webdav)

    assert result.ingested == 1
    webdav.delete.assert_called_once()


@respx.mock
def test_duplicate_reported_in_legacy_result_string_counts_as_ingested(webdav):
    _listing(webdav, [_entry("scan001.pdf")])
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
    _listing(webdav, [_entry("scan001.pdf")])
    _mock_paperless(task_status="started")

    result = _ingest(webdav, poll_timeout=0)

    webdav.delete.assert_not_called()
    assert result.failed == 1


@respx.mock
def test_one_bad_file_does_not_abort_the_batch(webdav):
    _listing(webdav, [_entry("bad.pdf"), _entry("good.pdf")])
    webdav.get.side_effect = [httpx.ReadTimeout("boom"), b"%PDF-1.4"]
    _mock_paperless()

    result = _ingest(webdav)

    assert (result.ingested, result.failed) == (1, 1)
    webdav.delete.assert_called_once_with("homes/scanner/good.pdf")


@respx.mock
def test_a_file_that_vanished_mid_run_is_not_a_failure(webdav):
    """The sweep and a trigger-driven run can list the same file; whoever loses
    the race gets a 404 and nothing is lost."""
    _listing(webdav, [_entry("scan001.pdf")])
    webdav.get.return_value = None
    _mock_paperless()

    result = _ingest(webdav)

    assert (result.ingested, result.failed, result.pending) == (1, 0, 0)
    webdav.delete.assert_not_called()


@respx.mock
def test_ineligible_files_are_ignored_and_never_hold_the_alert_open(webdav):
    _listing(webdav, [
        _entry(".hidden.pdf"),
        _entry("notes.txt"),
        _entry("scan001.pdf.part"),
        _entry("2026-08", is_collection=True),
    ])

    result = _ingest(webdav)

    assert (result.ingested, result.failed, result.pending) == (0, 0, 0)
    assert result.ignored == 3
    assert result.oldest_pending_age_seconds == 0.0
    webdav.delete.assert_not_called()


@respx.mock
def test_oldest_pending_age_reports_the_stalest_leftover(webdav):
    _listing(webdav, [_entry("old.pdf", age_hours=9), _entry("new.pdf", age_hours=1)])
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


# --- the mail queue (homelab#1590) ---

def _mail_entry(name: str, **kwargs) -> WebDAVEntry:
    entry = _entry(name, **kwargs)
    return WebDAVEntry(
        path=f"homes/scanner/mail/{name}", name=name, is_collection=entry.is_collection,
        size=entry.size, last_modified=entry.last_modified,
    )


@respx.mock
def test_mail_queue_files_are_tagged_mail_and_counted_under_their_source(webdav):
    _listing(webdav, root=[_entry("scan001.pdf")], mail=[_mail_entry("42-invoice.pdf")])
    post = _mock_paperless()
    _mock_paperless(tag_results=({"id": 9, "name": "mail"},), tag="mail")

    result = _ingest(webdav)

    tags = [c.request.content.split(b'name="tags"\r\n\r\n')[1][:1] for c in post.calls]
    assert tags == [b"7", b"9"]
    assert (result.sources["scanner"].ingested, result.sources["mail"].ingested) == (1, 1)
    assert result.ingested == 2
    webdav.delete.assert_any_call("homes/scanner/mail/42-invoice.pdf")


@respx.mock
def test_both_sources_are_reported_even_when_one_is_empty(webdav):
    _listing(webdav, mail=[_mail_entry("42-invoice.pdf")])
    _mock_paperless(task_status="failure", tag="mail")

    result = _ingest(webdav)

    assert set(result.sources) == {"scanner", "mail"}
    assert (result.sources["scanner"].failed, result.sources["scanner"].pending) == (0, 0)
    assert (result.sources["mail"].failed, result.sources["mail"].pending) == (1, 1)
    assert result.sources["mail"].oldest_pending_age_seconds > 0
    assert result.sources["scanner"].oldest_pending_age_seconds == 0.0


@respx.mock
def test_the_mail_collection_in_the_root_listing_is_skipped(webdav, caplog):
    """The root listing now always carries the mail/ collection; it is drained
    as its own source, not warned about as an unknown subdirectory."""
    _listing(webdav, root=[_entry("mail", is_collection=True), _entry("2026-08", is_collection=True)])

    result = _ingest(webdav)

    assert result.ingested == 0
    assert "'mail'" not in caplog.text
    assert "'2026-08'" in caplog.text


def test_source_path_places_the_mail_queue_under_the_scan_path():
    assert scan.source_path("/file/scanner@prins.id", "mail") == "/file/scanner@prins.id/mail"
    assert scan.source_path("/file/scanner@prins.id/", "mail") == "/file/scanner@prins.id/mail"
    assert scan.source_path("/file/scanner@prins.id", "scanner") == "/file/scanner@prins.id"


@respx.mock
def test_a_file_that_lands_mid_run_is_ingested_by_the_same_run(webdav):
    """`/trigger-scan` answers 202 while a run is in flight on the promise that
    the run picks the upload up; a run that listed once could not keep it (#56)."""
    first, second = _entry("scan001.pdf"), _entry("scan002.pdf")
    root_listings = iter([[first], [second], []])
    webdav.list.side_effect = lambda path: next(root_listings) if path == SCAN_PATH else []
    post = _mock_paperless()

    result = _ingest(webdav)

    assert (result.ingested, result.failed, result.pending) == (2, 0, 0)
    assert post.call_count == 2
    assert webdav.list.call_count == 4  # three root listings, one for mail/


@respx.mock
def test_a_failing_file_is_tried_once_and_does_not_loop_the_run(webdav):
    _listing(webdav, [_entry("scan001.pdf")])
    post = _mock_paperless(task_status="failure")

    result = _ingest(webdav)

    assert (result.ingested, result.failed, result.pending) == (0, 1, 1)
    assert post.call_count == 1
    assert webdav.list.call_count == 3  # root, root again (nothing new), mail/
