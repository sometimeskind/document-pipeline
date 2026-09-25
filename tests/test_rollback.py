"""Tests for document_pipeline.rollback — replaying the enrich before-state (#1562).

The two rules that make a rollback safe to run over a live library are what these
pin down: a document someone edited since is skipped rather than clobbered, and
a reverted document is left converged so the next sweep does not redo it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest
import respx

from document_pipeline import enrich, rollback


PAPERLESS = "http://paperless"
QUEUE_ID = 9
DECLINED_ID = 11
# The retired `ai-processed` marker (#1561): named by every record written
# before the queue tag, and deleted from paperless by the migration.
DELETED_MARKER_ID = 99
SINCE = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _client():
    return enrich.open_client("tok", suggest_timeout=5.0)


def _record(document_id=42, outcome="enriched", recorded_at="2026-09-10T12:00:00+00:00", **kw):
    """An enriched record: consumer title + tags 3/queue -> LLM title, tags 3/5, correspondent 8."""
    record = {
        "recorded_at": recorded_at,
        "document_id": document_id,
        "outcome": outcome,
        "title": "Invoice from Hermes",
        "matched_tags": [5],
        "suggested_tags": [],
        "correspondent": "Hermes",
        "duration_seconds": 1.0,
        "previous_title": "scan_0042",
        "previous_tags": [3, QUEUE_ID],
        "previous_correspondent": None,
        "tags": [3, 5],
        "correspondent_id": 8,
    }
    record.update(kw)
    return record


def _write(tmp_path, records, torn=False):
    target = tmp_path / "results.jsonl"
    text = "\n".join(json.dumps(r) for r in records) + "\n"
    if torn:
        text += '{"document_'
    target.write_text(text, encoding="utf-8")
    return str(target)


def _mock_tags(*, declined_exists=True, queue_exists=True):
    ids = {}
    if queue_exists:
        ids[enrich.QUEUE_TAG] = QUEUE_ID
    if declined_exists:
        ids[enrich.NO_CORRESPONDENT_TAG] = DECLINED_ID
    # Every id the tests use exists, except the deleted marker.
    existing = {3, 5, 7, 31} | set(ids.values())

    def respond(request):
        params = request.url.params
        if "id__in" in params:
            wanted = [int(i) for i in params["id__in"].split(",")]
            return httpx.Response(
                200, json={"results": [{"id": i} for i in wanted if i in existing]}
            )
        name = params["name__iexact"]
        results = [{"id": ids[name]}] if name in ids else []
        return httpx.Response(200, json={"results": results})

    return respx.get(f"{PAPERLESS}/api/tags/").mock(side_effect=respond)


def _mock_document(document_id=42, *, title="Invoice from Hermes", tags=(3, 5),
                   correspondent=8):
    return respx.get(f"{PAPERLESS}/api/documents/{document_id}/").mock(
        return_value=httpx.Response(
            200,
            json={"id": document_id, "title": title, "tags": list(tags),
                  "correspondent": correspondent},
        )
    )


def _mock_patch(document_id=42):
    return respx.patch(f"{PAPERLESS}/api/documents/{document_id}/").mock(
        return_value=httpx.Response(200, json={"id": document_id})
    )


def _run(records, *, write=False):
    with _client() as client:
        return rollback.run(client, PAPERLESS, records, write=write)


# --- reading the log ---

def test_newest_record_per_document_wins(tmp_path):
    path = _write(tmp_path, [
        _record(42, recorded_at="2026-09-10T12:00:00+00:00", title="First"),
        _record(43),
        _record(42, recorded_at="2026-09-11T12:00:00+00:00", title="Second"),
    ])

    latest = rollback.load_latest(path, SINCE)

    assert sorted(latest) == [42, 43]
    assert latest[42]["title"] == "Second"


def test_only_write_outcomes_are_revertible(tmp_path):
    """A later no-op record must not shadow the write it follows."""
    path = _write(tmp_path, [
        _record(42, outcome="enriched"),
        _record(42, outcome="already-enriched"),
        _record(43, outcome="dry-run"),
        _record(44, outcome="skipped-curated-title"),
        _record(45, outcome="backfilled"),
        _record(46, outcome=enrich.PENDING_OUTCOME),
    ])

    latest = rollback.load_latest(path, SINCE)

    assert sorted(latest) == [42, 45, 46]
    assert latest[42]["outcome"] == "enriched"


def test_since_excludes_older_records_and_unstamped_ones(tmp_path):
    old = _record(42, recorded_at="2026-08-31T23:59:59+00:00")
    unstamped = _record(43)
    del unstamped["recorded_at"]
    path = _write(tmp_path, [old, unstamped, _record(44)])

    assert sorted(rollback.load_latest(path, SINCE)) == [44]


def test_document_filter(tmp_path):
    path = _write(tmp_path, [_record(42), _record(43)])

    assert sorted(rollback.load_latest(path, SINCE, document_id=43)) == [43]


def test_a_torn_final_line_is_skipped(tmp_path):
    path = _write(tmp_path, [_record(42)], torn=True)

    assert sorted(rollback.load_latest(path, SINCE)) == [42]


def test_parse_since_treats_a_naive_timestamp_as_utc():
    assert rollback.parse_since("2026-09-01") == SINCE
    assert rollback.parse_since("2026-09-01T02:00:00+02:00") == SINCE


# --- dry run and write ---

@respx.mock
def test_dry_run_lists_the_reversion_and_writes_nothing():
    _mock_tags()
    _mock_document()
    patch = _mock_patch()

    [outcome] = _run({42: _record()})

    assert outcome.status == "would-revert"
    assert outcome.payload == {
        "title": "scan_0042",
        "tags": [3, DECLINED_ID],
        "correspondent": None,
    }
    assert not patch.called


@respx.mock
def test_dry_run_never_creates_a_missing_tag():
    _mock_tags(declined_exists=False)
    _mock_document()
    create = respx.post(f"{PAPERLESS}/api/tags/").mock(
        return_value=httpx.Response(201, json={"id": DECLINED_ID})
    )

    [outcome] = _run({42: _record()})

    assert outcome.status == "would-revert"
    assert not create.called


@respx.mock
def test_write_reverts_title_tags_and_correspondent():
    """The correspondent goes back to null explicitly — omitting it would keep it."""
    _mock_tags()
    _mock_document()
    patch = _mock_patch()

    [outcome] = _run({42: _record()}, write=True)

    assert outcome.status == "reverted"
    assert json.loads(patch.calls.last.request.content) == {
        "title": "scan_0042",
        "tags": [3, DECLINED_ID],
        "correspondent": None,
    }


@respx.mock
def test_write_creates_the_decline_tag_when_missing():
    _mock_tags(declined_exists=False)
    _mock_document()
    respx.post(f"{PAPERLESS}/api/tags/").mock(
        return_value=httpx.Response(201, json={"id": DECLINED_ID})
    )
    patch = _mock_patch()

    _run({42: _record()}, write=True)

    assert json.loads(patch.calls.last.request.content)["tags"] == [3, DECLINED_ID]


@respx.mock
def test_a_reverted_document_is_left_without_queue_so_the_sweep_does_not_redo_it():
    """The before-state carried `queue`; replaying it verbatim would re-enrich.
    No correspondent was assigned, so no decline tag either."""
    _mock_tags()
    _mock_document(correspondent=4)
    patch = _mock_patch()

    _run({42: _record(previous_correspondent=4, correspondent_id=None)}, write=True)

    assert json.loads(patch.calls.last.request.content) == {
        "title": "scan_0042",
        "tags": [3],
        "correspondent": 4,
    }


@respx.mock
def test_a_backfilled_record_reverts_only_what_it_changed():
    """Title and tags were not written (None), so the expected state is the before-state."""
    _mock_tags()
    _mock_document(title="Curated", tags=(3,), correspondent=31)
    patch = _mock_patch()
    record = _record(
        outcome="backfilled", title=None, tags=None, correspondent_id=31,
        previous_title="Curated", previous_tags=[3], previous_correspondent=None,
    )

    [outcome] = _run({42: record}, write=True)

    assert outcome.status == "reverted"
    assert json.loads(patch.calls.last.request.content) == {
        "title": "Curated",
        "tags": [3, DECLINED_ID],
        "correspondent": None,
    }


def _extract_record(**kw):
    """An extract-mode record (#1563) that also wrote `created`."""
    return _record(mode="extract", created="2026-09-01", previous_created="2026-09-20", **kw)


def _mock_dated_document(created):
    return respx.get(f"{PAPERLESS}/api/documents/42/").mock(return_value=httpx.Response(200, json={
        "id": 42, "title": "Invoice from Hermes", "tags": [3, 5, MARKER_ID],
        "correspondent": 8, "created": created,
    }))


@respx.mock
def test_an_extract_record_reverts_created_too():
    _mock_tags()
    _mock_dated_document("2026-09-01")
    patch = _mock_patch()

    [outcome] = _run({42: _extract_record()}, write=True)

    assert outcome.status == "reverted"
    assert json.loads(patch.calls.last.request.content) == {
        "title": "scan_0042",
        "tags": [3, MARKER_ID, DECLINED_ID],
        "correspondent": None,
        "created": "2026-09-20",
    }


@respx.mock
def test_a_created_date_changed_since_counts_as_an_edit():
    _mock_tags()
    _mock_dated_document("2026-08-15")
    patch = _mock_patch()

    [outcome] = _run({42: _extract_record()}, write=True)

    assert outcome.status == "changed-since"
    assert "created" in outcome.detail
    assert not patch.called


@respx.mock
def test_an_extract_record_that_left_created_alone_does_not_touch_it():
    _mock_tags()
    _mock_dated_document("2026-09-20")
    patch = _mock_patch()

    _run({42: _record(mode="extract", created=None, previous_created="2026-09-20")}, write=True)

    assert "created" not in json.loads(patch.calls.last.request.content)


@respx.mock
def test_a_document_edited_since_is_skipped_and_reported():
    _mock_tags()
    _mock_document(title="Renamed by hand")
    patch = _mock_patch()

    [outcome] = _run({42: _record()}, write=True)

    assert outcome.status == "changed-since"
    assert "title" in outcome.detail
    assert not patch.called


@respx.mock
def test_a_tag_added_since_counts_as_an_edit():
    _mock_tags()
    _mock_document(tags=(3, 5, 7))
    patch = _mock_patch()

    [outcome] = _run({42: _record()}, write=True)

    assert outcome.status == "changed-since"
    assert "tags" in outcome.detail
    assert not patch.called


@respx.mock
def test_a_second_run_reports_already_reverted():
    _mock_tags()
    _mock_document(title="scan_0042", tags=(3, DECLINED_ID), correspondent=None)
    patch = _mock_patch()

    [outcome] = _run({42: _record()}, write=True)

    assert outcome.status == "already-reverted"
    assert not patch.called


@respx.mock
def test_write_never_creates_queue_just_to_leave_it_off():
    _mock_tags(queue_exists=False)
    _mock_document()
    create = respx.post(f"{PAPERLESS}/api/tags/").mock(
        return_value=httpx.Response(201, json={"id": QUEUE_ID})
    )
    patch = _mock_patch()

    [outcome] = _run({42: _record(previous_tags=[3])}, write=True)

    assert outcome.status == "reverted"
    assert not create.called
    assert json.loads(patch.calls.last.request.content)["tags"] == [3, DECLINED_ID]


# --- records from before #1561 name the deleted `ai-processed` marker ---

@respx.mock
def test_a_pre_queue_record_is_not_mistaken_for_an_edit_made_since():
    """Its after-state carries the marker, which the migration deleted from every
    document along with the tag — so the comparison must not count it."""
    _mock_tags()
    _mock_document(tags=(3, 5))
    patch = _mock_patch()
    record = _record(previous_tags=[3], tags=[3, 5, DELETED_MARKER_ID])

    [outcome] = _run({42: record}, write=True)

    assert outcome.status == "reverted"
    assert json.loads(patch.calls.last.request.content)["tags"] == [3, DECLINED_ID]
    assert f"[{DELETED_MARKER_ID}]" in outcome.detail


@respx.mock
def test_a_deleted_tag_in_the_before_state_is_dropped_not_reintroduced():
    """PATCHing an id that no longer exists would 400; dropping it is also right —
    the marker's whole meaning is gone."""
    _mock_tags()
    _mock_document(title="Curated", tags=(3,), correspondent=31)
    patch = _mock_patch()
    record = _record(
        outcome="backfilled", title=None, tags=None, correspondent_id=31,
        previous_title="Curated", previous_tags=[3, DELETED_MARKER_ID],
        previous_correspondent=None,
    )

    [outcome] = _run({42: record}, write=True)

    assert outcome.status == "reverted"
    assert json.loads(patch.calls.last.request.content) == {
        "title": "Curated",
        "tags": [3, DECLINED_ID],
        "correspondent": None,
    }
    assert "dropped deleted tag" in outcome.detail


@respx.mock
def test_a_dry_run_reports_the_dropped_tag_too():
    _mock_tags()
    _mock_document(tags=(3, 5))
    patch = _mock_patch()

    [outcome] = _run({42: _record(previous_tags=[3, DELETED_MARKER_ID])})

    assert outcome.status == "would-revert"
    assert outcome.payload["tags"] == [3, DECLINED_ID]
    assert "dropped deleted tag" in outcome.detail
    assert not patch.called


@respx.mock
def test_a_record_without_before_state_is_skipped():
    _mock_tags()
    patch = _mock_patch()

    [outcome] = _run({42: _record(previous_tags=None)}, write=True)

    assert outcome.status == "no-before-state"
    assert not patch.called


@respx.mock
def test_write_appends_a_rolled_back_record(tmp_path, monkeypatch):
    target = tmp_path / "results.jsonl"
    monkeypatch.setenv("ENRICH_RESULTS_PATH", str(target))
    _mock_tags()
    _mock_document()
    _mock_patch()

    _run({42: _record()}, write=True)

    [record] = [json.loads(line) for line in target.read_text().splitlines()]
    assert record["outcome"] == "rolled-back"
    assert record["previous_title"] == "Invoice from Hermes"
    assert record["title"] == "scan_0042"
    # Not itself revertible, so it can never shadow the record it undid.
    assert "rolled-back" not in rollback.REVERTIBLE_OUTCOMES


# --- the CLI ---

@respx.mock
def test_cli_is_a_dry_run_by_default(tmp_path, monkeypatch, capsys):
    from document_pipeline import cli

    path = _write(tmp_path, [_record()])
    monkeypatch.setenv("ENRICH_RESULTS_PATH", path)
    monkeypatch.setenv("PAPERLESS_URL", PAPERLESS)
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "tok")
    monkeypatch.setattr("sys.argv", ["document_pipeline", "rollback", "--since", "2026-09-01"])
    _mock_tags()
    _mock_document()
    patch = _mock_patch()

    cli.main()

    out = capsys.readouterr().out
    assert "would-revert" in out
    assert "42" in out
    assert "--write" in out
    assert not patch.called


def test_cli_requires_since(monkeypatch):
    from document_pipeline import cli

    monkeypatch.setattr("sys.argv", ["document_pipeline", "rollback"])
    with pytest.raises(SystemExit):
        cli.main()
