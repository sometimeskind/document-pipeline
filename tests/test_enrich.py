"""Tests for document_pipeline.enrich — the behaviours the shell hook never had.

The tag-union, the empty-title bail, the short-content skip and the PATCH payload
shape were all untested shell in post-consume.sh. They are the reason this moved
into Python, so they are what these tests pin down.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from document_pipeline import enrich


PAPERLESS = "http://paperless"
DOC_ID = 42
MARKER_ID = 9

_LONG_CONTENT = "x" * enrich.MIN_CONTENT_CHARS


def _client():
    return enrich.open_client("tok", suggest_timeout=5.0)


def _mock_document(*, content=_LONG_CONTENT, tags=(3,), title="scan_0042"):
    return respx.get(f"{PAPERLESS}/api/documents/{DOC_ID}/").mock(
        return_value=httpx.Response(
            200, json={"id": DOC_ID, "title": title, "content": content, "tags": list(tags)}
        )
    )


def _mock_suggestions(*, title="Invoice from Hermes", tags=(5,), suggested_tags=("shipping",)):
    return respx.get(f"{PAPERLESS}/api/documents/{DOC_ID}/ai_suggestions/").mock(
        return_value=httpx.Response(
            200,
            json={"title": title, "tags": list(tags), "suggested_tags": list(suggested_tags)},
        )
    )


def _mock_patch():
    return respx.patch(f"{PAPERLESS}/api/documents/{DOC_ID}/").mock(
        return_value=httpx.Response(200, json={"id": DOC_ID})
    )


def _enrich():
    with _client() as client:
        return enrich.enrich_document(client, PAPERLESS, DOC_ID, MARKER_ID)


# --- merge_tags: the union that keeps a PATCH from destroying existing tags ---

def test_merge_tags_unions_existing_matched_and_marker():
    assert enrich.merge_tags([3, 1], [5, 3], MARKER_ID) == [1, 3, 5, MARKER_ID]


def test_merge_tags_keeps_existing_tags_when_the_llm_matches_none():
    """PATCHing tags REPLACES the list — the scan flow's `scanner` tag must survive."""
    assert enrich.merge_tags([7], [], MARKER_ID) == [7, MARKER_ID]


def test_merge_tags_is_idempotent_when_the_marker_is_already_present():
    assert enrich.merge_tags([MARKER_ID, 7], [7], MARKER_ID) == [7, MARKER_ID]


# --- title normalisation ---

def test_normalize_title_collapses_whitespace():
    assert enrich.normalize_title("  Invoice\n  from   Hermes ") == "Invoice from Hermes"


def test_normalize_title_truncates_to_the_column_width():
    assert len(enrich.normalize_title("a" * 300)) == enrich.MAX_TITLE_CHARS


# --- enrich_document ---

@respx.mock
def test_patch_payload_carries_the_title_and_the_unioned_tags():
    _mock_document(tags=(3,))
    _mock_suggestions(tags=(5,))
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "enriched"
    assert json.loads(patch.calls.last.request.content) == {
        "tags": [3, 5, MARKER_ID],
        "title": "Invoice from Hermes",
    }


@respx.mock
def test_suggested_tags_are_recorded_but_never_applied():
    """Applying unmatched names would let the LLM grow the vocabulary per document."""
    _mock_document(tags=())
    _mock_suggestions(tags=(5,), suggested_tags=("shipping", "hermes"))
    patch = _mock_patch()

    result = _enrich()

    assert result.suggested_tags == ["shipping", "hermes"]
    assert json.loads(patch.calls.last.request.content)["tags"] == [5, MARKER_ID]


@respx.mock
def test_short_content_skips_the_llm_and_only_marks_the_document():
    """Schema-constrained generation always emits a title, so a blank scan would
    get an invented one. Skip it, but mark it so the sweep stops re-picking it."""
    _mock_document(content="too short")
    suggestions = _mock_suggestions()
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "skipped-short-content"
    assert result.title is None
    assert not suggestions.called
    assert json.loads(patch.calls.last.request.content) == {"tags": [3, MARKER_ID]}


@respx.mock
def test_empty_llm_title_raises_rather_than_patching():
    _mock_document()
    _mock_suggestions(title="   ")
    patch = _mock_patch()

    with pytest.raises(ValueError, match="empty title"):
        _enrich()

    assert not patch.called


@respx.mock
def test_an_already_marked_document_is_a_cheap_no_op():
    """The guard that makes a replayed trigger free."""
    _mock_document(tags=(3, MARKER_ID))
    suggestions = _mock_suggestions()
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "already-enriched"
    assert not suggestions.called
    assert not patch.called


@respx.mock
def test_a_suggestions_failure_propagates_so_prefect_can_retry():
    """The 503 that used to cost a document its title permanently."""
    _mock_document()
    respx.get(f"{PAPERLESS}/api/documents/{DOC_ID}/ai_suggestions/").mock(
        return_value=httpx.Response(503)
    )
    patch = _mock_patch()

    with pytest.raises(httpx.HTTPStatusError):
        _enrich()

    assert not patch.called


# --- find_unenriched ---

@respx.mock
def test_find_unenriched_queries_on_the_absence_of_the_marker():
    route = respx.get(f"{PAPERLESS}/api/documents/").mock(
        return_value=httpx.Response(200, json={"results": [{"id": 1}, {"id": 2}]})
    )

    with _client() as client:
        assert enrich.find_unenriched(client, PAPERLESS, MARKER_ID, 20) == [1, 2]

    params = route.calls.last.request.url.params
    assert params["tags__id__none"] == str(MARKER_ID)
    assert params["page_size"] == "20"
    assert params["ordering"] == "id"


# --- the JSONL harvest artifact ---

def test_append_result_writes_one_json_object_per_line(tmp_path):
    target = tmp_path / "nested" / "results.jsonl"
    enrich.append_result(
        enrich.EnrichResult(
            document_id=DOC_ID,
            outcome="enriched",
            title="Invoice from Hermes",
            matched_tags=[5],
            suggested_tags=["shipping"],
            duration_seconds=1.5,
        ),
        path=str(target),
    )
    enrich.append_result(
        enrich.EnrichResult(document_id=43, outcome="skipped-short-content"), path=str(target)
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["document_id"] for line in lines] == [DOC_ID, 43]
    assert json.loads(lines[0]) == {
        "document_id": DOC_ID,
        "outcome": "enriched",
        "title": "Invoice from Hermes",
        "matched_tags": [5],
        "suggested_tags": ["shipping"],
        "duration_seconds": 1.5,
    }


def test_append_result_survives_an_unwritable_path(tmp_path):
    """The record is an artifact, not the job — losing a line must not fail the flow."""
    blocker = tmp_path / "results.jsonl"
    blocker.write_text("")
    enrich.append_result(
        enrich.EnrichResult(document_id=DOC_ID, outcome="enriched"),
        path=str(blocker / "nested.jsonl"),
    )
