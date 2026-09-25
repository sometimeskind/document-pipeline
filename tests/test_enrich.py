"""Tests for document_pipeline.enrich — the behaviours the shell hook never had.

The tag-union, the empty-title bail, the short-content skip and the PATCH payload
shape were all untested shell in post-consume.sh. They are the reason this moved
into Python, so they are what these tests pin down.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import httpx
import pytest
import respx

from document_pipeline import enrich


PAPERLESS = "http://paperless"
DOC_ID = 42
QUEUE_ID = 9

_LONG_CONTENT = "x" * enrich.MIN_CONTENT_CHARS


@pytest.fixture(autouse=True)
def results_path(tmp_path, monkeypatch):
    """Every write path appends a pre-PATCH record (#1562) — keep it off /state."""
    target = tmp_path / "results.jsonl"
    monkeypatch.setenv("ENRICH_RESULTS_PATH", str(target))
    return target


def _records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _client():
    return enrich.open_client("tok", suggest_timeout=5.0)


def _mock_document(
    *, content=_LONG_CONTENT, tags=(3,), title="scan_0042", original="scan_0042.pdf",
    correspondent=None,
):
    """A document as paperless's consumer leaves it: title == filename stem."""
    return respx.get(f"{PAPERLESS}/api/documents/{DOC_ID}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": DOC_ID,
                "title": title,
                "content": content,
                "tags": list(tags),
                "original_file_name": original,
                "correspondent": correspondent,
            },
        )
    )


def _mock_suggestions(
    *, title="Invoice from Hermes", tags=(5,), suggested_tags=("shipping",),
    correspondents=(), suggested_correspondents=(),
):
    return respx.get(f"{PAPERLESS}/api/documents/{DOC_ID}/ai_suggestions/").mock(
        return_value=httpx.Response(
            200,
            json={
                "title": title,
                "tags": list(tags),
                "suggested_tags": list(suggested_tags),
                "correspondents": list(correspondents),
                "suggested_correspondents": list(suggested_correspondents),
            },
        )
    )


def _mock_correspondent_search(*, results=()):
    return respx.get(f"{PAPERLESS}/api/correspondents/").mock(
        return_value=httpx.Response(200, json={"results": list(results)})
    )


def _mock_correspondent_create(correspondent_id=17):
    return respx.post(f"{PAPERLESS}/api/correspondents/").mock(
        return_value=httpx.Response(201, json={"id": correspondent_id})
    )


def _mock_patch():
    return respx.patch(f"{PAPERLESS}/api/documents/{DOC_ID}/").mock(
        return_value=httpx.Response(200, json={"id": DOC_ID})
    )


def _enrich(*, dry_run=False):
    with _client() as client:
        return enrich.enrich_document(client, PAPERLESS, DOC_ID, QUEUE_ID, dry_run=dry_run)


# --- merge_tags: the union that keeps a PATCH from destroying existing tags ---

def test_merge_tags_unions_existing_and_added():
    assert enrich.merge_tags([3, 1], [5, 3]) == [1, 3, 5]


def test_merge_tags_keeps_existing_tags_when_the_llm_matches_none():
    """PATCHing tags REPLACES the list — the scan flow's `scanner` tag must survive."""
    assert enrich.merge_tags([7], []) == [7]


# --- converged_tags: the one place "done" is applied to a tag list (#1561) ---

def test_converged_tags_removes_the_queue_tag():
    assert enrich.converged_tags([QUEUE_ID, 7, 3], QUEUE_ID) == [3, 7]


def test_converged_tags_is_idempotent_when_queue_is_absent():
    assert enrich.converged_tags([7, 3], QUEUE_ID) == [3, 7]


# --- title normalisation ---

def test_normalize_title_collapses_whitespace():
    assert enrich.normalize_title("  Invoice\n  from   Hermes ") == "Invoice from Hermes"


def test_normalize_title_truncates_to_the_column_width():
    assert len(enrich.normalize_title("a" * 300)) == enrich.MAX_TITLE_CHARS


# --- enrich_document ---

@respx.mock
def test_patch_payload_carries_the_title_and_the_unioned_tags_without_queue():
    _mock_document(tags=(3, QUEUE_ID))
    _mock_suggestions(tags=(5,))
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "enriched"
    assert json.loads(patch.calls.last.request.content) == {
        "tags": [3, 5],
        "title": "Invoice from Hermes",
    }


@respx.mock
def test_the_trigger_path_enriches_a_document_that_never_got_queue():
    """A UI upload can dodge the workflow; the trigger must not require the tag."""
    _mock_document(tags=(3,))
    suggestions = _mock_suggestions(tags=(5,))
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "enriched"
    assert suggestions.called
    assert json.loads(patch.calls.last.request.content)["tags"] == [3, 5]


@respx.mock
def test_queue_matched_by_the_model_is_still_stripped():
    """Convergence is applied last, so it wins over the matched tags."""
    _mock_document(tags=(3, QUEUE_ID))
    _mock_suggestions(tags=(5, QUEUE_ID))
    patch = _mock_patch()

    _enrich()

    assert json.loads(patch.calls.last.request.content)["tags"] == [3, 5]


@respx.mock
def test_suggested_tags_are_recorded_but_never_applied():
    """Applying unmatched names would let the LLM grow the vocabulary per document."""
    _mock_document(tags=())
    _mock_suggestions(tags=(5,), suggested_tags=("shipping", "hermes"))
    patch = _mock_patch()

    result = _enrich()

    assert result.suggested_tags == ["shipping", "hermes"]
    assert json.loads(patch.calls.last.request.content)["tags"] == [5]


# --- the before-state record (#1562): what `rollback` replays ---

@respx.mock
def test_an_enriched_result_carries_the_before_and_after_state():
    _mock_document(tags=(3, QUEUE_ID), title="scan_0042", correspondent=None)
    _mock_suggestions(tags=(5,), correspondents=(8,))
    respx.get(f"{PAPERLESS}/api/correspondents/8/").mock(
        return_value=httpx.Response(200, json={"id": 8, "name": "Hermes"})
    )
    _mock_patch()

    result = _enrich()

    assert result.previous_title == "scan_0042"
    assert result.previous_tags == [3, QUEUE_ID]
    assert result.previous_correspondent is None
    # The after-state as ids, so rollback can tell whether anyone edited since.
    assert result.tags == [3, 5]
    assert result.correspondent_id == 8


@respx.mock
def test_a_tags_only_write_carries_the_before_state_too():
    _mock_document(content="", tags=(3,), title="scan_0042", correspondent=4)
    _mock_patch()

    result = _enrich()

    assert result.outcome == "skipped-short-content"
    assert (result.previous_title, result.previous_tags, result.previous_correspondent) == (
        "scan_0042", [3], 4,
    )


@respx.mock
def test_the_record_is_written_before_the_patch(results_path):
    """A crash mid-PATCH must leave evidence: the intent is on disk first."""
    _mock_document(tags=(3,))
    _mock_suggestions(tags=(5,))
    seen_at_patch_time = []

    def respond(request):
        seen_at_patch_time.extend(_records(results_path))
        return httpx.Response(200, json={"id": DOC_ID})

    respx.patch(f"{PAPERLESS}/api/documents/{DOC_ID}/").mock(side_effect=respond)

    _enrich()

    assert len(seen_at_patch_time) == 1
    record = seen_at_patch_time[0]
    assert record["outcome"] == enrich.PENDING_OUTCOME
    assert record["document_id"] == DOC_ID
    assert record["previous_title"] == "scan_0042"
    assert record["previous_tags"] == [3]
    assert record["title"] == "Invoice from Hermes"
    assert record["tags"] == [3, 5]


@respx.mock
def test_a_failed_patch_still_leaves_the_pending_record(results_path):
    _mock_document(tags=(3,))
    _mock_suggestions(tags=(5,))
    respx.patch(f"{PAPERLESS}/api/documents/{DOC_ID}/").mock(
        return_value=httpx.Response(500)
    )

    with pytest.raises(httpx.HTTPStatusError):
        _enrich()

    assert [r["outcome"] for r in _records(results_path)] == [enrich.PENDING_OUTCOME]


@respx.mock
def test_a_dry_run_writes_no_pending_record(results_path):
    _mock_document(tags=(3,))
    _mock_suggestions(tags=(5,))

    _enrich(dry_run=True)

    assert _records(results_path) == []


# --- correspondents (#1363) ---

OLLAMA = "http://ollama"


def _fallback_env(monkeypatch, url=OLLAMA, model="qwen-test"):
    monkeypatch.setenv("ENRICH_OLLAMA_URL", url)
    monkeypatch.setenv("ENRICH_OLLAMA_MODEL", model)


def _mock_ollama(name="symbox", title="Factuur van Hermes"):
    """Serve both dedicated queries — they POST the same /api/chat and are only
    distinguishable by which field their schema requires."""
    def respond(request):
        field = json.loads(request.content)["format"]["required"][0]
        value = title if field == "title" else name
        return httpx.Response(
            200, json={"message": {"content": json.dumps({field: value})}}
        )

    return respx.post(f"{OLLAMA}/api/chat").mock(side_effect=respond)


# --- the #1366 fallback: paperless's own pass reliably suggests nothing ---

@respx.mock
def test_fallback_asks_ollama_when_paperless_suggests_nothing(monkeypatch):
    _fallback_env(monkeypatch)
    _mock_document()
    _mock_suggestions()  # no correspondents, no suggested_correspondents
    ollama = _mock_ollama("Cloudflare")
    _mock_correspondent_search(results=())
    create = _mock_correspondent_create(correspondent_id=31)
    patch = _mock_patch()

    result = _enrich()

    request = json.loads(ollama.calls.last.request.content)
    assert request["model"] == "qwen-test"
    assert request["format"]["required"] == ["correspondent"]
    assert json.loads(create.calls.last.request.content) == {"name": "Cloudflare", "owner": None}
    assert json.loads(patch.calls.last.request.content)["correspondent"] == 31
    assert result.correspondent == "Cloudflare"


@respx.mock
def test_fallback_sends_capped_content(monkeypatch):
    _fallback_env(monkeypatch)
    _mock_document(content="x" * 5000)
    _mock_suggestions()
    ollama = _mock_ollama("")
    _mock_patch()

    _enrich()

    prompt = json.loads(ollama.calls.last.request.content)["messages"][0]["content"]
    assert prompt.endswith("x" * enrich.FALLBACK_CONTENT_CHARS)
    assert "x" * (enrich.FALLBACK_CONTENT_CHARS + 1) not in prompt


@respx.mock
def test_fallback_empty_string_means_no_correspondent(monkeypatch):
    """The required field lets the model decline; an invented blank must not create."""
    _fallback_env(monkeypatch)
    _mock_document()
    _mock_suggestions()
    _mock_ollama("")
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "enriched"
    assert result.correspondent is None
    assert "correspondent" not in json.loads(patch.calls.last.request.content)


@respx.mock
def test_fallback_is_not_consulted_when_paperless_suggested_a_name(monkeypatch):
    _fallback_env(monkeypatch)
    _mock_document()
    _mock_suggestions(suggested_correspondents=("symbox",))
    ollama = _mock_ollama()
    _mock_correspondent_search(results=())
    _mock_correspondent_create()
    _mock_patch()

    result = _enrich()

    assert result.correspondent == "symbox"
    # Only the title query reached ollama — no correspondent query fired.
    fields = [json.loads(c.request.content)["format"]["required"] for c in ollama.calls]
    assert fields == [["title"]]


@respx.mock
def test_fallback_failure_never_costs_the_title(monkeypatch):
    _fallback_env(monkeypatch)
    _mock_document()
    _mock_suggestions()
    respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(500))
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "enriched"
    assert result.correspondent is None
    assert json.loads(patch.calls.last.request.content)["title"] == "Invoice from Hermes"


@respx.mock
def test_fallback_is_off_when_unconfigured(monkeypatch):
    """No env vars, no ollama call — the image can land before the manifest."""
    monkeypatch.delenv("ENRICH_OLLAMA_URL", raising=False)
    monkeypatch.delenv("ENRICH_OLLAMA_MODEL", raising=False)
    _mock_document()
    _mock_suggestions()
    patch = _mock_patch()

    result = _enrich()

    assert result.correspondent is None
    assert "correspondent" not in json.loads(patch.calls.last.request.content)


# --- the #43 dedicated title query: paperless's prompt has no language pin ---
# Documents are mocked with a correspondent so that path stays out of the way.

@respx.mock
def test_title_comes_from_the_dedicated_query_not_the_suggestions(monkeypatch):
    _fallback_env(monkeypatch)
    _mock_document(correspondent=1)
    _mock_suggestions(title="Invoice from Hermes")
    ollama = _mock_ollama(title="Factuur van Hermes")
    patch = _mock_patch()

    result = _enrich()

    request = json.loads(ollama.calls.last.request.content)
    assert request["format"]["required"] == ["title"]
    assert "never translate the title" in request["messages"][0]["content"]
    assert json.loads(patch.calls.last.request.content)["title"] == "Factuur van Hermes"
    assert result.title == "Factuur van Hermes"


@respx.mock
def test_title_falls_back_to_suggestions_when_unconfigured(monkeypatch):
    """Pre-#43 behavior when the env is absent — the image can land before the manifest."""
    monkeypatch.delenv("ENRICH_OLLAMA_URL", raising=False)
    monkeypatch.delenv("ENRICH_OLLAMA_MODEL", raising=False)
    _mock_document(correspondent=1)
    _mock_suggestions()
    _mock_patch()

    result = _enrich()

    assert result.title == "Invoice from Hermes"


@respx.mock
def test_title_query_failure_falls_back_to_suggestions(monkeypatch):
    _fallback_env(monkeypatch)
    _mock_document(correspondent=1)
    _mock_suggestions()
    respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(500))
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "enriched"
    assert json.loads(patch.calls.last.request.content)["title"] == "Invoice from Hermes"


@respx.mock
def test_title_query_sends_capped_content(monkeypatch):
    _fallback_env(monkeypatch)
    _mock_document(content="x" * 5000, correspondent=1)
    _mock_suggestions()
    ollama = _mock_ollama()
    _mock_patch()

    _enrich()

    prompt = json.loads(ollama.calls.last.request.content)["messages"][0]["content"]
    assert prompt.endswith("x" * enrich.FALLBACK_CONTENT_CHARS)
    assert "x" * (enrich.FALLBACK_CONTENT_CHARS + 1) not in prompt


@respx.mock
def test_empty_title_from_both_sources_raises(monkeypatch):
    _fallback_env(monkeypatch)
    _mock_document(correspondent=1)
    _mock_suggestions(title="")
    _mock_ollama(title="")
    patch = _mock_patch()

    with pytest.raises(ValueError):
        _enrich()

    assert not patch.calls


@respx.mock
def test_title_and_correspondent_queries_both_fire(monkeypatch):
    """One extra query each, title first — the whole budget of #42 plus #43."""
    _fallback_env(monkeypatch)
    _mock_document()
    _mock_suggestions(title="")
    ollama = _mock_ollama(name="Cloudflare", title="Factuur maart")
    _mock_correspondent_search(results=())
    _mock_correspondent_create(correspondent_id=31)
    patch = _mock_patch()

    result = _enrich()

    fields = [json.loads(c.request.content)["format"]["required"] for c in ollama.calls]
    assert fields == [["title"], ["correspondent"]]
    payload = json.loads(patch.calls.last.request.content)
    assert payload["title"] == "Factuur maart"
    assert payload["correspondent"] == 31
    assert result.correspondent == "Cloudflare"

@respx.mock
def test_a_suggested_correspondent_is_created_unowned_and_assigned():
    """The `owner: None` in the create payload is the #1292 rule: an owned
    correspondent is invisible to paperless's matching on other users' documents."""
    _mock_document()
    _mock_suggestions(suggested_correspondents=("symbox",))
    _mock_correspondent_search(results=())
    create = _mock_correspondent_create(correspondent_id=17)
    patch = _mock_patch()

    result = _enrich()

    assert json.loads(create.calls.last.request.content) == {"name": "symbox", "owner": None}
    assert json.loads(patch.calls.last.request.content)["correspondent"] == 17
    assert result.correspondent == "symbox"


@respx.mock
def test_an_existing_correspondent_is_reused_rather_than_duplicated():
    """A replayed trigger after a failed PATCH must find its own earlier create."""
    _mock_document()
    _mock_suggestions(suggested_correspondents=("symbox",))
    _mock_correspondent_search(results=({"id": 21, "name": "Symbox"},))
    patch = _mock_patch()

    _enrich()

    assert json.loads(patch.calls.last.request.content)["correspondent"] == 21


@respx.mock
def test_a_matched_correspondent_id_wins_over_a_suggested_name():
    _mock_document()
    _mock_suggestions(correspondents=(7,), suggested_correspondents=("Symbox GmbH & Co",))
    respx.get(f"{PAPERLESS}/api/correspondents/7/").mock(
        return_value=httpx.Response(200, json={"id": 7, "name": "Symbox"})
    )
    patch = _mock_patch()

    result = _enrich()

    assert json.loads(patch.calls.last.request.content)["correspondent"] == 7
    assert result.correspondent == "Symbox"


@respx.mock
def test_an_existing_assignment_is_never_overwritten():
    """However a correspondent got onto the document, it outranks the LLM."""
    _mock_document(correspondent=4)
    _mock_suggestions(correspondents=(7,), suggested_correspondents=("symbox",))
    patch = _mock_patch()

    result = _enrich()

    assert "correspondent" not in json.loads(patch.calls.last.request.content)
    assert result.correspondent is None


@respx.mock
def test_dry_run_reports_the_correspondent_without_creating_it():
    """No search, no POST — respx would raise on any unmocked correspondent call."""
    _mock_document()
    _mock_suggestions(suggested_correspondents=("symbox",))

    result = _enrich(dry_run=True)

    assert result.outcome == "dry-run"
    assert result.correspondent == "symbox"


@respx.mock
def test_short_content_skips_the_llm_and_only_strips_queue():
    """Schema-constrained generation always emits a title, so a blank scan would
    get an invented one. Skip it, but converge it so the sweep stops re-picking it."""
    _mock_document(content="too short", tags=(3, QUEUE_ID))
    suggestions = _mock_suggestions()
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "skipped-short-content"
    assert result.title is None
    assert not suggestions.called
    assert json.loads(patch.calls.last.request.content) == {"tags": [3]}


@respx.mock
def test_short_content_without_queue_writes_nothing():
    """Nothing to converge, so no PATCH — and no pointless filename re-render."""
    _mock_document(content="too short", tags=(3,))
    patch = _mock_patch()

    assert _enrich().outcome == "skipped-short-content"
    assert not patch.called


@respx.mock
def test_empty_llm_title_raises_rather_than_patching():
    _mock_document()
    _mock_suggestions(title="   ")
    patch = _mock_patch()

    with pytest.raises(ValueError, match="empty title"):
        _enrich()

    assert not patch.called


@respx.mock
def test_a_replayed_trigger_on_an_enriched_document_is_a_cheap_no_op():
    """No `queue` and an enriched title: the curated-title check makes it free —
    no LLM call and no write."""
    _mock_document(tags=(3,), title="Invoice from Hermes")
    suggestions = _mock_suggestions()
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "skipped-curated-title"
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
def test_find_unenriched_queries_on_the_presence_of_queue():
    route = respx.get(f"{PAPERLESS}/api/documents/").mock(
        return_value=httpx.Response(200, json={"results": [{"id": 1}, {"id": 2}]})
    )

    with _client() as client:
        assert enrich.find_unenriched(client, PAPERLESS, QUEUE_ID, 20) == [1, 2]

    params = route.calls.last.request.url.params
    assert params["tags__id__all"] == str(QUEUE_ID)
    assert "tags__id__none" not in params
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
    first = json.loads(lines[0])
    # Stamped at write time, in UTC — what `rollback --since` filters on.
    recorded_at = datetime.fromisoformat(first.pop("recorded_at"))
    assert recorded_at.utcoffset() == timedelta(0)
    # A superset check: the #1563 gate fields are pinned in their own test.
    assert first.items() >= {
        "document_id": DOC_ID,
        "outcome": "enriched",
        "title": "Invoice from Hermes",
        "matched_tags": [5],
        "suggested_tags": ["shipping"],
        "correspondent": None,
        "duration_seconds": 1.5,
        "previous_title": None,
        "previous_tags": None,
        "previous_correspondent": None,
        "tags": None,
        "correspondent_id": None,
    }.items()


def test_append_result_records_the_gate_fields(tmp_path):
    """Mode and pass count are what the #1563 comparison reads back."""
    target = tmp_path / "results.jsonl"
    enrich.append_result(
        enrich.EnrichResult(document_id=DOC_ID, outcome="dry-run", mode="extract",
                            model_passes=1, created="2026-09-01",
                            created_proposed="2026-09-01"),
        path=str(target),
    )
    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["mode"] == "extract"
    assert record["model_passes"] == 1
    assert record["created"] == "2026-09-01"
    assert record["created_proposed"] == "2026-09-01"


def test_append_result_survives_an_unwritable_path(tmp_path):
    """The record is an artifact, not the job — losing a line must not fail the flow."""
    blocker = tmp_path / "results.jsonl"
    blocker.write_text("")
    enrich.append_result(
        enrich.EnrichResult(document_id=DOC_ID, outcome="enriched"),
        path=str(blocker / "nested.jsonl"),
    )


# --- the curated-title guard: the #1280 backfill's only safety rule ---

def test_a_consumer_generated_title_is_not_curated():
    assert not enrich.has_curated_title(
        {"title": "scan_0042", "original_file_name": "scan_0042.pdf"}
    )


def test_a_long_filename_is_compared_at_the_consumer_truncation_point():
    """consumer.py stores stem[:127], so a longer stem must still compare equal."""
    stem = "a" * 200
    assert not enrich.has_curated_title(
        {"title": stem[:127], "original_file_name": f"{stem}.pdf"}
    )


def test_a_hand_written_title_is_curated():
    assert enrich.has_curated_title(
        {"title": "Geburtsurkunde", "original_file_name": "upload_kMvk1i.pdf"}
    )


def test_a_document_with_no_original_filename_is_enriched_rather_than_skipped():
    """Unprovable is not the same as curated, and a skip is silent and permanent."""
    assert not enrich.has_curated_title({"title": "Anything", "original_file_name": None})


@respx.mock
def test_a_curated_title_is_converged_but_never_retitled():
    _mock_document(title="Geburtsurkunde", original="upload_kMvk1i.pdf", tags=(3, QUEUE_ID))
    suggestions = _mock_suggestions()
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "skipped-curated-title"
    assert not suggestions.called  # costs no LLM call at all
    # Converged, so the sweep stops instead of re-reading it every hour forever.
    assert json.loads(patch.calls.last.request.content) == {"tags": [3]}


# --- dry run: the review pass that must not be able to write ---

@respx.mock
def test_dry_run_reports_the_title_without_patching_anything():
    _mock_document(tags=(3,))
    _mock_suggestions(tags=(5,), suggested_tags=("shipping",))
    patch = _mock_patch()

    result = _enrich(dry_run=True)

    assert result.outcome == "dry-run"
    assert result.title == "Invoice from Hermes"
    assert result.matched_tags == [5]
    assert result.suggested_tags == ["shipping"]
    assert not patch.called


@respx.mock
def test_dry_run_does_not_converge_a_curated_document():
    """`queue` stays, no rename and no state change — a dry run is repeatable."""
    _mock_document(title="Geburtsurkunde", original="upload_kMvk1i.pdf", tags=(3, QUEUE_ID))
    patch = _mock_patch()

    assert _enrich(dry_run=True).outcome == "skipped-curated-title"
    assert not patch.called


@respx.mock
def test_dry_run_does_not_converge_a_short_content_document():
    _mock_document(content="too short", tags=(3, QUEUE_ID))
    patch = _mock_patch()

    assert _enrich(dry_run=True).outcome == "skipped-short-content"
    assert not patch.called


# --- the vocabulary harvest ---

def _write_results(tmp_path, records):
    target = tmp_path / "results.jsonl"
    target.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return str(target)


def test_rank_suggested_tags_ranks_by_document_count(tmp_path):
    path = _write_results(tmp_path, [
        {"document_id": 1, "suggested_tags": ["invoice", "shipping"]},
        {"document_id": 2, "suggested_tags": ["invoice"]},
        {"document_id": 3, "suggested_tags": ["invoice", "tax"]},
    ])

    documents, ranked = enrich.rank_suggested_tags(path)

    assert documents == 3
    assert ranked == [("invoice", 3), ("shipping", 1), ("tax", 1)]


def test_rank_suggested_tags_groups_spellings_the_way_paperless_matches(tmp_path):
    """paperless_ai.matching case-folds before comparing, so ranking must too —
    otherwise one tag splits across two entries and neither looks worth creating."""
    path = _write_results(tmp_path, [
        {"document_id": 1, "suggested_tags": ["Invoice"]},
        {"document_id": 2, "suggested_tags": ["invoice"]},
        {"document_id": 3, "suggested_tags": ["Invoice"]},
    ])

    _, ranked = enrich.rank_suggested_tags(path)

    assert ranked == [("Invoice", 3)]  # most common spelling reported


def test_rank_suggested_tags_counts_a_repeated_name_once_per_document(tmp_path):
    path = _write_results(tmp_path, [{"document_id": 1, "suggested_tags": ["tax", "tax"]}])

    assert enrich.rank_suggested_tags(path)[1] == [("tax", 1)]


def test_rank_suggested_tags_survives_a_torn_final_line(tmp_path):
    """A pod killed mid-write must not cost the whole harvest."""
    target = tmp_path / "results.jsonl"
    target.write_text(
        json.dumps({"document_id": 1, "suggested_tags": ["invoice"]}) + "\n{\"document_",
        encoding="utf-8",
    )

    assert enrich.rank_suggested_tags(str(target)) == (1, [("invoice", 1)])


def test_rank_suggested_tags_ignores_the_pre_patch_record(tmp_path):
    """The pending record repeats the final one; counting both would double every name."""
    path = _write_results(tmp_path, [
        {"document_id": 1, "outcome": enrich.PENDING_OUTCOME, "suggested_tags": ["tax"]},
        {"document_id": 1, "outcome": "enriched", "suggested_tags": ["tax"]},
        {"document_id": 2, "outcome": "enriched", "suggested_tags": ["tax"]},
    ])

    assert enrich.rank_suggested_tags(path) == (2, [("tax", 2)])



# --- the correspondent backfill (#1373) ---

DECLINED_ID = 11


def _backfill(*, dry_run=False):
    with _client() as client:
        return enrich.backfill_correspondent(
            client, PAPERLESS, DOC_ID, DECLINED_ID, dry_run=dry_run
        )


@respx.mock
def test_backfill_patches_only_the_correspondent(monkeypatch):
    """No title, no tags: some of these titles are curated, and the PATCH must
    leave both byte-identical. And no ai_suggestions call at all — the paperless
    pass is known-dry, and respx would raise on the unmocked route."""
    _fallback_env(monkeypatch)
    _mock_document(tags=(3,), title="Curated by hand")
    ollama = _mock_ollama("Cloudflare")
    _mock_correspondent_search(results=())
    create = _mock_correspondent_create(correspondent_id=31)
    patch = _mock_patch()

    result = _backfill()

    fields = [json.loads(c.request.content)["format"]["required"] for c in ollama.calls]
    assert fields == [["correspondent"]]
    assert json.loads(create.calls.last.request.content) == {"name": "Cloudflare", "owner": None}
    assert json.loads(patch.calls.last.request.content) == {"correspondent": 31}
    assert result.outcome == "backfilled"
    assert result.correspondent == "Cloudflare"


@respx.mock
def test_backfill_records_the_before_state_before_the_patch(monkeypatch, results_path):
    _fallback_env(monkeypatch)
    _mock_document(tags=(3,), title="Curated by hand")
    _mock_ollama("Cloudflare")
    _mock_correspondent_search(results=({"id": 31},))
    seen_at_patch_time = []

    def respond(request):
        seen_at_patch_time.extend(_records(results_path))
        return httpx.Response(200, json={"id": DOC_ID})

    respx.patch(f"{PAPERLESS}/api/documents/{DOC_ID}/").mock(side_effect=respond)

    result = _backfill()

    assert [r["outcome"] for r in seen_at_patch_time] == [enrich.PENDING_OUTCOME]
    assert (result.previous_title, result.previous_tags, result.previous_correspondent) == (
        "Curated by hand", [3], None,
    )
    # Title and tags untouched, so None — "this write did not set it".
    assert (result.title, result.tags, result.correspondent_id) == (None, None, 31)
    assert seen_at_patch_time[0]["correspondent_id"] == 31


@respx.mock
def test_backfill_reuses_an_existing_correspondent(monkeypatch):
    _fallback_env(monkeypatch)
    _mock_document(tags=(3,))
    _mock_ollama("symbox")
    _mock_correspondent_search(results=({"id": 21, "name": "Symbox"},))
    patch = _mock_patch()

    _backfill()

    assert json.loads(patch.calls.last.request.content) == {"correspondent": 21}


@respx.mock
def test_backfill_marks_a_declined_document_so_it_is_never_re_queried(monkeypatch):
    """The terminal marker: without it the document matches correspondent__isnull
    again next hour, forever. Existing tags are merged back in, nothing else moves."""
    _fallback_env(monkeypatch)
    _mock_document(tags=(3,))
    _mock_ollama("")
    patch = _mock_patch()

    result = _backfill()

    assert json.loads(patch.calls.last.request.content) == {"tags": [3, DECLINED_ID]}
    assert result.outcome == "declined"
    assert result.correspondent is None


@respx.mock
def test_backfill_marks_an_ocr_floor_document_without_asking(monkeypatch):
    """The sweep converges sub-floor documents untitled, so they land
    in this query too. There is nothing to ask about; mark them, don't query."""
    _fallback_env(monkeypatch)
    _mock_document(content="", tags=(7,))
    ollama = _mock_ollama()
    patch = _mock_patch()

    result = _backfill()

    assert not ollama.called
    assert json.loads(patch.calls.last.request.content) == {"tags": [7, DECLINED_ID]}
    assert result.outcome == "skipped-short-content"


@respx.mock
def test_backfill_raises_on_an_ollama_failure_rather_than_marking(monkeypatch):
    """The one place the fallback's fold-to-None would be wrong: a transient
    timeout marked `no-correspondent` is lost to the backfill for good. Raise,
    write nothing, let Prefect retry."""
    _fallback_env(monkeypatch)
    _mock_document(tags=(7,))
    respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(500))
    patch = _mock_patch()

    with pytest.raises(httpx.HTTPStatusError):
        _backfill()

    assert not patch.called


@respx.mock
def test_backfill_raises_when_unconfigured(monkeypatch):
    """Unlike the fallback, off cannot mean "decline everything"."""
    monkeypatch.delenv("ENRICH_OLLAMA_URL", raising=False)
    monkeypatch.delenv("ENRICH_OLLAMA_MODEL", raising=False)
    _mock_document(tags=(7,))
    patch = _mock_patch()

    with pytest.raises(RuntimeError):
        _backfill()

    assert not patch.called


@respx.mock
def test_backfill_leaves_a_document_that_gained_a_correspondent_alone(monkeypatch):
    """The sweep or a hand edit can get there between the query and the fetch."""
    _fallback_env(monkeypatch)
    _mock_document(tags=(7,), correspondent=4)
    ollama = _mock_ollama()
    patch = _mock_patch()

    result = _backfill()

    assert result.outcome == "already-has-correspondent"
    assert not ollama.called
    assert not patch.called


@respx.mock
def test_backfill_dry_run_reports_without_creating_or_patching(monkeypatch):
    """No search, no POST, no PATCH — respx would raise on any of them."""
    _fallback_env(monkeypatch)
    _mock_document(tags=(7,))
    _mock_ollama("Cloudflare")

    result = _backfill(dry_run=True)

    assert result.outcome == "dry-run"
    assert result.correspondent == "Cloudflare"


@respx.mock
def test_backfill_dry_run_does_not_mark_a_declined_document(monkeypatch):
    _fallback_env(monkeypatch)
    _mock_document(tags=(7,))
    _mock_ollama("")
    patch = _mock_patch()

    result = _backfill(dry_run=True)

    assert result.outcome == "declined"
    assert not patch.called


@respx.mock
def test_find_without_correspondent_queries_enriched_unmarked_documents():
    route = respx.get(f"{PAPERLESS}/api/documents/").mock(
        return_value=httpx.Response(200, json={"results": [{"id": 1}, {"id": 2}]})
    )

    with _client() as client:
        assert enrich.find_without_correspondent(
            client, PAPERLESS, QUEUE_ID, DECLINED_ID, 8
        ) == [1, 2]

    params = route.calls.last.request.url.params
    # Enriched = no `queue` (#1561); paperless excludes each id in the list.
    assert "tags__id__all" not in params
    assert params["tags__id__none"] == f"{QUEUE_ID},{DECLINED_ID}"
    assert params["correspondent__isnull"] == "true"
    assert params["page_size"] == "8"
    assert params["ordering"] == "id"


# --- ENRICH_MODE=extract: one structured query per document (homelab#1563) ---


TODAY = date(2026, 9, 25)

# A fixed vocabulary, keyed the way fetch_tag_vocabulary returns it.
VOCAB = {"invoice": 5, "shipping": 6, "insurance": 7}


def _extract_env(monkeypatch):
    _fallback_env(monkeypatch)
    monkeypatch.setenv("ENRICH_MODE", "extract")


def _mock_extract_document(
    *, content=_LONG_CONTENT, tags=(3,), correspondent=None,
    created="2026-09-20", added="2026-09-20T14:03:12.123456+02:00", **kwargs,
):
    """A freshly consumed document whose created date paperless could not find."""
    return respx.get(f"{PAPERLESS}/api/documents/{DOC_ID}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": DOC_ID,
                "title": kwargs.get("title", "scan_0042"),
                "content": content,
                "tags": list(tags),
                "original_file_name": kwargs.get("original", "scan_0042.pdf"),
                "correspondent": correspondent,
                "created": created,
                "added": added,
            },
        )
    )


def _mock_extraction(
    *, title="Factuur van Hermes", correspondent="Hermes", tags=("Invoice",), created="2026-09-01",
):
    return respx.post(f"{OLLAMA}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": json.dumps({
            "title": title, "correspondent": correspondent,
            "tags": list(tags), "created": created,
        })}})
    )


def _mock_tag_list(pages=({"invoice": 5, "Shipping": 6},)):
    """Serve /api/tags/ as paginated results, one page per dict."""
    responses = []
    for i, page in enumerate(pages):
        more = i + 1 < len(pages)
        responses.append(httpx.Response(200, json={
            "next": f"{PAPERLESS}/api/tags/?page={i + 2}" if more else None,
            "results": [{"id": tag_id, "name": name} for name, tag_id in page.items()],
        }))
    return respx.get(f"{PAPERLESS}/api/tags/").mock(side_effect=responses)


def _extract(*, dry_run=False, vocab=VOCAB, sample=False):
    with _client() as client:
        return enrich.enrich_document(
            client, PAPERLESS, DOC_ID, MARKER_ID, dry_run=dry_run,
            tag_vocabulary=vocab, sample=sample,
        )


# mode selection

def test_mode_defaults_to_suggest(monkeypatch):
    monkeypatch.delenv("ENRICH_MODE", raising=False)
    assert enrich.resolve_mode(None) == "suggest"


def test_mode_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ENRICH_MODE", "extract")
    assert enrich.resolve_mode(None) == "extract"


def test_an_explicit_mode_outranks_the_environment(monkeypatch):
    """A flow-run parameter can dry-run the other path without an env change."""
    monkeypatch.setenv("ENRICH_MODE", "extract")
    assert enrich.resolve_mode("suggest") == "suggest"


def test_an_unknown_mode_is_rejected_rather_than_silently_defaulted(monkeypatch):
    monkeypatch.setenv("ENRICH_MODE", "extarct")
    with pytest.raises(ValueError):
        enrich.resolve_mode(None)


# tag matching against a fixed vocabulary

def test_match_tags_is_case_and_whitespace_insensitive():
    matched, unmatched = enrich.match_tags(["INVOICE", "  shipping "], VOCAB)
    assert matched == [5, 6]
    assert unmatched == []


def test_match_tags_records_unknown_names_and_never_invents_an_id():
    matched, unmatched = enrich.match_tags(["Invoice", "Tax  return", ""], VOCAB)
    assert matched == [5]
    assert unmatched == ["Tax return"]


def test_match_tags_counts_a_repeated_name_once():
    matched, unmatched = enrich.match_tags(["invoice", "Invoice", "tax", "TAX"], VOCAB)
    assert matched == [5]
    assert unmatched == ["tax"]


def test_match_tags_never_matches_the_pipelines_own_markers():
    """The model naming `ai-processed` must not be what marks a document."""
    vocab = {**VOCAB, enrich.MARKER_TAG: MARKER_ID, enrich.NO_CORRESPONDENT_TAG: 11}
    matched, unmatched = enrich.match_tags(
        [enrich.MARKER_TAG, enrich.NO_CORRESPONDENT_TAG.upper()], vocab
    )
    assert matched == []
    assert unmatched == []


@respx.mock
def test_fetch_tag_vocabulary_follows_every_page():
    route = _mock_tag_list(pages=({"invoice": 5}, {"Shipping": 6}))
    with _client() as client:
        vocab = enrich.fetch_tag_vocabulary(client, PAPERLESS)
    assert vocab == {"invoice": 5, "shipping": 6}
    assert route.call_count == 2


# the created-date rule

def _doc(created="2026-09-20", added="2026-09-20T14:03:12+02:00"):
    return {"created": created, "added": added}


def test_created_is_applied_when_paperless_fell_back_to_the_consume_date():
    assert enrich.pick_created("2026-09-01", _doc(), TODAY) == "2026-09-01"


def test_created_is_never_applied_over_a_date_paperless_found_itself():
    assert enrich.pick_created("2026-09-01", _doc(created="2025-03-14"), TODAY) is None


def test_created_in_the_future_is_rejected():
    assert enrich.pick_created("2026-09-26", _doc(), TODAY) is None


def test_created_today_is_accepted():
    assert enrich.pick_created("2026-09-25", _doc(), TODAY) == "2026-09-25"


@pytest.mark.parametrize("raw", ["", "2026-02-30", "01.09.2026", "September 2026",
                                 "20260901", "2026-9-1", "2026-09-01T10:00:00"])
def test_created_that_is_not_a_plain_valid_date_is_rejected(raw):
    assert enrich.pick_created(raw, _doc(), TODAY) is None


def test_created_equal_to_the_current_value_is_not_rewritten():
    assert enrich.pick_created("2026-09-20", _doc(), TODAY) is None


def test_created_is_left_alone_when_paperless_dates_are_missing():
    assert enrich.pick_created("2026-09-01", {"created": None, "added": None}, TODAY) is None


def test_created_accepts_a_datetime_shaped_created_field():
    """Pre-2.16 paperless served `created` as a datetime; compare its date part."""
    assert enrich.pick_created(
        "2026-09-01", _doc(created="2026-09-20T00:00:00+02:00"), TODAY
    ) == "2026-09-01"


# enrich_document in extract mode

@respx.mock
def test_extract_mode_is_one_ollama_query_and_no_ai_suggestions(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3,))
    suggestions = _mock_suggestions()
    ollama = _mock_extraction(tags=("invoice", "Tax return"))
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    result = _extract()

    assert not suggestions.called
    assert ollama.call_count == 1
    request = json.loads(ollama.calls.last.request.content)
    assert request["format"]["required"] == ["title", "correspondent", "tags", "created"]
    assert json.loads(patch.calls.last.request.content) == {
        "tags": [3, 5, MARKER_ID],
        "title": "Factuur van Hermes",
        "correspondent": 17,
        "created": "2026-09-01",
    }
    assert result.outcome == "enriched"
    assert result.mode == "extract"
    assert result.model_passes == 1
    assert result.matched_tags == [5]
    assert result.suggested_tags == ["Tax return"]
    assert result.correspondent == "Hermes"
    assert result.created == "2026-09-01"


@respx.mock
def test_extract_mode_never_creates_a_tag(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(tags=("brand new tag",))
    _mock_correspondent_search(results=({"id": 17},))
    create_tag = respx.post(f"{PAPERLESS}/api/tags/").mock(
        return_value=httpx.Response(201, json={"id": 99})
    )
    patch = _mock_patch()

    result = _extract()

    assert not create_tag.called
    assert json.loads(patch.calls.last.request.content)["tags"] == [3, MARKER_ID]
    assert result.suggested_tags == ["brand new tag"]


@respx.mock
def test_extract_mode_fetches_the_vocabulary_when_not_handed_one(monkeypatch):
    """The trigger path enriches one document per run, so it fetches its own."""
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(tags=("shipping",))
    tags = _mock_tag_list(pages=({"Shipping": 6},))
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    _extract(vocab=None)

    assert tags.call_count == 1
    assert json.loads(patch.calls.last.request.content)["tags"] == [3, 6, MARKER_ID]


@respx.mock
def test_extract_mode_leaves_a_created_date_paperless_found_alone(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document(created="2025-03-14")
    _mock_extraction(created="2026-09-01")
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    result = _extract()

    assert "created" not in json.loads(patch.calls.last.request.content)
    assert result.created is None
    assert result.created_proposed == "2026-09-01"


@respx.mock
def test_extract_mode_never_overwrites_an_existing_correspondent(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document(correspondent=4)
    _mock_extraction()
    search = _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    result = _extract()

    assert not search.called
    assert "correspondent" not in json.loads(patch.calls.last.request.content)
    assert result.correspondent is None


@respx.mock
def test_extract_mode_creates_a_new_correspondent_unowned(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(correspondent="Cloudflare")
    _mock_correspondent_search(results=())
    create = _mock_correspondent_create(correspondent_id=31)
    patch = _mock_patch()

    _extract()

    assert json.loads(create.calls.last.request.content) == {"name": "Cloudflare", "owner": None}
    assert json.loads(patch.calls.last.request.content)["correspondent"] == 31


@respx.mock
def test_extract_mode_empty_title_raises_rather_than_patching(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(title="  ")
    patch = _mock_patch()

    with pytest.raises(ValueError):
        _extract()
    assert not patch.called


@respx.mock
def test_extract_mode_raises_on_an_ollama_failure(monkeypatch):
    """No ai_suggestions to fall back to — the task retry is the fallback."""
    _extract_env(monkeypatch)
    _mock_extract_document()
    respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(500))
    patch = _mock_patch()

    with pytest.raises(httpx.HTTPStatusError):
        _extract()
    assert not patch.called


@respx.mock
def test_extract_mode_raises_when_ollama_is_unconfigured(monkeypatch):
    monkeypatch.setenv("ENRICH_MODE", "extract")
    monkeypatch.delenv("ENRICH_OLLAMA_URL", raising=False)
    monkeypatch.delenv("ENRICH_OLLAMA_MODEL", raising=False)
    _mock_extract_document()
    patch = _mock_patch()

    with pytest.raises(RuntimeError):
        _extract()
    assert not patch.called


@respx.mock
def test_extract_mode_sends_capped_content(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document(content="x" * 5000)
    ollama = _mock_extraction()
    _mock_correspondent_search(results=({"id": 17},))
    _mock_patch()

    _extract()

    prompt = json.loads(ollama.calls.last.request.content)["messages"][0]["content"]
    assert "x" * enrich.FALLBACK_CONTENT_CHARS in prompt
    assert "x" * (enrich.FALLBACK_CONTENT_CHARS + 1) not in prompt


@respx.mock
def test_extract_mode_dry_run_writes_nothing(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(correspondent="Cloudflare", tags=("invoice", "tax"))
    _mock_correspondent_search(results=())
    create = _mock_correspondent_create()
    patch = _mock_patch()

    result = _extract(dry_run=True)

    assert not patch.called
    assert not create.called
    assert result.outcome == "dry-run"
    assert result.mode == "extract"
    assert result.title == "Factuur van Hermes"
    assert result.matched_tags == [5]
    assert result.suggested_tags == ["tax"]
    assert result.correspondent == "Cloudflare"
    assert result.created == "2026-09-01"


@respx.mock
def test_extract_mode_short_content_skips_the_query(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document(content="too short")
    ollama = _mock_extraction()
    _mock_patch()

    result = _extract()

    assert not ollama.called
    assert result.outcome == "skipped-short-content"
    assert result.model_passes == 0


# the default path records its pass count too, so the gate can compare

@respx.mock
def test_suggest_mode_records_its_model_passes(monkeypatch):
    _fallback_env(monkeypatch)
    monkeypatch.delenv("ENRICH_MODE", raising=False)
    _mock_document()
    _mock_suggestions()  # no correspondent -> the fallback query fires too
    _mock_ollama("Cloudflare")
    _mock_correspondent_search(results=())
    _mock_correspondent_create()
    _mock_patch()

    result = _enrich()

    # ai_suggestions (classification + localization) + title + correspondent
    assert result.mode == "suggest"
    assert result.model_passes == 4


@respx.mock
def test_suggest_mode_pass_count_without_the_dedicated_queries(monkeypatch):
    monkeypatch.delenv("ENRICH_MODE", raising=False)
    monkeypatch.delenv("ENRICH_OLLAMA_URL", raising=False)
    monkeypatch.delenv("ENRICH_OLLAMA_MODEL", raising=False)
    _mock_document()
    _mock_suggestions(suggested_correspondents=("Hermes",))
    _mock_correspondent_search(results=({"id": 17},))
    _mock_patch()

    assert _enrich().model_passes == 2


# sample: dry-run re-enrichment of already-processed documents for the gate

@respx.mock
def test_sample_reenriches_a_marked_curated_document_on_a_dry_run(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3, MARKER_ID), title="Factuur", original="scan.pdf",
                           correspondent=4)
    ollama = _mock_extraction()
    patch = _mock_patch()

    result = _extract(dry_run=True, sample=True)

    assert ollama.called
    assert not patch.called
    assert result.outcome == "dry-run"
    # The existing assignment is reported past, so the gate can compare answers.
    assert result.correspondent == "Hermes"


@respx.mock
def test_sample_in_suggest_mode_reenriches_too(monkeypatch):
    monkeypatch.delenv("ENRICH_MODE", raising=False)
    _fallback_env(monkeypatch)
    _mock_document(tags=(MARKER_ID,), title="Factuur", correspondent=4)
    _mock_suggestions()
    _mock_ollama("Hermes")
    patch = _mock_patch()

    with _client() as client:
        result = enrich.enrich_document(
            client, PAPERLESS, DOC_ID, MARKER_ID, dry_run=True, sample=True
        )

    assert not patch.called
    assert result.outcome == "dry-run"
    assert result.correspondent == "Hermes"


def test_sample_refuses_to_write():
    with _client() as client:
        with pytest.raises(ValueError):
            enrich.enrich_document(client, PAPERLESS, DOC_ID, MARKER_ID, sample=True)


# the gate comparison over the JSONL

def test_compare_modes_pairs_the_latest_dry_run_of_each_mode(tmp_path):
    path = _write_results(tmp_path, [
        {"document_id": 1, "outcome": "dry-run", "mode": "suggest", "title": "old",
         "correspondent": "Hermes", "matched_tags": [], "suggested_tags": ["x"],
         "model_passes": 4, "duration_seconds": 90.0},
        {"document_id": 1, "outcome": "dry-run", "mode": "suggest", "title": "Invoice",
         "correspondent": "Hermes", "matched_tags": [5], "suggested_tags": [],
         "model_passes": 4, "duration_seconds": 100.0},
        {"document_id": 1, "outcome": "dry-run", "mode": "extract", "title": "Factuur",
         "correspondent": "hermes", "matched_tags": [5, 6], "suggested_tags": [],
         "model_passes": 1, "duration_seconds": 30.0, "created": "2026-09-01"},
        {"document_id": 2, "outcome": "dry-run", "mode": "extract", "title": "Lonely",
         "model_passes": 1, "duration_seconds": 20.0},
        {"document_id": 3, "outcome": "enriched", "mode": "suggest", "title": "Live"},
        {"document_id": 4, "outcome": "dry-run", "title": "pre-1563 record"},
    ])

    pairs = enrich.compare_modes(path)

    assert [p["document_id"] for p in pairs] == [1]
    pair = pairs[0]
    assert pair["suggest"]["title"] == "Invoice"
    assert pair["extract"]["title"] == "Factuur"
    assert pair["correspondent_agrees"] is True  # case-insensitive
