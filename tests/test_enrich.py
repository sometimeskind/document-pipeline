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
        return enrich.enrich_document(client, PAPERLESS, DOC_ID, MARKER_ID, dry_run=dry_run)


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
        "correspondent": None,
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
def test_a_curated_title_is_marked_but_never_retitled():
    _mock_document(title="Geburtsurkunde", original="upload_kMvk1i.pdf", tags=(3,))
    suggestions = _mock_suggestions()
    patch = _mock_patch()

    result = _enrich()

    assert result.outcome == "skipped-curated-title"
    assert not suggestions.called  # costs no LLM call at all
    # Marked, so the sweep converges instead of re-reading it every hour forever.
    assert json.loads(patch.calls.last.request.content) == {"tags": [3, MARKER_ID]}


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
def test_dry_run_does_not_mark_a_curated_document():
    """No marker means no rename and no state change — a dry run is repeatable."""
    _mock_document(title="Geburtsurkunde", original="upload_kMvk1i.pdf")
    patch = _mock_patch()

    assert _enrich(dry_run=True).outcome == "skipped-curated-title"
    assert not patch.called


@respx.mock
def test_dry_run_does_not_mark_a_short_content_document():
    _mock_document(content="too short")
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

