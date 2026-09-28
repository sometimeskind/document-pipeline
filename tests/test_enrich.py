"""Tests for document_pipeline.enrich — the behaviours the shell hook never had.

The tag-union, the short-content skip and the PATCH payload shape were all
untested shell in post-consume.sh. They are the reason this moved into Python,
so they are what these tests pin down.
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
    return enrich.open_client("tok")


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


# --- the before-state record (#1562): what `rollback` replays ---

@respx.mock
def test_a_tags_only_write_carries_the_before_state_too():
    _mock_document(content="", tags=(3,), title="scan_0042", correspondent=4)
    _mock_patch()

    result = _enrich()

    assert result.outcome == "skipped-short-content"
    assert (result.previous_title, result.previous_tags, result.previous_correspondent) == (
        "scan_0042", [3], 4,
    )


# --- correspondents (#1363) ---

OLLAMA = "http://ollama"


def _fallback_env(monkeypatch, url=OLLAMA, model="qwen-test"):
    monkeypatch.setenv("ENRICH_OLLAMA_URL", url)
    monkeypatch.setenv("ENRICH_OLLAMA_MODEL", model)


def _mock_ollama(name="symbox"):
    """Serve the backfill's correspondent query."""
    return respx.post(f"{OLLAMA}/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": json.dumps({"correspondent": name})}}
        )
    )


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
    # A superset check: the #1563 fields are pinned in their own test.
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


def test_append_result_records_the_pass_count_and_created(tmp_path):
    target = tmp_path / "results.jsonl"
    enrich.append_result(
        enrich.EnrichResult(document_id=DOC_ID, outcome="dry-run",
                            model_passes=2, created="2026-09-01",
                            created_proposed="2026-09-01"),
        path=str(target),
    )
    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["model_passes"] == 2
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
    patch = _mock_patch()

    result = _enrich()  # respx refuses any unmocked call: no model query at all

    assert result.outcome == "skipped-curated-title"
    # Converged, so the sweep stops instead of re-reading it every hour forever.
    assert json.loads(patch.calls.last.request.content) == {"tags": [3]}


# --- dry run: the review pass that must not be able to write ---

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


# --- enrich_document: a title query and a facts query (homelab#1563) ---


TODAY = date(2026, 9, 25)

# A fixed vocabulary, keyed the way fetch_tag_vocabulary returns it.
VOCAB = {"invoice": 5, "shipping": 6, "insurance": 7}


def _extract_env(monkeypatch):
    _fallback_env(monkeypatch)


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
    title_status=200, facts_status=200,
):
    """Serve extract mode's two queries, told apart by the schema they require:
    the title query (["title"]) and the facts query. `tags` may be any JSON
    value, to feed the model's malformed answers through."""
    def respond(request):
        if json.loads(request.content)["format"]["required"] == ["title"]:
            status, answer = title_status, {"title": title}
        else:
            status, answer = facts_status, {
                "correspondent": correspondent,
                "tags": list(tags) if isinstance(tags, tuple) else tags,
                "created": created,
            }
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json={"message": {"content": json.dumps(answer)}})

    return respx.post(f"{OLLAMA}/api/chat").mock(side_effect=respond)


def _chat_schemas(route) -> list[list[str]]:
    """The `required` list of every /api/chat call, in order."""
    return [json.loads(c.request.content)["format"]["required"] for c in route.calls]


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
            client, PAPERLESS, DOC_ID, QUEUE_ID, dry_run=dry_run,
            tag_vocabulary=vocab, sample=sample,
        )


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


def test_match_tags_never_matches_the_pipelines_own_tags():
    """The model naming `queue` or the decline tag must not be what applies it."""
    vocab = {**VOCAB, enrich.QUEUE_TAG: QUEUE_ID, enrich.NO_CORRESPONDENT_TAG: 11}
    matched, unmatched = enrich.match_tags(
        [enrich.QUEUE_TAG, enrich.NO_CORRESPONDENT_TAG.upper()], vocab
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


# enrich_document

@respx.mock
def test_enrich_is_a_title_and_a_facts_query_and_no_ai_suggestions(monkeypatch):
    """respx refuses any unmocked call, ai_suggestions included."""
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3, QUEUE_ID))
    ollama = _mock_extraction(tags=("invoice", "Tax return"))
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    result = _extract()

    # The title is asked alone, with the default path's prompt (#43); the facts
    # query carries no title at all (#1563: four fields at once lost titles).
    assert _chat_schemas(ollama) == [["title"], ["correspondent", "tags", "created"]]
    title_prompt = json.loads(ollama.calls[0].request.content)["messages"][0]["content"]
    assert title_prompt == enrich.TITLE_PROMPT.format(content=_LONG_CONTENT)
    assert json.loads(patch.calls.last.request.content) == {
        "tags": [3, 5],  # `queue` stripped: the document converged
        "title": "Factuur van Hermes",
        "correspondent": 17,
        "created": "2026-09-01",
    }
    assert result.outcome == "enriched"
    assert result.model_passes == 2
    assert result.matched_tags == [5]
    assert result.suggested_tags == ["Tax return"]
    assert result.correspondent == "Hermes"
    assert result.created == "2026-09-01"


@respx.mock
def test_enrich_never_creates_a_tag(monkeypatch):
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
    assert json.loads(patch.calls.last.request.content)["tags"] == [3]
    assert result.suggested_tags == ["brand new tag"]


@respx.mock
def test_enrich_fetches_the_vocabulary_when_not_handed_one(monkeypatch):
    """The trigger path enriches one document per run, so it fetches its own."""
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(tags=("shipping",))
    tags = _mock_tag_list(pages=({"Shipping": 6},))
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    _extract(vocab=None)

    assert tags.call_count == 1
    assert json.loads(patch.calls.last.request.content)["tags"] == [3, 6]


@respx.mock
def test_enrich_leaves_a_created_date_paperless_found_alone(monkeypatch):
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
def test_enrich_never_overwrites_an_existing_correspondent(monkeypatch):
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
def test_enrich_creates_a_new_correspondent_unowned(monkeypatch):
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
def test_enrich_empty_title_leaves_the_title_alone_and_writes_the_rest(monkeypatch):
    """An empty answer is the model saying "nothing here", not a failure: the
    #1563 gate lost 8/10 documents to retrying one. The other fields still land."""
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3, QUEUE_ID))
    _mock_extraction(title="  ", tags=("invoice",))
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    result = _extract()

    assert json.loads(patch.calls.last.request.content) == {
        "tags": [3, 5], "correspondent": 17, "created": "2026-09-01",
    }
    assert result.outcome == "enriched"
    assert result.title is None


@respx.mock
def test_enrich_empty_facts_leave_their_fields_alone(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3, QUEUE_ID))
    _mock_extraction(correspondent=" ", tags=(), created="")
    search = _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    result = _extract()

    assert not search.called
    assert json.loads(patch.calls.last.request.content) == {
        "tags": [3], "title": "Factuur van Hermes",
    }
    assert (result.correspondent, result.created, result.matched_tags) == (None, None, [])


@respx.mock
def test_enrich_all_fields_empty_still_converges(monkeypatch):
    """Nothing to write but `queue` goes, so the sweep stops re-reading it."""
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3, QUEUE_ID))
    _mock_extraction(title="", correspondent="", tags=(), created="")
    patch = _mock_patch()

    result = _extract()

    assert json.loads(patch.calls.last.request.content) == {"tags": [3]}
    assert result.outcome == "enriched"


@respx.mock
@pytest.mark.parametrize("tags", [None, "invoice", {"name": "invoice"}, [None, 7, ""]])
def test_enrich_malformed_tags_leave_the_tags_alone(monkeypatch, tags):
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3, QUEUE_ID))
    _mock_extraction(tags=tags)
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    result = _extract()

    assert json.loads(patch.calls.last.request.content)["tags"] == [3]
    assert result.matched_tags == []


@respx.mock
def test_enrich_invalid_created_leaves_created_alone(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(created="sometime in September")
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    result = _extract()

    body = json.loads(patch.calls.last.request.content)
    assert "created" not in body
    assert body["title"] == "Factuur van Hermes"
    assert result.created_proposed == "sometime in September"


@respx.mock
def test_enrich_raises_on_an_ollama_failure(monkeypatch):
    """Nothing to fall back to — the task retry is the fallback."""
    _extract_env(monkeypatch)
    _mock_extract_document()
    respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(500))
    patch = _mock_patch()

    with pytest.raises(httpx.HTTPStatusError):
        _extract()
    assert not patch.called


@respx.mock
@pytest.mark.parametrize("failing", ["title_status", "facts_status"])
def test_enrich_raises_when_either_query_fails(monkeypatch, failing):
    """Unlike the default path's degrading title query: a failed call must not
    read as an empty answer, or a timeout would converge a document untitled."""
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(**{failing: 500})
    patch = _mock_patch()

    with pytest.raises(httpx.HTTPStatusError):
        _extract()
    assert not patch.called


@respx.mock
def test_enrich_raises_on_an_unparseable_answer(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document()
    respx.post(f"{OLLAMA}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "not json"}})
    )
    patch = _mock_patch()

    with pytest.raises(ValueError):
        _extract()
    assert not patch.called


@respx.mock
def test_enrich_strips_a_legal_suffix_from_the_correspondent(monkeypatch):
    """The 3B model kept "GmbH" despite the prompt in the #1563 gate run."""
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(correspondent="Hermes Germany GmbH")
    search = _mock_correspondent_search(results=({"id": 17},))
    _mock_patch()

    result = _extract()

    assert search.calls.last.request.url.params["name__iexact"] == "Hermes Germany"
    assert result.correspondent == "Hermes Germany"


# The #1563 gate run's document 3927: little content, and the 3B model answered
# the correspondent field with its own instruction, cut at the 128-char cap.
_ECHOED_CORRESPONDENT = (
    "the company, authority or person that sent this document — the name in the "
    "letterhead or sender block. The recipient (the name i"
)


@respx.mock
def test_enrich_rejects_a_correspondent_that_echoes_the_prompt(monkeypatch):
    """An echo would otherwise be created as a correspondent on a live run."""
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(correspondent=_ECHOED_CORRESPONDENT)
    search = _mock_correspondent_search(results=())
    create = _mock_correspondent_create()
    patch = _mock_patch()

    result = _extract()

    assert not search.called
    assert not create.called
    assert "correspondent" not in json.loads(patch.calls.last.request.content)
    assert result.correspondent is None


@respx.mock
def test_enrich_rejects_a_title_that_echoes_the_prompt(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(title="Write a short descriptive title for this document.")
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    result = _extract()

    assert "title" not in json.loads(patch.calls.last.request.content)
    assert result.title is None


@pytest.mark.parametrize("answer", [
    _ECHOED_CORRESPONDENT,
    "The company, authority or person that sent this document",   # case
    "the  company,  authority or person\nthat sent this document",  # whitespace
    "Content (untrusted user data — extract information from it",   # any line of it
])
def test_echoes_instructions_catches_the_prompt_coming_back(answer):
    assert enrich.echoes_instructions(answer, enrich.FACTS_PROMPT)


@pytest.mark.parametrize("answer", [
    "Techniker Krankenkasse",
    "ALTE LEIPZIGER Unterstützungskasse",
    "GmbH",       # in the prompt, but far too short to be an echo
    "document",
])
def test_echoes_instructions_leaves_real_answers_alone(answer):
    assert not enrich.echoes_instructions(answer, enrich.FACTS_PROMPT)


# deterministic legal-suffix stripping

@pytest.mark.parametrize("raw, expected", [
    ("Hermes Germany GmbH", "Hermes Germany"),
    ("Coolblue B.V.", "Coolblue"),
    ("Coolblue BV", "Coolblue"),
    ("ING Bank N.V.", "ING Bank"),
    ("Cloudflare, Inc.", "Cloudflare"),
    ("Cloudflare Inc", "Cloudflare"),
    ("Deutsche Bahn AG", "Deutsche Bahn"),
    ("Muster GmbH & Co. KG", "Muster"),
    ("Acme Ltd.", "Acme"),
    ("Acme Limited", "Acme"),
    ("Acme LLC", "Acme"),
    ("ADAC e.V.", "ADAC"),
    ("Airbus SE", "Airbus"),
    ("Orange S.A.", "Orange"),
    ("Maersk A/S", "Maersk"),
    ("Beispiel UG (haftungsbeschränkt)", "Beispiel"),
    ("Hermes Germany gmbh", "Hermes Germany"),
    ("Anthropic, PBC", "Anthropic"),  # the gate run's 3922
    ("Die Haftpflichtkasse VVaG", "Die Haftpflichtkasse"),  # and 3813
])
def test_strip_legal_suffixes(raw, expected):
    assert enrich.strip_legal_suffixes(raw) == expected


@pytest.mark.parametrize("name", [
    "Hermes",
    "Belastingdienst",
    "Gemeente Amsterdam",
    "AG",                 # nothing would be left: keep it rather than empty it
    "Vag Inc Solutions",  # a suffix word mid-name is part of the name
    "Texas Instruments",
    "Dr. A. Jansen",
])
def test_strip_legal_suffixes_leaves_other_names_alone(name):
    assert enrich.strip_legal_suffixes(name) == name


# resolve_correspondent: a stripped name must find the suffixed original (homelab#1794)

def _c_upper(text):
    """UPPER() under the paperless database's C collation: ASCII letters only
    (homelab#1802: upper('möbel') is 'MöBEL')."""
    return "".join(ch.upper() if ch.isascii() else ch for ch in text)


def _mock_correspondents(existing):
    """Serve /api/correspondents/ over `existing` ({id: name}), filtering the way
    paperless does: name__iexact / name__istartswith, compared through the C
    collation's ASCII-only UPPER(). Any other filter is refused rather than
    ignored, which is what django-filter would do with it: return everything."""
    def respond(request):
        params = dict(request.url.params)
        params.pop("page_size", None)
        [(lookup, value)] = params.items()
        wanted = _c_upper(value)
        if lookup == "name__iexact":
            hits = [i for i, n in existing.items() if _c_upper(n) == wanted]
        elif lookup == "name__istartswith":
            hits = [i for i, n in existing.items() if _c_upper(n).startswith(wanted)]
        else:
            raise AssertionError(f"paperless has no {lookup} filter")
        return httpx.Response(
            200, json={"results": [{"id": i, "name": existing[i]} for i in hits]}
        )

    return respx.get(f"{PAPERLESS}/api/correspondents/").mock(side_effect=respond)


def _resolve(name):
    with _client() as client:
        return enrich.resolve_correspondent(client, PAPERLESS, name)


@respx.mock
def test_a_stripped_name_resolves_to_the_suffixed_original():
    _mock_correspondents({5: "Scalable Capital Bank GmbH"})
    create = _mock_correspondent_create()

    assert _resolve("Scalable Capital Bank") == 5
    assert not create.called


@respx.mock
def test_a_suffixed_name_resolves_to_a_suffix_free_original():
    _mock_correspondents({5: "Symbox"})
    create = _mock_correspondent_create()

    assert _resolve("symbox GmbH") == 5
    assert not create.called


@respx.mock
def test_an_exact_match_wins_over_a_suffixed_one():
    """A duplicate created before the fix is exact; don't flip between the two."""
    route = _mock_correspondents({5: "Foo GmbH", 9: "Foo"})

    assert _resolve("Foo") == 9
    assert [list(c.request.url.params) for c in route.calls] == [["name__iexact"]]


@respx.mock
def test_the_oldest_of_several_suffixed_spellings_wins():
    _mock_correspondents({12: "Foo AG", 5: "Foo GmbH"})

    assert _resolve("Foo") == 5


@respx.mock
def test_a_longer_name_sharing_the_prefix_is_not_a_match():
    """istartswith only narrows; "Foo" must not land on "Foobar GmbH"."""
    _mock_correspondents({5: "Foobar GmbH", 6: "Foo Bar GmbH"})
    create = _mock_correspondent_create(correspondent_id=17)

    assert _resolve("Foo") == 17
    assert json.loads(create.calls.last.request.content) == {"name": "Foo", "owner": None}


@respx.mock
def test_the_prefix_lookup_asks_for_the_stripped_name():
    route = _mock_correspondents({})
    _mock_correspondent_create()

    _resolve("Hermes Germany GmbH")

    assert route.calls.last.request.url.params["name__istartswith"] == "Hermes Germany"


# The C collation folds ASCII only (homelab#1802)

@pytest.mark.parametrize("name, existing", [
    ("Möbel-Eins", "MÖBEL-EINS"),
    ("MÖBEL-EINS", "Möbel-Eins"),
    ("KB KÜPPER UND KOLLEGEN BERLIN", "KB Küpper und Kollegen Berlin GmbH"),
    ("ÄRZTEKAMMER BERLIN", "Ärztekammer Berlin"),   # the first letter is the one
    ("ärztekammer berlin", "Ärztekammer Berlin"),
    ("Öko-Test", "ÖKO-TEST GmbH"),
])
@respx.mock
def test_a_non_ascii_case_difference_still_matches(name, existing):
    _mock_correspondents({5: existing})
    create = _mock_correspondent_create()

    assert _resolve(name) == 5
    assert not create.called


@respx.mock
def test_a_non_ascii_leading_letter_does_not_match_a_different_name():
    _mock_correspondents({5: "Ärzteblatt", 6: "Ökotest"})
    create = _mock_correspondent_create(correspondent_id=17)

    assert _resolve("Ärztekammer") == 17


@respx.mock
@pytest.mark.parametrize("answer", ["Foo GmbH", "Foo"])
def test_enrich_assigns_the_existing_suffixed_correspondent(monkeypatch, answer):
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(correspondent=answer)
    _mock_correspondents({5: "Foo GmbH"})
    create = _mock_correspondent_create()
    patch = _mock_patch()

    _extract()

    assert not create.called
    assert json.loads(patch.calls.last.request.content)["correspondent"] == 5


@respx.mock
@pytest.mark.parametrize("answer", ["Foo GmbH", "Foo"])
def test_backfill_assigns_the_existing_suffixed_correspondent(monkeypatch, answer):
    _fallback_env(monkeypatch)
    _mock_document(tags=(3,))
    _mock_ollama(answer)
    _mock_correspondents({5: "Foo GmbH"})
    create = _mock_correspondent_create()
    patch = _mock_patch()

    _backfill()

    assert not create.called
    assert json.loads(patch.calls.last.request.content) == {"correspondent": 5}


@respx.mock
def test_backfill_strips_a_legal_suffix_like_enrich(monkeypatch):
    """One rule in both paths, or each keeps creating its own spelling."""
    _fallback_env(monkeypatch)
    _mock_document(tags=(3,))
    _mock_ollama("Hermes Germany GmbH")
    _mock_correspondents({})
    create = _mock_correspondent_create(correspondent_id=31)
    _mock_patch()

    result = _backfill()

    assert json.loads(create.calls.last.request.content)["name"] == "Hermes Germany"
    assert result.correspondent == "Hermes Germany"


@respx.mock
def test_backfill_rejects_a_correspondent_that_echoes_the_prompt(monkeypatch):
    """An echo is no issuer: marked declined, never created."""
    _fallback_env(monkeypatch)
    _mock_document(tags=(3,))
    _mock_ollama("Name the organization or person that issued or sent this document")
    create = _mock_correspondent_create()
    patch = _mock_patch()

    result = _backfill()

    assert not create.called
    assert json.loads(patch.calls.last.request.content) == {"tags": [3, DECLINED_ID]}
    assert result.outcome == "declined"


@respx.mock
def test_enrich_raises_when_ollama_is_unconfigured(monkeypatch):
    monkeypatch.delenv("ENRICH_OLLAMA_URL", raising=False)
    monkeypatch.delenv("ENRICH_OLLAMA_MODEL", raising=False)
    _mock_extract_document()
    patch = _mock_patch()

    with pytest.raises(RuntimeError):
        _extract()
    assert not patch.called


@respx.mock
def test_enrich_sends_capped_content(monkeypatch):
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
def test_enrich_dry_run_writes_nothing(monkeypatch):
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
    assert result.title == "Factuur van Hermes"
    assert result.matched_tags == [5]
    assert result.suggested_tags == ["tax"]
    assert result.correspondent == "Cloudflare"
    assert result.created == "2026-09-01"


@respx.mock
def test_enrich_short_content_skips_the_query(monkeypatch):
    _extract_env(monkeypatch)
    _mock_extract_document(content="too short")
    ollama = _mock_extraction()
    _mock_patch()

    result = _extract()

    assert not ollama.called
    assert result.outcome == "skipped-short-content"
    assert result.model_passes == 0


@respx.mock
def test_the_trigger_path_enriches_a_document_that_never_got_queue(monkeypatch):
    """A UI upload can dodge the workflow; the trigger must not require the tag."""
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3,))
    _mock_extraction(tags=("invoice",))
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    result = _extract()

    assert result.outcome == "enriched"
    assert json.loads(patch.calls.last.request.content)["tags"] == [3, 5]


@respx.mock
def test_an_existing_correspondent_is_reused_rather_than_duplicated(monkeypatch):
    """A replayed trigger after a failed PATCH must find its own earlier create."""
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction(correspondent="Symbox")
    _mock_correspondent_search(results=({"id": 21, "name": "Symbox"},))
    create = _mock_correspondent_create()
    patch = _mock_patch()

    _extract()

    assert not create.called
    assert json.loads(patch.calls.last.request.content)["correspondent"] == 21


# the before-state record for a write (#1562)

@respx.mock
def test_enrich_writes_the_pending_record_before_the_patch(monkeypatch, results_path):
    """Same crash-safety as the default path, plus `created`, which only extract writes."""
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3,), created="2026-09-20")
    _mock_extraction(tags=("invoice",), created="2026-09-01")
    _mock_correspondent_search(results=({"id": 17},))
    seen_at_patch_time = []

    def respond(request):
        seen_at_patch_time.extend(_records(results_path))
        return httpx.Response(200, json={"id": DOC_ID})

    respx.patch(f"{PAPERLESS}/api/documents/{DOC_ID}/").mock(side_effect=respond)

    result = _extract()

    [record] = seen_at_patch_time
    assert record["outcome"] == enrich.PENDING_OUTCOME
    assert record["previous_title"] == "scan_0042"
    assert record["previous_tags"] == [3]
    assert record["previous_correspondent"] is None
    assert record["previous_created"] == "2026-09-20"
    assert record["title"] == "Factuur van Hermes"
    assert record["tags"] == [3, 5]
    assert record["correspondent_id"] == 17
    assert record["created"] == "2026-09-01"
    assert result.outcome == "enriched"
    assert (result.tags, result.correspondent_id) == ([3, 5], 17)
    assert (result.previous_title, result.previous_tags, result.previous_created) == (
        "scan_0042", [3], "2026-09-20",
    )


@respx.mock
def test_enrich_failed_patch_still_leaves_the_pending_record(monkeypatch, results_path):
    _extract_env(monkeypatch)
    _mock_extract_document()
    _mock_extraction()
    _mock_correspondent_search(results=({"id": 17},))
    respx.patch(f"{PAPERLESS}/api/documents/{DOC_ID}/").mock(
        return_value=httpx.Response(500)
    )

    with pytest.raises(httpx.HTTPStatusError):
        _extract()

    assert [r["outcome"] for r in _records(results_path)] == [enrich.PENDING_OUTCOME]


@respx.mock
def test_enrich_dry_run_carries_the_before_state_but_writes_no_record(
    monkeypatch, results_path
):
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3,), correspondent=4)
    _mock_extraction()

    result = _extract(dry_run=True)

    assert _records(results_path) == []
    assert (result.previous_title, result.previous_tags, result.previous_correspondent) == (
        "scan_0042", [3], 4,
    )
    # Nothing written, so no after-state ids.
    assert (result.tags, result.correspondent_id) == (None, None)


@respx.mock
def test_enrich_converges_through_converged_tags(monkeypatch):
    """One place decides what "converged" looks like (#1561), extract path included."""
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3,))
    _mock_extraction(tags=("invoice",))
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()
    monkeypatch.setattr(enrich, "converged_tags", lambda tags, queue_id: sorted(tags) + [777])

    result = _extract()

    assert json.loads(patch.calls.last.request.content)["tags"] == [3, 5, 777]
    assert result.tags == [3, 5, 777]


@respx.mock
def test_enrich_strips_queue_even_when_the_model_names_it(monkeypatch):
    """`queue` is never matched in from the model, and always stripped (#1561)."""
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3, QUEUE_ID))
    _mock_extraction(tags=("invoice", "queue", " QUEUE "))
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()

    result = _extract(vocab={**VOCAB, enrich.QUEUE_TAG: QUEUE_ID})

    assert json.loads(patch.calls.last.request.content)["tags"] == [3, 5]
    assert result.tags == [3, 5]
    assert result.matched_tags == [5]
    assert result.suggested_tags == []


@respx.mock
def test_enrich_strips_queue_from_a_matched_tag_that_slipped_through(monkeypatch):
    """Belt and braces: converged_tags runs last, after the union with the matches."""
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3,))
    _mock_extraction(tags=("invoice",))
    _mock_correspondent_search(results=({"id": 17},))
    patch = _mock_patch()
    monkeypatch.setattr(enrich, "match_tags", lambda names, vocab: ([5, QUEUE_ID], []))

    result = _extract()

    assert json.loads(patch.calls.last.request.content)["tags"] == [3, 5]
    assert result.tags == [3, 5]


# sample: dry-run re-enrichment of already-enriched documents (#1563)

@respx.mock
def test_sample_reenriches_a_converged_curated_document_on_a_dry_run(monkeypatch):
    """No `queue` left and a curated title: exactly what a comparison samples."""
    _extract_env(monkeypatch)
    _mock_extract_document(tags=(3,), title="Factuur", original="scan.pdf",
                           correspondent=4)
    ollama = _mock_extraction()
    patch = _mock_patch()

    result = _extract(dry_run=True, sample=True)

    assert ollama.called
    assert not patch.called
    assert result.outcome == "dry-run"
    # The existing assignment is reported past, so answers can be compared.
    assert result.correspondent == "Hermes"


def test_sample_refuses_to_write():
    with _client() as client:
        with pytest.raises(ValueError):
            enrich.enrich_document(client, PAPERLESS, DOC_ID, QUEUE_ID, sample=True)
