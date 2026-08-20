"""Enrich a consumed Paperless document with an LLM title and matched tags.

Paperless 3.0 ships LLM suggestions but only behind the manual "Suggest" button
on the document detail page — nothing runs during consumption. This is that
missing automation, ported from the `post-consume.sh` hook it replaces.

Paperless's consume is an external async step in this pipeline: the flows POST a
document and paperless does the OCR. Enrichment reads `document.content`, so it
can only run once that step has finished — which is why it is triggered by the
post-consume hook rather than done inline at submit time.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from document_pipeline.scan import resolve_tag

logger = logging.getLogger(__name__)

# Below this much OCR text, don't ask. Schema-constrained generation means the
# model MUST emit a title, so a blank or failed scan gets a confidently invented
# one rather than an error.
MIN_CONTENT_CHARS = 50

MAX_TITLE_CHARS = 128  # documents.models.Document.title max_length

# What paperless's own consumer sets a new document's title to:
# `Path(filename).stem[:127]` (documents/consumer.py). One character short of
# the column width, and that is the point — it makes "is this title still the
# one the consumer generated?" an exact test rather than a guess about what a
# machine-generated title looks like.
CONSUME_TITLE_CHARS = 127

# Applied to every document this module touches. `find_unenriched` queries on its
# absence, so it is both the sweep's resume marker and the guard that makes a
# replayed trigger cheap.
MARKER_TAG = "ai-processed"

# `ai_suggestions` is one request that runs TWO Ollama queries: a classification
# query, then a localization query because PAPERLESS_AI_LLM_OUTPUT_LANGUAGE is
# set. So this must exceed twice PAPERLESS_AI_LLM_REQUEST_TIMEOUT (300s). The
# point is for paperless's own timeout to fire first and return a clean 503,
# rather than us severing the connection while Ollama is still generating.
# Typical real cost is ~60s; this ceiling only matters when something is wrong.
DEFAULT_SUGGEST_TIMEOUT = 650.0

DEFAULT_RESULTS_PATH = "/state/enrich/results.jsonl"


@dataclass
class EnrichResult:
    """Outcome of enriching one document. Serialized verbatim to the JSONL."""

    document_id: int
    outcome: str
    title: str | None = None
    matched_tags: list[int] = field(default_factory=list)
    # Names the model proposed that matched no existing tag. Deliberately never
    # applied — that would let an LLM grow the vocabulary one document at a time.
    # Recorded because they are the only evidence of which tags are worth
    # creating: matching can never fire for a tag that does not exist yet.
    suggested_tags: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


def open_client(paperless_token: str, suggest_timeout: float | None = None) -> httpx.Client:
    """A Paperless client whose read timeout accommodates the LLM round trip."""
    if suggest_timeout is None:
        suggest_timeout = float(os.environ.get("ENRICH_SUGGEST_TIMEOUT", DEFAULT_SUGGEST_TIMEOUT))
    return httpx.Client(
        headers={"Authorization": f"Token {paperless_token}"},
        timeout=httpx.Timeout(30.0, read=suggest_timeout),
    )


def fetch_document(client: httpx.Client, paperless_url: str, document_id: int) -> dict:
    resp = client.get(f"{paperless_url}/api/documents/{document_id}/")
    resp.raise_for_status()
    return resp.json()


def fetch_suggestions(client: httpx.Client, paperless_url: str, document_id: int) -> dict:
    """Ask Paperless for the LLM suggestion.

    This is the endpoint that actually runs the configured PAPERLESS_AI_*
    backend. The similarly named `/suggestions/` is classifier-only and returns
    no title at all.
    """
    resp = client.get(f"{paperless_url}/api/documents/{document_id}/ai_suggestions/")
    resp.raise_for_status()
    return resp.json()


def patch_document(
    client: httpx.Client,
    paperless_url: str,
    document_id: int,
    tags: list[int],
    title: str | None = None,
) -> None:
    """Write back tags, and the title only when we have one to write.

    Omitting `title` is what keeps the short-content path from rewriting a title
    it never generated — it still gets the marker so the sweep stops picking it.
    """
    payload: dict[str, object] = {"tags": tags}
    if title is not None:
        payload["title"] = title
    resp = client.patch(f"{paperless_url}/api/documents/{document_id}/", json=payload)
    resp.raise_for_status()


def normalize_title(raw: str | None) -> str:
    """Collapse whitespace and truncate to what the column will hold."""
    return " ".join((raw or "").split())[:MAX_TITLE_CHARS]


def merge_tags(existing: list[int], matched: list[int], marker_id: int) -> list[int]:
    """Union of the tags already on the document, the matched ones, and the marker.

    PATCHing `tags` REPLACES the list, so the document's existing tags must be
    merged back in — otherwise enrichment silently strips the `scanner` tag the
    scan flow applies at ingest.
    """
    return sorted({int(t) for t in existing} | {int(t) for t in matched} | {int(marker_id)})


def find_unenriched(
    client: httpx.Client, paperless_url: str, marker_id: int, limit: int
) -> list[int]:
    """Ids of documents that have never been enriched, oldest first."""
    resp = client.get(
        f"{paperless_url}/api/documents/",
        params={
            "tags__id__none": marker_id,
            "page_size": limit,
            "ordering": "id",
            "fields": "id",
        },
    )
    resp.raise_for_status()
    return [int(r["id"]) for r in resp.json().get("results") or []]


def has_curated_title(document: dict) -> bool:
    """True when something other than paperless's consumer named this document.

    The backfill (#1280) reaches documents that predate auto-titling, and some of
    those were titled by hand. There is no "title edited" flag to read, but there
    does not need to be one: the consumer's title is exactly
    `Path(original_file_name).stem[:127]`, so any inequality means a human, a
    workflow or an earlier enrichment run wrote that title.

    A freshly consumed document always compares equal, so this never fires on the
    post-consume trigger path — it only ever bites on the sweep.

    A document with no `original_file_name` cannot be tested at all. Those are
    treated as un-curated and enriched, deliberately: the alternative is marking
    them processed and silently never titling them, and the contract is that this
    must never cost a document its title.
    """
    original = document.get("original_file_name")
    if not original:
        return False
    return document.get("title") != Path(original).stem[:CONSUME_TITLE_CHARS]


def enrich_document(
    client: httpx.Client,
    paperless_url: str,
    document_id: int,
    marker_id: int,
    *,
    dry_run: bool = False,
) -> EnrichResult:
    """Retitle and tag one document. Raises on any Paperless or LLM failure.

    `dry_run` reports what would be written without writing anything: no PATCH,
    so no marker, no filename rename and no state change of any kind. That also
    makes it non-resuming — it re-reports the same documents every time — which
    is exactly what makes a sample reviewable before the live pass.
    """
    started = time.perf_counter()

    document = fetch_document(client, paperless_url, document_id)
    existing_tags = [int(t) for t in document.get("tags") or []]
    if marker_id in existing_tags:
        logger.info("Document %s already enriched, skipping", document_id)
        return EnrichResult(
            document_id=document_id,
            outcome="already-enriched",
            duration_seconds=time.perf_counter() - started,
        )

    if has_curated_title(document):
        # Marked anyway, so the sweep converges instead of re-reading this
        # document every hour for the rest of the library's life.
        logger.info(
            "Document %s has a curated title %r — leaving it alone",
            document_id, document.get("title"),
        )
        if not dry_run:
            patch_document(
                client, paperless_url, document_id, merge_tags(existing_tags, [], marker_id)
            )
        return EnrichResult(
            document_id=document_id,
            outcome="skipped-curated-title",
            duration_seconds=time.perf_counter() - started,
        )

    content_length = len((document.get("content") or "").strip())
    if content_length < MIN_CONTENT_CHARS:
        # Not a failure: an empty scan has nothing to title from. Marked anyway,
        # so the sweep does not keep re-picking it forever.
        logger.info(
            "Document %s has only %d chars of OCR content (min %d) — leaving title unchanged",
            document_id, content_length, MIN_CONTENT_CHARS,
        )
        if not dry_run:
            patch_document(
                client, paperless_url, document_id, merge_tags(existing_tags, [], marker_id)
            )
        return EnrichResult(
            document_id=document_id,
            outcome="skipped-short-content",
            duration_seconds=time.perf_counter() - started,
        )

    suggestions = fetch_suggestions(client, paperless_url, document_id)
    title = normalize_title(suggestions.get("title"))
    # `tags` are ids of tags that ALREADY EXIST — paperless's match_tags_by_name
    # never creates one, so applying them can never grow the vocabulary.
    matched_tags = [int(t) for t in suggestions.get("tags") or []]
    suggested_tags = [" ".join(str(name).split()) for name in suggestions.get("suggested_tags") or []]

    if not title:
        raise ValueError(f"LLM returned an empty title for document {document_id}")

    if dry_run:
        logger.info(
            "Document %s WOULD be retitled -> %r (tags: %s + %s, unmatched: %s)",
            document_id, title, existing_tags or "none", matched_tags or "none",
            suggested_tags or "none",
        )
        return EnrichResult(
            document_id=document_id,
            outcome="dry-run",
            title=title,
            matched_tags=matched_tags,
            suggested_tags=suggested_tags,
            duration_seconds=time.perf_counter() - started,
        )

    tags = merge_tags(existing_tags, matched_tags, marker_id)
    patch_document(client, paperless_url, document_id, tags, title=title)

    logger.info(
        "Document %s retitled -> %r (tags: %s + %s)",
        document_id, title, existing_tags or "none", matched_tags or "none",
    )
    return EnrichResult(
        document_id=document_id,
        outcome="enriched",
        title=title,
        matched_tags=matched_tags,
        suggested_tags=suggested_tags,
        duration_seconds=time.perf_counter() - started,
    )


def resolve_marker_tag(client: httpx.Client, paperless_url: str) -> int:
    return resolve_tag(client, paperless_url, MARKER_TAG)


def append_result(result: EnrichResult, path: str | None = None) -> None:
    """Append one JSONL record to the durable results log.

    stdout cannot carry this: the log goes to the paperless webserver pod and
    does not survive a restart, and the #1292 harvest needs a `doc_id -> [names]`
    mapping that outlives both.
    """
    target = Path(path or os.environ.get("ENRICH_RESULTS_PATH", DEFAULT_RESULTS_PATH))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    except OSError as exc:
        # The record is an artifact, not the job. Losing a line must not cost
        # the document its title.
        logger.warning("Could not append enrich result for %s: %s", result.document_id, exc)


def rank_suggested_tags(path: str | None = None) -> tuple[int, list[tuple[str, int]]]:
    """Frequency-rank the proposed tag names that matched nothing, from the JSONL.

    Tagging cannot bootstrap itself. `match_tags_by_name` only ever matches tags
    that ALREADY EXIST and has no creation path, so until a name is in the
    vocabulary no document can be given it — and the #1280 backfill would spend
    hours proposing names into the void. These are the names the corpus itself
    asked for, which beats a vocabulary invented from memory.

    Counted once per document, so one suggestion repeating a name cannot inflate
    its own rank. Names are grouped case-insensitively because paperless matches
    that way (`paperless_ai/matching.py` case-folds before comparing); the
    reported spelling is the most common one seen.

    Returns (documents considered, [(name, document count)]) ranked descending.
    """
    target = Path(path or os.environ.get("ENRICH_RESULTS_PATH", DEFAULT_RESULTS_PATH))
    documents: set[int] = set()
    counts: collections.Counter[str] = collections.Counter()
    spellings: dict[str, collections.Counter[str]] = {}

    with target.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A torn final line from a killed pod must not cost the whole
                # harvest — every earlier line is still good.
                logger.warning("Skipping malformed line in %s", target)
                continue
            documents.add(record.get("document_id"))
            names = [n.strip() for n in record.get("suggested_tags") or [] if n.strip()]
            for key in {n.lower() for n in names}:
                counts[key] += 1
            for name in names:
                spellings.setdefault(name.lower(), collections.Counter())[name] += 1

    ranked = [
        (spellings[name].most_common(1)[0][0], count)
        for name, count in counts.most_common()
    ]
    return len(documents), ranked
