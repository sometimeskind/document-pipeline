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


def enrich_document(
    client: httpx.Client, paperless_url: str, document_id: int, marker_id: int
) -> EnrichResult:
    """Retitle and tag one document. Raises on any Paperless or LLM failure."""
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

    content_length = len((document.get("content") or "").strip())
    if content_length < MIN_CONTENT_CHARS:
        # Not a failure: an empty scan has nothing to title from. Marked anyway,
        # so the sweep does not keep re-picking it forever.
        logger.info(
            "Document %s has only %d chars of OCR content (min %d) — leaving title unchanged",
            document_id, content_length, MIN_CONTENT_CHARS,
        )
        patch_document(client, paperless_url, document_id, merge_tags(existing_tags, [], marker_id))
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
