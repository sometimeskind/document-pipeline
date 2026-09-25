"""Replay the enrich before-state from the results JSONL (#1562).

Enrichment rewrites titles, tags and correspondents across the whole library, so
a bad batch — a regressed prompt, a wrong model tag — used to be recoverable only
by a Velero restore of the namespace. Every title/correspondent PATCH now leaves
the document's prior state in the JSONL first, and this puts it back.

Dry-run by default, like the rest of the pipeline. Two rules make it safe over a
live library:

- **A document edited since is skipped, not clobbered.** The record carries the
  after-state it wrote; if the document no longer looks like that, someone (or
  something) changed it since, and replaying the before-state would destroy
  their edit.
- **A reverted document is left converged.** It is left without the sweep's
  `queue` tag, and a correspondent reverted to none gets the backfill's decline
  tag — otherwise the next hourly run would simply redo the write that was just
  undone.

A record can name a tag that has since been deleted — every record written
before #1561 carries the retired `ai-processed` marker's id. Such ids are dropped
from both sides of the comparison and from the write, and reported in the detail,
rather than failing the PATCH or reading as an edit made since.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from document_pipeline import enrich
from document_pipeline.scan import resolve_tag

logger = logging.getLogger(__name__)

# The outcomes whose PATCH changed title or correspondent. `pending` is the
# pre-PATCH record: if the task died before its outcome record, the PATCH may
# or may not have landed, and the "edited since" check tells which.
REVERTIBLE_OUTCOMES = frozenset({"enriched", "backfilled", enrich.PENDING_OUTCOME})


@dataclass
class Reversion:
    """What rollback did (or would do) to one document."""

    document_id: int
    # would-revert | reverted | changed-since | already-reverted | no-before-state
    status: str
    payload: dict = field(default_factory=dict)
    detail: str = ""


def parse_since(value: str) -> datetime:
    """An ISO date or timestamp; a naive one is taken as UTC, like the records."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_latest(
    path: str | None, since: datetime, document_id: int | None = None
) -> dict[int, dict]:
    """The newest revertible record per document at or after `since`.

    Newest by position in the file, not by timestamp: the log is append-only,
    so line order is write order even if the clock stepped. Records from before
    #1562 carry no `recorded_at` and cannot be placed in time, so they never
    match — they carry no before-state to replay anyway.
    """
    target = Path(path or os.environ.get("ENRICH_RESULTS_PATH", enrich.DEFAULT_RESULTS_PATH))
    latest: dict[int, dict] = {}
    with target.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", target)
                continue
            if record.get("outcome") not in REVERTIBLE_OUTCOMES:
                continue
            if document_id is not None and record.get("document_id") != document_id:
                continue
            recorded_at = record.get("recorded_at")
            if not recorded_at or parse_since(recorded_at) < since:
                continue
            latest[int(record["document_id"])] = record
    return latest


def _state(title, tags, correspondent) -> dict:
    return {"title": title, "tags": sorted(int(t) for t in tags), "correspondent": correspondent}


def _day(value) -> str | None:
    """A `created` value as YYYY-MM-DD, whether paperless sent a date or a datetime."""
    return str(value)[:10] if value else None


def _writes_created(record: dict) -> bool:
    """Only extract mode writes `created` (#1563), and only when pick_created allowed it."""
    return record.get("created") is not None and record.get("previous_created") is not None


def _expected_after(record: dict) -> dict:
    """The state the record's write left behind. None fields were not written."""
    def written(key: str, previous_key: str):
        value = record.get(key)
        return record[previous_key] if value is None else value

    state = _state(
        written("title", "previous_title"),
        written("tags", "previous_tags"),
        written("correspondent_id", "previous_correspondent"),
    )
    if _writes_created(record):
        state["created"] = _day(record["created"])
    return state


class _Tags:
    """Resolves tags once, and never creates one on a dry run."""

    def __init__(self, client: httpx.Client, paperless_url: str, write: bool):
        self._client = client
        self._url = paperless_url
        self._write = write
        self._ids: dict[str, int | None] = {}
        self._exists: dict[int, bool] = {}

    def get(self, name: str, *, create: bool = True) -> int | None:
        """The tag's id; created on a write run when `create`, else None if missing."""
        if name not in self._ids:
            if self._write and create:
                self._ids[name] = resolve_tag(self._client, self._url, name)
            else:
                resp = self._client.get(f"{self._url}/api/tags/", params={"name__iexact": name})
                resp.raise_for_status()
                results = resp.json().get("results") or []
                self._ids[name] = int(results[0]["id"]) if results else None
            if self._ids[name] is not None:
                self._exists[self._ids[name]] = True  # found, or just created
        return self._ids[name]

    def existing(self, ids: set[int]) -> set[int]:
        """The subset of `ids` that are still tags in paperless, cached per run."""
        unseen = sorted(i for i in ids if i not in self._exists)
        if unseen:
            resp = self._client.get(
                f"{self._url}/api/tags/",
                params={"id__in": ",".join(str(i) for i in unseen), "page_size": len(unseen)},
            )
            resp.raise_for_status()
            found = {int(r["id"]) for r in resp.json().get("results") or []}
            self._exists.update({i: i in found for i in unseen})
        return {i for i in ids if self._exists[i]}


def _target(record: dict, expected: dict, tags: _Tags) -> dict:
    """The before-state, converged so the sweep and backfill leave it alone."""
    target_tags = list(record["previous_tags"])
    # Never created just to be absent: a missing `queue` is already converged.
    queue_id = tags.get(enrich.QUEUE_TAG, create=False)
    if queue_id is not None:
        target_tags = enrich.converged_tags(target_tags, queue_id)
    if record["previous_correspondent"] is None and expected["correspondent"] is not None:
        # A marked document with no correspondent is exactly what the backfill
        # queries for; without the decline tag it re-assigns the one just undone.
        declined_id = tags.get(enrich.NO_CORRESPONDENT_TAG)
        if declined_id is not None:
            target_tags = sorted(set(target_tags) | {declined_id})
    state = _state(record["previous_title"], target_tags, record["previous_correspondent"])
    if _writes_created(record):
        state["created"] = _day(record["previous_created"])
    return state


def _diff(current: dict, expected: dict) -> str:
    return ", ".join(
        f"{key} {expected[key]!r} -> {current[key]!r}"
        for key in expected
        if current[key] != expected[key]
    )


def revert_document(
    client: httpx.Client, paperless_url: str, record: dict, tags: _Tags, *, write: bool
) -> Reversion:
    document_id = int(record["document_id"])
    if record.get("previous_tags") is None or "previous_title" not in record:
        return Reversion(document_id, "no-before-state", detail="record predates #1562")

    expected = _expected_after(record)
    target = _target(record, expected, tags)
    named = set(expected["tags"]) | set(target["tags"])
    deleted = sorted(named - tags.existing(named))
    if deleted:
        expected["tags"] = [t for t in expected["tags"] if t not in deleted]
        target["tags"] = [t for t in target["tags"] if t not in deleted]
    note = f"dropped deleted tag id(s) {deleted}" if deleted else ""

    def detail(diff: str = "") -> str:
        return "; ".join(part for part in (diff, note) if part)

    document = enrich.fetch_document(client, paperless_url, document_id)
    current = _state(
        document.get("title"), document.get("tags") or [], document.get("correspondent")
    )
    if "created" in expected:
        current["created"] = _day(document.get("created"))

    if current == target:
        return Reversion(document_id, "already-reverted", target, detail=detail())
    if current != expected:
        return Reversion(
            document_id, "changed-since", target, detail=detail(_diff(current, expected))
        )
    if not write:
        return Reversion(document_id, "would-revert", target, detail=detail(_diff(target, current)))

    # Sent whole, correspondent included: None here means "clear it", unlike
    # patch_document where it means "leave it alone". `created` is in it only
    # when the record wrote it (extract mode, #1563).
    resp = client.patch(f"{paperless_url}/api/documents/{document_id}/", json=target)
    resp.raise_for_status()
    enrich.append_result(
        enrich.EnrichResult(
            document_id=document_id,
            outcome="rolled-back",
            title=target["title"],
            tags=target["tags"],
            correspondent_id=target["correspondent"],
            previous_title=current["title"],
            previous_tags=current["tags"],
            previous_correspondent=current["correspondent"],
            created=target.get("created"),
            previous_created=current.get("created"),
        )
    )
    logger.info("Document %s rolled back: %s", document_id, detail(_diff(target, current)))
    return Reversion(document_id, "reverted", target, detail=detail(_diff(target, current)))


def run(
    client: httpx.Client, paperless_url: str, records: dict[int, dict], *, write: bool = False
) -> list[Reversion]:
    """Revert each document to its record's before-state, in document order."""
    tags = _Tags(client, paperless_url, write)
    return [
        revert_document(client, paperless_url, records[document_id], tags, write=write)
        for document_id in sorted(records)
    ]
