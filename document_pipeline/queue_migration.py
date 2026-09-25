"""One-off: move the library from the `ai-processed` marker to the `queue` tag (#1561).

Before #1561 every converged document carried `ai-processed` and the sweep
queried on its absence. Now the sweep queries on the PRESENCE of `queue`, so the
documents still waiting — exactly those without `ai-processed` — must be given
`queue` once, or the sweep would never see them again.

Operator order, so no sweep ever runs against a half-migrated library:

1. deploy the pipeline image that queries on `queue`;
2. run this with `--write` (dry-run by default, like the rest of the pipeline);
3. delete the `ai-processed` tag in the paperless UI.

Idempotent: a document that already carries `queue` is not counted again, and
once `ai-processed` is gone there is nothing left to do. Delete this module with
the marker's last mention once the migration has run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from document_pipeline import enrich

LEGACY_MARKER_TAG = "ai-processed"

# bulk_edit takes an explicit id list; chunked so one request stays small.
CHUNK_SIZE = 500


@dataclass
class Migration:
    marker_id: int | None
    queue_id: int | None
    document_ids: list[int] = field(default_factory=list)
    created_queue: bool = False
    written: bool = False


def find_tag(client: httpx.Client, paperless_url: str, name: str) -> int | None:
    resp = client.get(f"{paperless_url}/api/tags/", params={"name__iexact": name})
    resp.raise_for_status()
    results = resp.json().get("results") or []
    return int(results[0]["id"]) if results else None


def unmarked_ids(
    client: httpx.Client, paperless_url: str, marker_id: int, queue_id: int | None
) -> list[int]:
    """Every document without the marker (and not already queued), in id order."""
    excluded = [marker_id] if queue_id is None else [marker_id, queue_id]
    ids: list[int] = []
    page = 1
    while True:
        resp = client.get(
            f"{paperless_url}/api/documents/",
            params={
                "tags__id__none": ",".join(str(t) for t in excluded),
                "fields": "id",
                "ordering": "id",
                "page_size": CHUNK_SIZE,
                "page": page,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        ids.extend(int(r["id"]) for r in body.get("results") or [])
        if not body.get("next"):
            return ids
        page += 1


def run(client: httpx.Client, paperless_url: str, *, write: bool = False) -> Migration:
    marker_id = find_tag(client, paperless_url, LEGACY_MARKER_TAG)
    queue_id = find_tag(client, paperless_url, enrich.QUEUE_TAG)
    migration = Migration(marker_id=marker_id, queue_id=queue_id)
    if marker_id is None:
        return migration  # already migrated: the marker is gone

    migration.document_ids = unmarked_ids(client, paperless_url, marker_id, queue_id)
    if not write or not migration.document_ids:
        return migration

    if queue_id is None:
        queue_id = enrich.resolve_queue_tag(client, paperless_url)
        migration.queue_id = queue_id
        migration.created_queue = True

    for start in range(0, len(migration.document_ids), CHUNK_SIZE):
        chunk = migration.document_ids[start:start + CHUNK_SIZE]
        resp = client.post(
            f"{paperless_url}/api/documents/bulk_edit/",
            json={"documents": chunk, "method": "add_tag", "parameters": {"tag": queue_id}},
        )
        resp.raise_for_status()
    migration.written = True
    return migration
