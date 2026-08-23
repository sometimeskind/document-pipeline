# document-pipeline

Dockerized document pipeline: PDF attachments from IMAP mail and scanned documents from WebDAV, into Paperless.

```
Proton ↔ Bridge ↔ mbsync ↔ /maildir ↔ Dovecot ↔ Thunderbird (or any IMAP client)
                              ↓
                           notmuch
                              ↓
                       extract PDFs → Paperless
                              ↓
                         push metrics → Pushgateway
```

`mbsync` is **bidirectional**. New mail flows down from Proton; local changes (deletes, moves, flag/Seen changes made by a mail client through Dovecot) flow back up. A single long-running container runs a Prefect flow on event-driven triggers from the cluster (with a cron-backstop) and exposes a small HTTP API for health probes and on-demand triggers — see [Trigger architecture](#trigger-architecture).

| Flow | Backstop schedule | Tasks |
|---|---|---|
| `mail` | `*/5 * * * *` (`FETCH_CRON`) | `sync_mail` → `index_mail` → `extract_pdfs` → `push_metrics` |
| `scan` | `0 * * * *` (`SCAN_CRON`) | `process_scans` → `push_scan_metrics` |
| `enrich` | none — trigger-driven | `enrich_document` → `push_enrich_metrics` |
| `enrich-sweep` | `0 * * * *` (`ENRICH_SWEEP_CRON`) | `find_unenriched` → `enrich_document` per document |

## HTTP API (port `8080`)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | open | k8s liveness/readiness probe |
| `POST` | `/sync/trigger` | bearer | Submit a `mail` flow run and return 202 |
| `POST` | `/trigger-flow` | bearer | Submit a `mail` flow run, 202 if one is already in flight |
| `POST` | `/trigger-scan` | bearer | Submit a `scan` flow run, 202 if one is already in flight |
| `POST` | `/trigger-enrich` | bearer | Submit an `enrich` flow run for `{"document_id": N}`. No coalescing — 400 on a bad id |

`/sync/trigger` and the cron schedule both submit runs of the same Prefect deployment. Overlapping runs are coalesced by a Prefect named concurrency limit (`mail-pipeline`, `occupy=1`, `timeout_seconds=0`) — a second run that finds the slot taken exits immediately rather than queuing.

## Trigger architecture

The `mail` flow is intended to run on **event-driven triggers** from two cluster-side sidecars. The cron schedule is a backstop, not the primary mechanism.

| Trigger | Direction | How | Latency |
|---|---|---|---|
| `goimapnotify` sidecar | Inbound (Proton → `/maildir`) | Watches Bridge over IMAP IDLE; on new mail, calls `POST /sync/trigger` | Near real-time |
| `inotifywait` sidecar | Outbound (`/maildir` → Proton) | Watches `/maildir` for local writes (Dovecot, mail clients); on change, calls `POST /sync/trigger` | Near real-time |
| Cron (`FETCH_CRON`) | Both | Submits a flow run regardless of activity | Up to `FETCH_CRON` minutes |

The sidecars themselves live in the cluster (`sometimeskind/homelab`), not this image. The integration contract is just the HTTP API:

```sh
curl -X POST http://localhost:8080/sync/trigger \
     -H "Authorization: Bearer $API_BEARER_TOKEN"
```

Overlapping triggers are coalesced by a Prefect named concurrency limit (`mail-pipeline`, `occupy=1`, `timeout_seconds=0`); a second run that finds the slot taken exits immediately and the next trigger or cron tick picks up any missed work. Fire `/sync/trigger` as often as you like.

`FETCH_CRON` defaults to `*/5 * * * *` so that a sidecar restart, crash, or network blip is caught within 5 minutes. With both sidecars reliable, raising this (e.g. `0 * * * *`) is safe.

No shared volumes, lock files, or other in-pod coordination are required.

## Bidirectional architecture (Dovecot + IMAP client)

`/maildir` is meant to be shared with a Dovecot sidecar so a mail client (e.g. Thunderbird) can read and write the same store. The expected layout:

```
Proton ↔ Bridge ↔ [this container: mbsync + notmuch + extract]
                          ↕
                       /maildir   (shared PVC)
                          ↕
                  [sidecar: Dovecot IMAP]
                          ↕
                    Thunderbird / mutt / …
```

### Concurrency

`mbsync` and Dovecot both write `/maildir`. Each handles its own atomic-rename and Maildir-level locking; they are designed to coexist. No flock or coordination from this codebase is required.

### Flag synchronisation

For Thunderbird's read/unread/flagged state to round-trip back to Proton, the cluster's `notmuch-config` should set `maildir.synchronize_flags = true`. The chain becomes:

```
Thunderbird marks read
  → Dovecot writes the `S` flag into the Maildir filename
  → next `notmuch new` reflects the flag in notmuch's DB
  → next `mbsync` syncs the flag to Bridge → Proton
```

Inbound flag changes (e.g. read on the Proton web UI) flow the same way in reverse.

### `+paperless` does not propagate to Proton

The `+paperless` tag is written only to notmuch's database — it is a local marker so already-processed messages are not re-submitted. It is **not** visible in Thunderbird or as a Proton label. Surfacing it requires a custom Maildir keyword mapped to a synchronisable flag in both `notmuch-config` and `mbsyncrc`; that mapping lives in the cluster, not in this image.

### Outbound trigger

Local changes (Thunderbird → Dovecot → `/maildir`) propagate to Proton when the cluster's `inotifywait` sidecar sees the write and calls `POST /sync/trigger`. See [Trigger architecture](#trigger-architecture).

## Scan ingestion (`scan` flow)

A network scanner writes straight to a WebDAV share; the `scan` flow drains that
share into Paperless and removes each file once Paperless confirms it landed.

```
Brother MFC ──WebDAV──> Davis ──inotify sidecar──> POST /trigger-scan
                          ↑                              ↓
                          └────── PROPFIND / GET / DELETE ┘   scan flow → Paperless
```

The safety property worth stating explicitly: `POST /api/documents/post_document/`
returns 2xx when the document is **queued**, not when it is ingested. Deleting on
that 2xx would destroy the only copy of a scan whose consume task later fails. So
each file is followed to a terminal Paperless task state and only deleted on
`success` — or on a `failure` that reports a duplicate, which is what a re-POST
after a poll timeout produces under `PAPERLESS_CONSUMER_DELETE_DUPLICATES`. Every
other outcome leaves the file in place for the next sweep, and the
`scan_pipeline_oldest_pending_file_age_seconds` metric is what alerts on files that
never clear.

WebDAV access goes through [`webdav4`](https://github.com/skshetry/webdav4), which
speaks RFC 4918 over `httpx` — already a dependency, so no second HTTP stack —
and whose own suite runs against wsgidav rather than the server we deploy. That
is the property this pipeline needs: moving to a different WebDAV server should
be an env repoint, not a code change, and a client tested against a *different*
server than ours is better evidence of that than fixtures we wrote ourselves.
`document_pipeline/webdav.py` is only a thin adapter for the three things the library
leaves to the caller: entry shape, already-gone resources treated as success, and
a not-yet-created scan directory treated as an empty one. The tests still run
every case against two differently-shaped multistatus responses.

Only `.pdf/.jpg/.jpeg/.png/.tif/.tiff` are eligible; dotfiles, other extensions and
subdirectories are left alone and excluded from the pending-file metric, so an
unrelated file in the share can never hold the staleness alert open forever.

Scan ingestion is opt-in on `WEBDAV_URL`: with it unset the `scan` deployment is
not registered and the image behaves exactly as before.

## Document enrichment

Paperless 3.0 ships LLM suggestions but only behind the manual "Suggest" button —
nothing runs during consumption. The `enrich` flow is that missing automation: it
reads the document, asks Paperless for an `ai_suggestions` title and tag matches,
and writes back the title plus the **union** of the document's existing tags, the
matched tags, and an `ai-processed` marker.

It is post-consume by necessity: `ai_suggestions` reads `document.content`, which
does not exist until Paperless has done the OCR. Paperless's consume is an
external async step in this pipeline, and the post-consume hook is that step's
completion callback — it POSTs `/trigger-enrich`, which is why enrichment covers
**every** ingest path, including documents uploaded through the Paperless UI that
never touched this service.

Two details are load-bearing:

- **Unmatched tag names are recorded, never applied.** Applying them would let the
  model grow the tag vocabulary one document at a time. They go to the results
  JSONL instead, which is what a vocabulary is curated from.
- **Ollama work is held in a `concurrency("ollama", occupy=1)` slot** with no
  timeout. Unlike the mail and scan slots, a busy slot must *queue* here: those
  flows drain a source wholesale so a skipped run is covered by the next one, but
  a skipped enrich run silently loses that document.

`enrich-sweep` enriches documents that carry no marker. It covers a dropped
trigger, and run on a cron it is also the backfill over a pre-existing library —
the same code path, repeated, rather than a separate one-off script. Its own
`concurrency("enrich-sweep", occupy=1)` slot stops a long batch from overlapping
the next cron firing and doing the same documents twice.

Two things exist for the backfill specifically:

- **A document whose title is not `Path(original_file_name).stem[:127]` is never
  retitled.** That is precisely what paperless's consumer writes at consume time,
  so inequality is an exact test for "a human or a workflow named this" rather
  than a guess. Such documents still get the marker, so the sweep converges. A
  freshly consumed document always compares equal, so this never fires on the
  trigger path.
- **`dry_run=true` writes nothing** — no PATCH, so no marker, no filename rename
  and no state change. It reports the proposed title and the unmatched names to
  the results JSONL for review. Because it leaves no marker it re-reads the same
  documents every time: it is a sample, not a pass over the library. It is a
  flow-run parameter, not an env var, deliberately — the sweep's steady-state job
  is catching dropped triggers, and a dry-run default would silently disable it.

Harvest the vocabulary the corpus asked for from the results JSONL:

```bash
kubectl exec -n mail deploy/document-pipeline -- python -m document_pipeline vocab
```

Tagging cannot bootstrap itself — `match_tags_by_name` only matches tags that
already exist — so those names have to be created before matching can ever fire.

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `PREFECT_API_URL` | yes | — | Prefect server URL |
| `PAPERLESS_URL` | yes | — | Paperless base URL |
| `PAPERLESS_API_TOKEN` | yes | — | Paperless API token |
| `API_BEARER_TOKEN` | yes | — | Bearer token guarding the trigger endpoints |
| `FETCH_CRON` | no | `*/5 * * * *` | Cron schedule for the `mail` deployment |
| `PUSHGATEWAY_URL` | no | unset → metrics skipped | Pushgateway URL |
| `NOTMUCH_CONFIG` | no | `/config/notmuch-config` | Path to notmuch config |
| `MBSYNC_CONFIG` | no | `/config/mbsyncrc` | Path to mbsync config |
| `WEBDAV_URL` | no | unset → `scan` flow disabled | WebDAV base URL holding the scans |
| `WEBDAV_USERNAME` | with `WEBDAV_URL` | — | WebDAV Basic-auth user |
| `WEBDAV_PASSWORD` | with `WEBDAV_URL` | — | WebDAV Basic-auth password |
| `WEBDAV_SCAN_PATH` | no | `/` | Path under `WEBDAV_URL` to drain |
| `SCAN_CRON` | no | `0 * * * *` | Sweep schedule for the `scan` deployment |
| `PAPERLESS_ADMIN_TOKEN` | no | falls back to `PAPERLESS_API_TOKEN` | Superuser Paperless token used by `enrich` |
| `ENRICH_SWEEP_CRON` | no | unset → sweep has no schedule | Cron for the `enrich-sweep` deployment |
| `ENRICH_SWEEP_BATCH_SIZE` | no | `20` | Documents per sweep run |
| `ENRICH_SUGGEST_TIMEOUT` | no | `650` | Read timeout for `ai_suggestions`, in seconds |
| `ENRICH_OLLAMA_URL` | no | unset → dedicated title/correspondent queries off | Ollama base URL for the queries enrich runs itself |
| `ENRICH_OLLAMA_MODEL` | no | unset → dedicated title/correspondent queries off | Model for those queries |
| `ENRICH_FALLBACK_TIMEOUT` | no | `300` | Read timeout for the dedicated Ollama queries, in seconds |
| `ENRICH_RESULTS_PATH` | no | `/state/enrich/results.jsonl` | Per-document enrichment result log |
| `PREFECT_LOGGING_EXTRA_LOGGERS` | no | unset → module logs stay out of the Prefect UI | Set to `document_pipeline` to route module logs into flow run logs |

## Volume mounts expected by the image

| Path | Purpose |
|---|---|
| `/maildir/` | Maildir — mbsync writes here, notmuch indexes here |
| `/state/` | Prefect client state (`PREFECT_HOME`) and the enrichment results JSONL |
| `/config/mbsyncrc` | mbsync config |
| `/config/notmuch-config` | notmuch config |
| `/secrets/bridge-imap-password/password` | Bridge IMAP password (referenced from `mbsyncrc`) |

## Local development

```bash
pip install -r requirements.txt -r requirements-dev.txt -e .
pytest tests/
```

Build and run the dev image to exercise tests against the installed `mbsync` and `notmuch`:

```bash
docker build --target dev -t document-pipeline:dev .
docker run --rm document-pipeline:dev
```

## CI

`.github/workflows/ci.yml` runs tests, builds the image, runs health checks against the built image, and on push to `main` pushes `ghcr.io/<owner>/document-pipeline:{latest,<sha>}`.

`Dockerfile.watcher` builds the inotify sidecar that fires `POST /trigger-scan`, published as `ghcr.io/<owner>/document-pipeline-watcher:{latest,<sha>}`. It carries only `inotify-tools` and `curl` — the watch loop is mounted from a ConfigMap in the cluster repo so tuning it needs no image rebuild.

## Bumping dependencies

Dependabot (`.github/dependabot.yml`) handles weekly bumps for:

- Python deps in `requirements.txt` and `requirements-dev.txt`
- The `FROM python:3.13-slim` base image
- GitHub Actions versions
