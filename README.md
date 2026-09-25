# document-pipeline

Dockerized document pipeline: PDF attachments from IMAP mail and scanned documents from WebDAV, into Paperless.

```
Proton ↔ Bridge ↔ mbsync ↔ /maildir ↔ Dovecot ↔ Thunderbird (or any IMAP client)
                              ↓
                           notmuch
                              ↓
                extract PDFs → WebDAV scan queue (mail/)
                              ↓
                   scan flow → Paperless (polled to a terminal state)
                              ↓
                         push metrics → Pushgateway
```

The mail flow never talks to Paperless itself. Each PDF attachment is PUT into
the `mail/` directory of the same WebDAV share the scanner writes to, named
`<imap-uid>-<sanitised-filename>.pdf`, and the message is flagged `$Processed`
only once every PUT for it succeeded — a failed PUT leaves it unflagged, the
next run retries, and the UID prefix makes that retry overwrite rather than
duplicate. The flow then triggers a `scan` run in-process (coalesced like
`/trigger-scan`), and the scan flow's poll-then-delete contract below is what
follows the file into Paperless. That is the point (homelab#1590): a consume
that fails is left in `mail/` and alerted on, instead of being a 2xx the mail
flow took for done.

A run that ends `Failed` still reports. `push_failure_metrics` POSTs
`document_pipeline_last_failure_timestamp`, and a
`document_pipeline_prefect_failures_24h` that counts the still-Running failing
run the Prefect query cannot yet see, into the same `mail-pipeline` group —
POST, not PUT, so `document_pipeline_last_success_timestamp` stays frozen at
the last good run instead of being wiped. A stale success next to a fresh
failure is what distinguishes "runs are failing" from "no new mail"; before
#60 a failing run published nothing at all, so the two read identically and
#58 ran 43 hours unnoticed. The success path still replaces the whole group,
which is what clears the failure gauges once a run recovers.

`mbsync` is **bidirectional**. New mail flows down from Proton; local changes (deletes, moves, flag/Seen changes made by a mail client through Dovecot) flow back up. A single long-running container runs a Prefect flow on event-driven triggers from the cluster (with a cron-backstop) and exposes a small HTTP API for health probes and on-demand triggers — see [Trigger architecture](#trigger-architecture).

| Flow | Backstop schedule | Tasks |
|---|---|---|
| `mail` | `*/5 * * * *` (`FETCH_CRON`) | `process_mail` (queue PDFs into `mail/`, flag `$Processed`, trigger `scan`) → `push_metrics`, or `push_failure_metrics` when the run fails |
| `scan` | `0 * * * *` (`SCAN_CRON`) | `process_scans` → `push_scan_metrics` |
| `enrich` | none — trigger-driven | `enrich_document` → `push_enrich_metrics` |
| `enrich-sweep` | `0 * * * *` (`ENRICH_SWEEP_CRON`) | `find_unenriched` → `enrich_document` per document |
| `correspondent-backfill` | none unless `CORRESPONDENT_BACKFILL_CRON` | `find_without_correspondent` → `backfill_correspondent` per document |
| `paperless-health` | `*/5 * * * *` (`PAPERLESS_HEALTH_CRON`) | `probe_paperless_health` → `push_paperless_health_metrics` |

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

A network scanner writes straight to a WebDAV share, and the `mail` flow queues
PDF attachments into `mail/` beneath it; the `scan` flow drains both into
Paperless and removes each file once Paperless confirms it landed.

```
Brother MFC ──WebDAV──> <scan path>/        ──trigger──> POST /trigger-scan
mail flow   ──PUT────>  <scan path>/mail/   ──trigger──> (in-process)
                          ↑                              ↓
                          └────── PROPFIND / GET / DELETE ┘   scan flow → Paperless
```

Each source is tagged by where it came from — `scanner` for the root, `mail`
for `mail/` — and the file gauges (`scan_pipeline_files_{ingested,failed,pending}`,
`scan_pipeline_oldest_pending_file_age_seconds`) carry a matching `source`
label, one series per source on every push, so a drained source reads 0 rather
than going stale. `scan_pipeline_last_success_timestamp`,
`scan_pipeline_run_duration_seconds` and `scan_pipeline_prefect_failures_24h`
stay per-run and unlabelled.

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
subdirectories other than `mail/` are left alone and excluded from the
pending-file metric, so an unrelated file in the share can never hold the
staleness alert open forever.

## Document enrichment

Paperless 3.0 ships LLM suggestions but only behind the manual "Suggest" button —
nothing runs during consumption. The `enrich` flow is that missing automation: it
reads the document, asks Paperless for an `ai_suggestions` title and tag matches,
and writes back the title plus the **union** of the document's existing tags and
the matched tags — minus the `queue` tag.

**`queue` marks the documents still waiting** (homelab#1561). A paperless Workflow
(trigger *Document Added*, action *assign tag* `queue`) puts it on every new
document; every terminal outcome here (enriched, skipped-curated-title,
skipped-short-content) removes it; a failed run leaves it for the next sweep. The
workflow is a paperless DB object, so its shape is recorded in the homelab repo
(`docs/scan-ingest-runbook.md`) rather than here. `enrich.converged_tags` is the
one place that decision is applied to a tag list. The trigger path never
*requires* the tag — a document that dodged the workflow is enriched anyway — and a
replayed trigger stays cheap because an enriched title no longer equals the
filename stem (see the curated-title rule below), so it costs no LLM call and no
write. If `queue` does not exist yet the pipeline creates it with matching *None*
and no owner.

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

`enrich-sweep` enriches documents that still carry `queue`. It covers a dropped
trigger, and run on a cron it is also the backfill over a pre-existing library —
the same code path, repeated, rather than a separate one-off script. Its own
`concurrency("enrich-sweep", occupy=1)` slot stops a long batch from overlapping
the next cron firing and doing the same documents twice.

Two things exist for the backfill specifically:

- **A document whose title is not `Path(original_file_name).stem[:127]` is never
  retitled.** That is precisely what paperless's consumer writes at consume time,
  so inequality is an exact test for "a human or a workflow named this" rather
  than a guess. Such documents still lose `queue`, so the sweep converges. A
  freshly consumed document always compares equal, so this never fires on the
  first trigger for a document.
- **`dry_run=true` writes nothing** — no PATCH, so `queue` stays, no filename
  rename and no state change. It reports the proposed title and the unmatched names
  to the results JSONL for review. Because `queue` stays it re-reads the same
  documents every time: it is a sample, not a pass over the library. It is a
  flow-run parameter, not an env var, deliberately — the sweep's steady-state job
  is catching dropped triggers, and a dry-run default would silently disable it.

### Single extraction query (`ENRICH_MODE=extract`, experiment — homelab#1563)

The default path (`ENRICH_MODE=suggest`) costs four model passes per document:
two inside `ai_suggestions` (classification, then localization), the dedicated
title query and the correspondent query. `extract` asks Ollama **once**, with
one schema requiring `{title, correspondent, tags: [string], created}`, and
derives the rest in code:

- **title** and **correspondent** use the dedicated prompts' wording verbatim
  (document's own language, issuer never recipient, no legal suffixes). The
  correspondent is applied only when the document has none, created unowned —
  same as the default path.
- **tags** are matched against `/api/tags/`, fetched once per run,
  case- and whitespace-insensitively against **existing** names only. Unmatched
  names go to `suggested_tags` in the JSONL and are never applied or created —
  the same rule and the same `vocab` harvest as `ai_suggestions`. The
  pipeline's own `queue` and `no-correspondent` are never matched, and the
  written tag list goes through the same convergence as the default path, so
  `queue` is always stripped.
- **created** is written only when the model's answer is a real `YYYY-MM-DD`
  date, not in the future, and paperless's own `created` still equals the date
  the document was `added` — i.e. its date regex found nothing. A date
  paperless parsed, or one set by hand, is never overwritten. The model's raw
  answer is recorded as `created_proposed` either way.

Every record carries `mode` and `model_passes` beside `duration_seconds`, so the
two paths are comparable from the JSONL alone.

**The gate.** `extract` is not the default until a dry-run comparison on the
same ~20-document sample shows it matches or beats `suggest` on title quality
(language, length, boilerplate), correspondent agreement and tag hit rate.
`enrich-sweep` takes `mode` and `document_ids` for this: `document_ids`
replaces the `queue` query with a fixed list and re-enriches those documents
whether or not they still carry `queue` and even past a curated title, and
reports the model's correspondent past an existing assignment. It is refused unless `dry_run=true`,
so nothing is written. Both parameters leave `ENRICH_MODE` and the hourly sweep
alone:

```bash
IDS='[101,102,103]'  # the ~20-document sample
kubectl exec -n mail deploy/document-pipeline -- \
  prefect deployment run enrich-sweep/enrich-sweep \
  -p dry_run=true -p mode='"suggest"' -p document_ids="$IDS"
# wait for that run to finish (Prefect UI), then:
kubectl exec -n mail deploy/document-pipeline -- \
  prefect deployment run enrich-sweep/enrich-sweep \
  -p dry_run=true -p mode='"extract"' -p document_ids="$IDS"
# then, side by side per document plus the totals:
kubectl exec -n mail deploy/document-pipeline -- python -m document_pipeline compare
```

`-p` values are parsed as JSON, hence the inner quotes on the mode. `compare`
pairs each document's latest `dry-run` record per mode, so re-running the
sample supersedes the earlier one. A sample run takes the `ollama` slot like
any sweep and is skipped if the hourly sweep holds the `enrich-sweep` slot.

**Pending the gate:** `ai_suggestions`, `ENRICH_SUGGEST_TIMEOUT` and the
`PAPERLESS_AI_*` env in homelab's `kubernetes/paperless/paperless.yaml` all
stay until `extract` passes it and becomes the default — only then do they go
(the Suggest button stops working; acceptable). If a 3B model with a
four-field schema degrades titles, the fallback is two queries (facts + title),
not four.

### Correspondent backfill

`correspondent-backfill` covers the complementary set: documents that no longer
carry `queue` but have no correspondent — everything enriched
before the dedicated correspondent query existed, plus every document that query
has since declined (~15–25% of the sweep's output: forms and certificates with no
obvious issuer). It skips `ai_suggestions` entirely and runs only the
correspondent query, then a PATCH that carries **only** the correspondent, so
titles (some curated) and tags are byte-identical before and after. Created
correspondents are unowned, same as in enrich.

Two things differ from the sweep, and both exist because here the correspondent
*is* the job rather than a bonus on top of the title:

- **A decline is terminal.** When the model answers with an empty string, or the
  document is under the OCR floor and there is nothing to ask about, the
  document is tagged `no-correspondent`. Without that it would match
  `correspondent__isnull` again every run, forever. A tag rather than a record in
  the results JSONL because it is visible in the paperless UI (assigning one by
  hand there drops the document out of the query on its own) and survives
  losing the state PVC.
- **A query failure raises instead of declining.** The sweep folds an Ollama
  error into "no correspondent" because the title outranks it; here that would
  tag a document `no-correspondent` on a transient timeout and lose it for good.
  The task retries, the document stays unmarked, and a run that lost its whole
  batch fails rather than finishing `Completed`.

Each PATCH renames the file to the `<created>_<correspondent>_<title>` format,
which is the point — and also why the cron is offset from the sweep's and
batched to the same memory budget: it is a slow rolling rename over the
library, paced by the `ollama` slot and the model's keep-alive.

Harvest the vocabulary the corpus asked for from the results JSONL:

```bash
kubectl exec -n mail deploy/document-pipeline -- python -m document_pipeline vocab
```

Tagging cannot bootstrap itself — `match_tags_by_name` only matches tags that
already exist — so those names have to be created before matching can ever fire.

### Rollback

Every record carries the document's state **before** the write —
`previous_title`, `previous_tags` and `previous_correspondent` (ids), straight from
the fetch the write started with — plus the after-state it wrote as ids (`tags`,
`correspondent_id`; `null` means "this write did not touch it") and a UTC
`recorded_at`. For the two writes that change content (`enriched`, `backfilled`)
a `pending` copy of the record is appended **before** the PATCH and the real
outcome after it, so a pod killed mid-PATCH still leaves the before-state on disk.

That makes a bad batch — a regressed prompt, a wrong model tag — revertible
without a Velero restore:

```bash
# Dry run: lists what would be reverted, writes nothing
kubectl exec -n mail deploy/document-pipeline -- \
  python -m document_pipeline rollback --since 2026-09-20T00:00:00Z [--document 1234]
# Apply
kubectl exec -n mail deploy/document-pipeline -- \
  python -m document_pipeline rollback --since 2026-09-20T00:00:00Z --write
```

It PATCHes `title`, `tags` and `correspondent` back to the recorded values for
each `enriched`/`backfilled`/`pending` record since `--since` (a naive timestamp is
UTC); the **newest record per document wins**. Extract-mode records (#1563) also
carry `previous_created`; when the write set `created`, rollback restores it too
and counts a `created` changed since as an edit. Three rules keep it safe on a live
library:

- **A document edited since is skipped and reported (`changed-since`)**, with the
  fields that differ. If its title, tags or correspondent no longer match what the
  record wrote, replaying the before-state would destroy that edit. A `pending`
  record whose PATCH never landed is skipped the same way.
- **A reverted document is left converged.** It is left without `queue` (applied
  through `enrich.converged_tags`, so a before-state that carried it is not
  replayed verbatim) and the sweep does not redo it, and a correspondent reverted
  to none also gets `no-correspondent` so the backfill does not reassign it.
- **Deleted tags are dropped, not reintroduced.** Records written before #1561
  name the retired `ai-processed` tag's id in their before- and after-state. Ids
  that no longer exist in paperless are left out of both the comparison and the
  PATCH (which would otherwise 400) and listed in the output as
  `dropped deleted tag id(s) [...]`.
- **A dry run creates nothing**, not even a missing `no-correspondent` tag.

`--write` appends a `rolled-back` record (itself carrying the before-state), and a
second run reports `already-reverted`. Created correspondents are left in place —
they are unowned and harmless — and the file rename undoes itself: paperless
re-renders the filename on the PATCH. Records from before this existed carry no
`recorded_at` or before-state and are never matched.

### Migrating from `ai-processed` (homelab#1561, one-off)

Before #1561 convergence was an `ai-processed` marker on every enriched document
and the sweep queried on its absence. Moving to `queue` needs one pass that gives
`queue` to every document *without* the marker, in this order so no sweep ever runs
against a half-migrated library:

1. **Deploy the image that queries on `queue`, and create the Workflow.** Until
   step 2 the sweep finds nothing, and the backfill may reach an unmigrated
   document early (harmless: it only assigns a correspondent). The Workflow's
   timing does not matter — the trigger path does not need the tag.
2. **Queue the unmarked documents** — dry-run by default, idempotent:

   ```bash
   kubectl exec -n mail deploy/document-pipeline -- python -m document_pipeline migrate-queue
   kubectl exec -n mail deploy/document-pipeline -- python -m document_pipeline migrate-queue --write
   ```

   It `bulk_edit`s `add_tag` onto every document matching
   `tags__id__none=<ai-processed>,<queue>`, creating `queue` (matching *None*,
   unowned) if the operator has not yet.
3. **Delete the `ai-processed` tag** in the paperless UI. Safe for filenames:
   `PAPERLESS_FILENAME_FORMAT` contains no tags, so no rename storm. Rollback copes
   with the deleted id in older records (above).

`queue_migration.py` and the `migrate-queue` subcommand can go once this has run.

## Paperless task-queue health (`paperless-health` flow)

Paperless exposes no metrics, but its `/api/tasks/` is the source of truth for
the two faults that stayed invisible in homelab#1589 — a consume that failed
(silently, on the pre-#1590 mail path, which only POSTed) and a starved celery worker
leaving tasks queued for hours. Every five minutes the flow reads it with the
admin token (the one with `view_paperlesstask`) and pushes, under the
`paperless-health` job:

- `paperless_tasks_failed` — unacknowledged `consume_file` tasks in `failure`.
  Acknowledging the task in the Paperless UI is what clears it.
- `paperless_task_oldest_unfinished_seconds` — age of the oldest task of any
  type still `pending` or `started`, 0 when none.
- `paperless_health_last_success_timestamp` — so a dead probe is itself loud.

Two page-of-one queries (`count` for the first, `ordering=date_created` for
the second) rather than a walk over the task list.

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
| `WEBDAV_URL` | yes | — | WebDAV base URL holding the scan queue (both flows use it) |
| `WEBDAV_USERNAME` | yes | — | WebDAV Basic-auth user |
| `WEBDAV_PASSWORD` | yes | — | WebDAV Basic-auth password |
| `WEBDAV_SCAN_PATH` | no | `/` | Path under `WEBDAV_URL` to drain; the mail flow queues into `mail/` beneath it |
| `SCAN_CRON` | no | `0 * * * *` | Sweep schedule for the `scan` deployment |
| `PAPERLESS_ADMIN_TOKEN` | no | falls back to `PAPERLESS_API_TOKEN` | Superuser Paperless token used by `enrich` |
| `ENRICH_SWEEP_CRON` | no | unset → sweep has no schedule | Cron for the `enrich-sweep` deployment |
| `ENRICH_SWEEP_BATCH_SIZE` | no | `20` | Documents per sweep run |
| `ENRICH_MODE` | no | `suggest` | `suggest` (ai_suggestions + dedicated queries) or `extract` (one structured query, needs the Ollama vars; homelab#1563). Unknown values fail the run |
| `ENRICH_SUGGEST_TIMEOUT` | no | `650` | Read timeout for `ai_suggestions`, in seconds (unused by `extract`; removal pending the #1563 gate) |
| `ENRICH_OLLAMA_URL` | no | unset → dedicated title/correspondent queries off | Ollama base URL for the queries enrich runs itself |
| `ENRICH_OLLAMA_MODEL` | no | unset → dedicated title/correspondent queries off | Model for those queries |
| `ENRICH_FALLBACK_TIMEOUT` | no | `300` | Read timeout for the dedicated Ollama queries, in seconds |
| `CORRESPONDENT_BACKFILL_CRON` | no | unset → backfill has no schedule | Cron for the `correspondent-backfill` deployment (registered only when the Ollama vars are set) |
| `CORRESPONDENT_BACKFILL_BATCH_SIZE` | no | `8` | Documents per backfill run |
| `PAPERLESS_HEALTH_CRON` | no | `*/5 * * * *` | Cron for the `paperless-health` deployment |
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
