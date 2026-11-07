# Oracle query repair — 5 November 2026

## Follow-up: incomplete answers and streaming, 07:15:27 UTC

User tests exposed incomplete sentences and words. The earlier 13 checks only established nonempty output and transport completion; they did NOT establish complete Gemini generation. Their success must not be interpreted as a finished production milestone.

A raw 4096-token Gemini probe returned `STOP`, 2163 characters, 1542 thinking tokens and 409 answer tokens. Oracle's old 1500-token cap was below that observed combined thinking-and-answer usage. A 1500-token comparison request returned 429, so its finish reason was not observed during this diagnostic.

Google then reported primary-model daily free-tier requests exhausted (20 requests/day), with a 2765-second cooldown at probe time. Both primary and fallback returned 429 during real staging smoke tests. Do not keep calling generation until quota resets or the user makes a billing/model decision. No billing or model change was made.

Deployed follow-up with 10 passing offline tests and a successful startup check. Backups and `deployment.json`: `/opt/omnissiah/app/backups/output_completion_20261105_071527/`.

- Oracle `dell_query` max_tokens raised from 1500 to 8192. The `lenovo_build` profile was compared and preserved.
- Gemini provider requires `STOP` to declare success; token limits, blocks, and missing completion reasons produce explicit errors.
- Stream errors no longer attach successful sources or report the complete stage; failed responses do not enter conversation memory.
- Multiline SSE tokens prefix every line with `data:` so clients retain paragraphs.
- Long provider quota cooldowns skip pointless immediate retries on that model.
- Test coverage includes partial stream token-limit termination, clean EOF without terminal frame, auth rejection, fallback regression, no midstream model switch, multiline events, failed state, no success sources after errors, and quota cooldown handling.

Full live 8192-token answer-length verification remains pending because of quota. Book filtering is supported by `book_filter`; the earlier user test requests omitted it. Non-streamed JSON routes are `/query/explore` and `/query/focus`. Narrate and Focus/stream intentionally return SSE; raw `data:` markers are not presentation-ready text.

## Requested OpenAI migration — staged, not activated

The user requested OpenAI instead of Gemini. Checked only credential presence: Oracle's environment file has a Gemini key and no OpenAI key. No secrets were printed. Prepared `/opt/omnissiah/staging/openai-migration/openai_provider.py`, staged `agent.py`, and `test_openai.py`; all nine offline tests passed and staged Python compiled. These staging tests make no provider calls.

The adapter uses OpenAI Responses with `store: false`, bounded attempts and transport timeout, configurable primary/fallback model list, explicit incomplete-response errors, and no switching after streamed text. `insufficient_quota` stops immediately with a billing/credit explanation. Agent dispatch uses a new explicit `openai_responses` provider value, preserving existing local OpenAI-compatible provider behavior. Production remains Gemini pending secure key provisioning, real model-access verification, model selection, a staged live sync/stream test, and a backed-up activation. No OpenAI model was selected or claimed accessible yet.

Environment: Dell is the operator machine; Oracle is production; Lenovo is the local build machine. Lenovo was never accessed or modified. Temporary Oracle source copies on Dell were used for staging, not merged into TheForge.

## Production changes

Earlier retrieval deployment: full-pool cross-encoder reranking, negative-score rejection, and Explorer evidence instructions. Backups: `/opt/omnissiah/app/backups/*pre_quality_20261105_030353*`.

Gemini repair backup directory: `/opt/omnissiah/app/backups/query_repair_20261105_033414/`.

- `Core/gemini_provider.py`: bounded retries for 429/500/502/503/504; primary model then configured fallback; no switch after emitted text; safe errors without API credentials or provider body disclosure.
- `Core/agent.py`: honor synchronous requests, use the provider in synchronous and streamed generation, avoid saving error responses in session memory. Other provider paths remain in place.
- `config.json`, Oracle `dell_query` LLM: primary `gemini-3.6-flash`, fallback `gemini-3.5-flash`, one retry per model.

No dependencies installed for this repair. Service restarted only after an idle check. Deployment script restores prior files if startup health verification fails.

## Evidence

- Seven mocked provider tests passed: sync fallback, fatal authentication stop, retry same model, streaming fallback before output, no switch after partial output, bounded exhaustion, empty output error.
- Staged agent passed real sync and streaming generation.
- Production passed every registered query route using Ferrus/Fulgrim: `/query/explore` (6 sources), `/query/focus` (2), `/query/focus/stream` (2), `/query/narrate` (6), `/query/inspect` (raw diagnostic hits), `/query/ghost` (clean unmatched response).
- Streaming checks required text, a source event, no error marker, and exactly one DONE event.
- All six routes rejected blank input with HTTP 400.
- Weak Isstvan opening-fire evidence remained withheld with zero sources.
- Final queue state green, active 0, queued 0. Approximately 6.4 GB available RAM.
- Public ngrok POST to `/query/ghost` worked from Dell. GET requests can return ngrok's free-tunnel landing page unless API clients supply `ngrok-skip-browser-warning: 1`.

Results saved on Oracle: `/opt/omnissiah/app/backups/query_repair_20261105_033414/live-results.json`. Deployment details and original file existence are in `deployment.json` in the same directory. Tests and deployment script remain under `/opt/omnissiah/staging/query-repair/`.

## Limits and remaining work

- Tests establish the listed requests work, not universal answer accuracy or permanent provider availability.
- Ghost's scene index is absent; unmatched is expected. Focus uses live corpus fallback.
- The Isstvan passage has not been recovered. Rejection is safer behavior, not a solved retrieval recall problem.
- Cross-encoder zero is an empirically chosen floor tested on few examples, not calibrated truth probability. One-candidate search still bypasses its floor and needs correction in a separate reviewed change.
- Inspect shows raw retrieval stages, not final reranked/filtered production selection.
- Broader relevance evaluation, duplicate editions, source score labeling, potential stitching across source boundaries, and request cancellation handling remain to review.
- A bare `/query` route is not registered. The six routes above are the current API contract.
- GitHub and Lenovo/Oracle reconciliation have not begun. Future code reconciliation must preserve both trees and keep runtime data and secrets out of Git.

