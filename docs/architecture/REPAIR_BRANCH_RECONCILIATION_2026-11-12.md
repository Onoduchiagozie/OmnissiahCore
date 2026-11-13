# Repair branch reconciliation — 12 November 2026

One temporary integration branch: codex/recovery-20261112. The permanent target
is main. Nothing in this recovery has been pushed or deployed. Do not delete
remote branches until Lenovo's unpushed work is preserved and main is accepted.

## Backend decisions

| Source branch | Disposition in the integration |
|---|---|
| main (668b91a) | Keep standalone setup, private settings, release identity, packaging and rollback checks. |
| dev/retrieval-repair-20261108 | Merge its RetrievalGraph, recovery, third-person prompts, safe stitching and conversation store. Preserve main's device-resolution and /version fixes. |
| fix/oracle-causal-retrieval-20261110 | Port causal candidate selection and bounded context from the later event-anchor branch. Replace rebellion-specific rewrites with question-derived probes, including lowercase queries. |
| fix/oracle-chronicle-coverage-20261111 | Port chronicle facet coverage and raw-passage reranking from the later event-anchor branch. |
| fix/oracle-adaptive-voice-20261111 | Port independent response_style/retrieval_scope, preview/respond/evidence endpoints. Reject its regressions to shared sessions and eyewitness narration. |
| fix/oracle-eval-corpus-20261111 | Preserve 100-query evaluation dataset, helper tests and evaluation tools. Historical observed results are not results from this recovery. |
| fix/oracle-event-anchor-recall-20261111 | Preserve event-anchor recall and tests, gated separately until full-corpus cost/accuracy are measured. |
| test/nonhorus-coverage-20261111 | Preserve additional 100-query dataset and checks. This sibling branch omits event_recall.py; do not overwrite the event branch with it. |
| fix/stream-terminal-contract-20261111 | Empty-answer/provider-failure behavior superseded by the shared API validator and actual runtime tests. No second terminal-contract helper needed. |
| fix/fastapi-truthful-sse-20261111 | Port validator and tests. Restore legacy routes, use it for all four streaming modes, bound buffering, close iterators and cancel queued work. |

Branch-specific deployment scripts and live-production CI jobs are deliberately
superseded by the existing main-only release workflow. Old scripts must not
reintroduce hardcoded repair branches into production.

## Frontend decisions

| Source branch | Disposition |
|---|---|
| codex/oracle-frontend-integration-20261106 (d1ac716) | Keep feature layout, published CSS versioning, workspace sessions and conversation-revision cache fixes. Port the session changes missing from Dell. |
| fix/forge-sse-terminal-20261111 | Port terminal/error behavior to Features/Workspace services. Do not restore the deleted monolithic workspace page. |
| fix/stream-protocol-nonhorus-20261111 | Retain multiline/event boundaries, failed-stage, EOF and empty-answer checks with tests of actual gateway and browser. Standalone legacy harness is superseded by the main application tests. |
| Dell uncommitted typography | Preserve per-role Settings, bundled fonts and Scene Archive styling. Browser test verifies persistence across reload/navigation. |

Public provider/model labels introduced by repair-branch health-display commits
are excluded, following the user's instruction. Existing UI tests verify they
are absent.

## Runtime selection and limits

All experimental retrieval switches remain off in the checked-in config.
In a private machine configuration, choose one:

1. langgraph_retrieval=true: RetrievalGraph owns recovery.
2. retrieval_quality=true: causal/chronicle selection owns candidate expansion.
   event_anchor_recall=true additionally enables the metadata scan.
3. adaptive_retrieval=true: inherited recovery path.
4. All false: existing baseline.

Selection precedence is graph, quality, adaptive, baseline; tests ensure paths
do not run on top of one another. EvidenceGraph's outer retrieval retries are
one when any recovering retriever is selected, otherwise at most three.
The metadata fallback bounds returned candidates, not full-corpus scan time.
It still relies on capitalized names and is a heuristic, not factual proof.

OMNISSIAH_GRAPH_ENABLED=1 registers the separate experimental JSON graph routes.
The Blazor SSE routes continue using the ordinary generation pipeline. The graph
is not silently presented as a token-streaming feature. Optional verification
is an extra model call and has not been activated.

The HTTP stream succeeds only after nonempty answer text and a completion stage.
Sources alone, EOF, failed stage, provider error or worker exception cannot emit
DONE. Error details stay in server logs. Failed/empty/cancelled requests are not
saved as completed history. Waiting stream work is cancellable; an already
blocking model read still depends on its provider timeout.

## Validation and remaining release gates

- Backend: 152 unit/integration cases plus 23 retrieval/session cases passed.
  Includes real LangGraph and real FastAPI HTTP routes with mocked ML/corpus.
- Frontend: 46 tests passed, including real browser interactions with a local
  deterministic API. This is transport/UI evidence, not lore-accuracy evidence.
- Real corpus/GPU/provider accuracy, original Nuceria/Fulgrim/Magnus regressions,
  Lenovo working tree, production identity and final release remain pending.
- API-key authorization, user roles and per-user limits remain a separate
  unfinished scope; anonymous session IDs are not authentication.
- No main merge, remote branch deletion, push or deployment in this batch.

Published-file browser checks: 7 passed using isolated Testing keys. Production-host configuration remains unverified. Checkpoint completed 13 November 2026.
