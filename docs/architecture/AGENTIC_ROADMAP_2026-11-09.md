# OmnissiahCore — agentic RAG development plan
Date: 2026-11-09
Status: approved architecture direction, implementation NOT complete; no production deployment authorized
Source base: `oracle` -> `dev/retrieval-repair-20261108`
Target development machines: Lenovo (GPU/index work), Dell (testing/frontend), Oracle (production API)
Frontend: separate `Onoduchiagozie/TheForge` repository

## Problem to solve
Normal queries such as `why did horus turn traitor` can return "The Chronicles do not record this."
That response is not evidence that the novels lack the answer. Raw FAISS/BM25 matches have
sometimes been discarded by grounding, cross-encoder score floor, or context assembly.
Seven real scene cases remain mostly unanswered or unverified. No test passes by HTTP 200,
a completed SSE stream, or mere generated text.

## Non-negotiable constraints
- Preserve Lenovo's real unpushed working tree: new isolated checkout; no force-pull/reset/clean.
- Do not modify Oracle production without exact-file backups, tested release and user approval.
- Free generation: no paid API activation, no key publication, no LangSmith Cloud dependency.
- Use existing FAISS 498850 vectors, BGE-M3, BM25, cached cross-encoder, Gemini and FastAPI.
- No initial rebuild of the large PDF corpus; no automatic source ingestion or book redistribution.
- Preserve source provenance, explicit book/source filters, and narrators' third-person grounded style.
- Production code currently uses `dell_query`; do not rename it without a coordinated migration.
- Changes in GitHub do not deploy to /opt/omnissiah/app automatically.
- User questions are ordinary natural language; query expansion must not require user-entered aliases.

## Architecture and boundaries
`TheForge` (Blazor Server) -> FastAPI query routes -> `RetrievalOrchestrator` (LangGraph)
-> independently unit-testable Python services -> existing `OmnissiahRetriever` and models
-> `EvidenceVerifier` -> `AnswerComposer` (existing Gemini provider) -> SSE/JSON -> TheForge.

Proposed OOP services:
- `QueryInterpreter`: original query, intent/entity detection, spelling/alias normalization (staged QueryRewriter is candidate).
- `EvidencePlanner`: bounded, generic candidate searches and missing-evidence plans. No hardcoded answers.
- `HybridRetrieverAdapter`: call the existing FAISS/BM25 index.
- `CandidateRanker`: evaluate original retrieval candidates before heavy context expansion; test competing fusion approaches.
- `ContextAssembler`: stitch within the same source, preserve exact chunk IDs/provenance and chronological hints.
- `EvidenceVerifier`: inspect relevance and factual coverage; not equivalent to a cross-encoder cutoff.
- `RetrievalOrchestrator`: LangGraph graph with state, branches, retry budget and stop conditions.
- `AnswerComposer`: provider-independent generation, citations, error handling, sync+SSE modes.

The retriever remains a backend service at first. LangGraph coordinates decisions and
calls its functions, but does not automatically improve search relevance. Postpone large
internal refactors until evaluation passes.

## Proposed graph state
Original query, session ID, intent, entities, explicit filters, query plan,
candidate chunk identities/scores (not 498k texts), selected evidence, provenance,
missing evidence types, search attempts, answer status, generation status,
source-backed claims and trace/latency metrics.

Core graph: interpret -> plan -> retrieve -> fuse/rank -> assemble -> verify
-> if incomplete and search budget left: plan again; else compose or acknowledge
verified limitations -> optionally verify citations -> answer.

Stop conditions: finite retrieval passes (suggest initial max 2), finite query variants
(suggest initial max 4), bounded candidate pool, per-node deadline, cancellation,
model quota and error classification. These are starting experiment budgets, not production defaults.

## Milestone 1 — baseline and retrieval repair
Deliver: reviewed scene gold evidence for Horus and seven user cases; final selected-hit
traces instead of raw-stage-only inspection; same-source stitching, explicit 0 windows,
BGE-M3 raw/prefixed A/B, reranker window analysis and calibrated filtering.
Use real indexed data without generation calls. Record per-case hits, missing evidence,
relevant chunk IDs and memory/latency. Distinguish mock/unit tests from real search.

Gate: evidence quality improves on held-out and original user queries, with no
material regression on previously successful Magnus/Fulgrim scenes.

## Milestone 2 — LangGraph retrieval workflow
Deliver: deterministic interpreter/planner, bounded multi-query retrieval, candidate
fusion, gap-aware evidence recovery, typed graph state, decision trace, cancellation.
A/B against legacy retrieval. Keep `adaptive_retrieval` opt-in until proven.

Gate: measurable recall/precision and question coverage gains in real corpus without
uncapped CPU cost, no hardcoded Horus-specific answer.

## Milestone 3 — API, TheForge, streaming and persistence
Deliver: integrate graph behind existing POST /query/inspect, /explore, /focus,
/narrate and /focus/stream routes; keep /query/ghost scene archive semantics.
SSE retains readable token, stage, sources, errors and completion; stage events represent
real nodes. Verify multiline SSE, disconnections/cancellation, no [DONE] after failures,
no accidental global default session mixing. Reuse prepared TheForge review fixes,
preserve Dell user changes and publish only after full browser verification.

Start with in-memory graph checkpointer for development; use private SQLite only if
resumable run state is needed. Checkpointed graph state is NOT the FAISS vector store and
is NOT a substitute for a user database. Enforce per-user sessions, retention and privacy.

Gate: user types previously failing question in REAL TheForge browser, request reaches
Oracle, evidence and source metadata display correctly, and stream completes accurately.
No "all passed" based merely on unit tests.

## Milestone 4 — narrative engine
Deliver: a separate workflow consuming VERIFIED evidence and grounded narration.
Segment scene/action/dialogue, detect supported speakers and emotions, build typed
screenplay JSON with provenance, consistent voice registry, offline TTS, six mood music
categories, sound effects, mixing/ducking, timing/synchronization and cancellation.
Do not use protected commercial soundtracks as bundled audio.

Gate: sample multi-speaker scene with verified source grounding, stable speaker labels,
playable synchronised narration, clear fallback when speaker unknown; benchmark resource use.
Do not introduce TTS work into the current retrieval hot path prematurely.

## Git/deployment workflow
Both machines can contribute to a temporary development branch in ONE source repository.
Do not treat `lenovo` and `oracle` branches as automatic runtime profiles.
Never overwrite unpushed original Lenovo code. Review PR -> commit main ->
record exact release SHA -> backup Oracle source/config -> controlled deployment ->
health and **real query** smoke -> rollback option; delete redundant branches only after
their unique content is safely reconciled and user approval.

## Test commands and evidence
Run existing no-LLM tests only in an isolated environment; their mock pass does not
prove search accuracy. `tools/evaluate_retrieval_recovery.py` can compare baseline and
staged retrieval using actual Oracle model/index assets, but opens a second retrieval
process and requires adequate memory. The Dell launcher performs a memory check, stages
temporary source under /tmp and saves the resulting JSON report. It does not deploy.

Formal production approval is a separate decision after reviewing the A/B report and
running actual TheForge-to-Oracle browser tests.
