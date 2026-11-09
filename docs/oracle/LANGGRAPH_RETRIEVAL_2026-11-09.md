# LangGraph evidence retrieval — development branch only (9 November 2026)

## Status
- Branch: `dev/retrieval-repair-20261108` based on Oracle source.
- Not merged to `oracle`, `lenovo`, or `main`.
- Not deployed; production active profile remains `dell_query`.
- No changes to Oracle systemd, API routes, paid API billing, index or models.
- Optional dependency: `requirements-langgraph.txt`. Never install on live
  Oracle without approval and tested rollback.

## Why
The historical engine sometimes selects too few or the wrong passages for
natural questions, then falsely says that the books do not record an event.
The first repairs on this branch include source-safe stitching, preserved
zero-window settings, an opt-in BGE-M3 raw-query comparison flag, real
selected-hit diagnostics and qualified insufficient-evidence wording.

## Graph implementation
- `Core/retrieval_nlp.py`: deterministic, source-independent intent,
  subject/action identification and search planning. Alias lexicon can be
  injected, but contains no hardcoded Horus or other story answers.
- `Core/graphs/retrieval_graph.py`: a genuine LangGraph StateGraph:
  interpret -> plan -> retrieve -> rank_raw -> stitch -> assess -> either
  recover -> retrieve or finish.
- Reuses the existing loaded FAISS, BGE-M3, BM25 and cross-encoder instances.
  Does not call a chat model, use LangSmith or fetch external content.
- Retrieval budgets: up to four initial queries, two recovery variants, two
  graph passes, 64 candidates for re-ranking, final top_k limited to 20.
- Every selected passage retains source/chunk ID and diagnostic stage trace.
- The cross-encoder scores raw passages before same-source neighbour assembly;
  no hard zero-logit factual-confidence cutoff is used inside the new graph.
- The lexical causal-evidence check is ONLY a bounded-retry heuristic;
  it is not a truth classifier, entailment model or historical proof.
- `Core/retriever.py` routes into the graph only if
  `retrieval_cfg['langgraph_retrieval'] == true`. Default absent/false.

## What has been tested
- Offline graph control flow and deterministic NLP through mocked FAISS/BM25
  data, with real LangGraph execution in GitHub Actions on Python 3.10/3.12.
- Offline recovery and source-boundary invariant tests.
- No real Oracle index query, benchmark accuracy, proof of correct Horus
  explanation, or full TheForge browser test has yet been reported.

## Read-only real-corpus comparison (requires user authorisation/resources)
From an isolated dev-branch checkout that can read Oracle's existing assets:

```bash
python tools/evaluate_langgraph_retrieval.py \
  --runtime-root /opt/omnissiah/app \
  --case horus \
  --out /tmp/omnissiah-langgraph-horus-ab.json
```

Requires the *isolated interpreter* to have the optional LangGraph dependency,
as well as the existing FAISS, embedding and reranking dependencies. It uses
the deployed root ONLY for data paths; the experiment runs staged source.
The evaluator refuses to rebuild a missing or stale BM25 cache.

WARNING: it loads a SECOND full retriever and cached models, consuming
Oracle RAM and CPU. Do not run it on a production VM without inspecting
free memory and obtaining permission. It writes only a temporary JSON report.

Review original query and all seven scene cases. Score the actual passage
content against verified book locations; inspect irrelevant high scorers,
missing evidence, source diversity, latency and memory. The scorecard must
not mark a positive HTTP result or a passage from the right *book* as
sufficient evidence.

## Required remaining work BEFORE production activation
1. Run and review real-corpus A/B; measure false positives and regressions,
   including previous raw-first reranker regressions.
2. Validate book filters/aliases on actual source metadata and lexical
   mismatch. Determine whether improved indexing or offline event extraction
   is required. Do not pretend a deterministic heuristic verifies claims.
3. Measure latency, CPU/RAM on Oracle; tune bounded query counts.
4. Connect the evidence package to grounded answer composition and later
   claim-level source validation; review actual generated answers.
5. Integrate and browser-test TheForge route/streaming with current Dell work,
   preserving uncommitted changes.
6. Only with explicit user approval: back up exact production files, deploy
   reviewed commit, install optional dependency, enable graph flag, restart
   service, smoke test, retain rollback plan.

## Longer-term architecture
Distinct modules for deterministic query interpretation, evidence planning,
hybrid retrieval, reranking, context assembly, evidence verification,
grounded answer synthesis and a separate LangGraph narrative/audio pipeline.

On Lenovo, optionally benchmark local instruction models for event extraction
and cause/participant/relation JSON from local book passages, with explicit
chunk attribution. This should run offline and populate a provenance-carrying
event index only after schema and license review. Do NOT silently add
unverified LLM-generated claims to factual evidence.

Content strategy: complement Deadlifts For The Dark Gods and Astartes
Anonymous videos with interactive, book-cited questions, scene chronology,
different witnesses, comparisons, and later narrated audio rather than copying
their videos/scripts, voices or artwork.
