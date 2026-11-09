# Adaptive evidence recovery — staged, not deployed (2026-11-09)

## Purpose
Fix false "The Chronicles do not record this" responses by recovering evidence
when the original retrieval returns no surviving passages and by expanding
causal questions such as "why did horus turn traitor" without hardcoded answers.

## Current status — NOT A PRODUCTION FIX
Development branch: `dev/retrieval-repair-20261108`.
Production Oracle service and `oracle` branch are untouched.
No real-corpus improvement has yet been demonstrated.

## Staged implementation
- `Core/retrieval_recovery.py`: generic bounded query expansion; four hybrid
  searches; rank fusion; reranking of raw chunks; retains legacy candidates to
  mitigate regressions; source-aware chunk identity; book/source filtering.
- `Core/retriever.py`: opt-in `adaptive_retrieval` flag, default **false**.
  Original retrieval is `_search_base`. Causal questions or empty selections
  trigger recovery only when the flag is enabled.
- Prior staged changes also make neighbouring chunk stitching same-source and
  expose selected evidence through inspection.
- `app_text.json` and `Core/prompt.py` no longer falsely claim that source
  books do not record an event simply because retrieval was insufficient.
- `Tests/test_retrieval_recovery.py`: five deterministic unit/mock tests.
- `tools/evaluate_retrieval_recovery.py`: real FAISS/BM25/reranker A/B
  comparison with no generation calls.

## Non-negotiable benchmark
Run the real-corpus script from an isolated development-branch checkout while
using Oracle's existing runtime config, assets and venv:

```bash
/opt/omnissiah/app/.venv/bin/python tools/evaluate_retrieval_recovery.py \
    --runtime-root /opt/omnissiah/app --case horus
```

Then run `--case all` if the initial test is safe and useful.

This script only reads the production model/index assets and writes a JSON
report outside the app (default `/tmp/omnissiah-retrieval-ab.json`). It must
not replace production source or restart systemd. It does load a second
retriever/model process, consuming temporary CPU/RAM. Check available memory
and obtain permission before running it on the production VM.

Score the selected **passages**, not only their book titles or counts. Compare
the causal evidence for Horus and all seven original scene requests. Inspect
false positives and regressions (notably the earlier raw-reranking regressions).

## Deployment gate
Do not enable `retrieval_cfg['adaptive_retrieval']` in the production service
until real-corpus tests improve evidence recall/precision without unacceptable
regressions and the user approves a backed-up production change.

LangGraph is proposed for orchestration, but has not been integrated into
this patch. It must not replace evidence validation or improve scores by
definition. No paid inference or billing is involved.
