# Raw-candidate reranking repair — staged, not deployed

This is a proposed replacement for Core/retriever.py, based on the deployed
Oracle snapshot. The branch-root deployed source remains unchanged.

The proposed order is raw retrieval, grounding, cross-encoder scoring of each
original chunk, the existing zero-logit heuristic cutoff, then same-source
neighbor stitching and final top_k selection. Negative single candidates cannot
bypass scoring. A zero stitching window remains zero. Lower-ranked anchors
already consumed by a higher-ranked group are skipped instead of inheriting a
score for absent text. Inspection uses the same selection helper as search.

The score cutoff is uncalibrated and is not proof of scene accuracy. It has not
been lowered or disabled. Deterministic query normalization remains a separate
proposal in staging/natural-queries; it is not part of this replacement file.

## Verification actually completed

Ten synthetic selection/stitching tests passed on Oracle in the new staging
directory /opt/omnissiah/staging/raw-rerank_20261105_175046. They exercise raw
scoring before stitching, same-source boundaries, single-candidate rejection,
zero windows, retained scored anchors, backfilling distinct groups, exact source
filtering, book filtering, no-reranker behavior, and inspection/search parity.
Retrieval candidates and relevance scores in these tests are synthetic.

evaluate_live.py fetched real deployed FAISS/BM25 candidates for the seven scene
requests and control queries, and loaded the real cached CrossEncoder. Its first
run failed while mapping metadata because some entries omit explicit chunk_id.
The runner was corrected locally to use the deployed positional-ID fallback,
but the corrected version was not uploaded/rerun: automatic approval review
failed because its usage limit was reached. No comparative selection result or
generated-answer accuracy result was established by that attempt.

Production Core/retriever.py, model settings, data/indexes and systemd service
were not modified or restarted. No activation backup or deployment approval
has been completed for this patch. Before activation, finish the real comparison,
review passages and regressions, create exact timestamped backups, and obtain
the user's production-change approval.

## Run in a new Oracle staging directory

Copy retriever.py, test_retriever.py, evaluate_live.py and tools/scene_cases.json
into that new directory. Keep /opt/omnissiah/staging/natural-queries/query_rewriter.py
available for the evaluator's separately labelled normalization comparisons.

```bash
cd /opt/omnissiah/app
OMNISSIAH_ACTIVE_PROFILE=dell_query .venv/bin/python /path/to/stage/test_retriever.py
nice -n 10 .venv/bin/python -u /path/to/stage/evaluate_live.py
```

The evaluator uses the live inspection endpoint for retrieval, then the real
offline CrossEncoder and staged selection helper. It does not instantiate a
second FAISS/BM25 index or embedder, rebuild data, or call an LLM. It is not a
full staged-API integration test. Passages and candidate caches stay on Oracle;
only a text-free summary should be considered for committing after review.

Update, 6 November: the later Horus comparison is recorded in
../../docs/oracle/HORUS_QUERY_DIAGNOSIS_2026-11-06.md. It did not establish an
answer-quality improvement, so this experiment remains excluded from releases.
