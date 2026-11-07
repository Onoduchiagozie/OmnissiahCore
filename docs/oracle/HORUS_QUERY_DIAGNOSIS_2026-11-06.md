# Horus question: retrieval diagnosis (6 November 2026)

User query: `why did horus turn traitor`.

## Reproduced outcome

- A fresh-session request through Dell TheForge `/api/query/explore` returned HTTP 200, two source groups, and exactly `The Archives do not clearly record enough here.` (47 characters).
- The deployed Explorer prompt explicitly instructs Gemini to output that sentence and stop when the supplied passages mention a subject without establishing the requested fact. This is an evidence refusal, not a transport truncation in the observed run.
- Deployed Oracle `Core/prompt.py` line 220 was checked directly. TheForge `Models/QueryRequest.cs` does not send a session ID; deployed `runtime_service.py` maps missing IDs to shared `default` memory. Gemini temperature is 0.4. These are plausible contributors to variation, but this one fresh-session refusal shows memory alone does not explain it.

## Retrieval-only inspection

First six stitched sources from `/query/inspect`:

| Wording | `False Gods` sources in first six |
| --- | ---: |
| `why did horus turn traitor` | 1 |
| `why did Horus betray the Emperor` | 2 |
| `Horus betrayal Emperor Davin Serpent Lodge False Gods` | 4 |
| `Horus Lupercal rebellion betrayal Emperor` | 0 |
| `Horus turn traitor reasons betrayal` | 0 |

Simple synonym substitution is therefore not a reliable repair. The user should not need to write the explicit version.

## Real staged reranker comparison

An isolated evaluator in `/opt/omnissiah/staging/raw-rerank_20261105_175046/evaluate_horus.py` used live FAISS/BM25 candidates and the real cached CrossEncoder. It made no LLM calls and changed no production files. The corrected evaluator completed successfully. For the natural wording, deployed selection had two groups including one `False Gods` group; staged raw-first selection had six including one `False Gods` group. For `why did Horus betray the Emperor`, deployed selection had six including two `False Gods` groups; staged selection had six including **zero** `False Gods` groups. Brief inspection of the selected passages showed several high-scoring groups concerned other conflicts or later events.

The staged raw-first reranking patch is **not approved as an answer-quality fix** on this evidence. Its synthetic tests pass, but source relevance is still unreliable. No staged generation answer was verified.

## Next repair and gate

1. Build candidate gathering for broad cause questions using more than one evidence-seeking search, preserving the user's original wording for the answer. Do not hardcode unsupported lore into the final prompt.
2. Add source and passage relevance checks before generation; retain distinct, cause-bearing passages rather than treating a positive cross-encoder score as proof.
3. Isolate TheForge sessions instead of sharing Oracle's `default` memory; separately assess a lower generation temperature for reproducibility.
4. Test retrieval without LLM quota on this question and the seven existing scene cases. Review selected passages, not just book titles and scores. Generate a small number of final answers only after the passages pass.
5. Back up and obtain the user's approval before changing the Oracle production service. The current live service remains unchanged.

