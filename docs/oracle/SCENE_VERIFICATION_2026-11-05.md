# Oracle scene verification — 5 November 2026

## Current state

Oracle has generation-completion checks, an 8192-token allowance, Gemini retry/fallback handling, streamed responses, and local cross-encoder reranking. This does not establish retrieval accuracy. Lenovo has not been accessed or changed. OpenAI remains staged and inactive because usable API credit/key has not been established; no paid API usage was enabled.

The source snapshot and further staged work are saved on `Onoduchiagozie/OmnissiahCore`, branch `codex/oracle-stabilization-20261105`. The repository root remains the existing main code. `deployments/oracle/2026-11-05/app/` is the Oracle source snapshot; do not blindly deploy the root or copy it over Lenovo.

## User-requested scene tests, original natural wording

All seven raw retrieval requests completed. Their six-or-fewer hits do not prove that the requested scene was found.

All seven live narrator requests completed with their transport contract. Five returned an insufficient-evidence response. Two generated substantial text and need source verification. None is marked verified accurate merely because HTTP/SSE succeeded.

| Case ID | Request | Live result |
|---|---|---|
| `loken-jubal` | Loken saving a girl from possessed Jubal | Insufficient evidence; retrieved Horus Rising material includes parts of the fight, but final selected context was too thin. |
| `horus-invisibles` | Horus kills the Invisibles | Insufficient evidence; final selected source was Prospero Burns despite raw hits in Horus Rising. Candidate selection is wrong. |
| `kakophoni-white-scars` | Kakaponi/3rd Legion attack 5th Legion with sonic weapons | Insufficient evidence; raw hits were mostly unrelated codices. |
| `lorgar-magnus` | Lorgar and Magnus discussion | 6,270 characters generated; references Betrayer, The First Heretic and Aurelian. Source accuracy still unverified. |
| `sea-of-souls` | Lorgar's Sea of Souls/ocean analogy | 5,507 characters generated from The First Heretic. Exact requested analogy and speaker still need verification. |
| `white-scars-daemon` | Slaaneshi daemon battle during the journey to Terra | No relevant sources survived; no scene generated. |
| `sharrowkyn-lucius` | Sharokin/Lucius fight in Angel Exterminatus | Insufficient evidence; needs name normalization and better scene retrieval. |

Full raw and generated reports remain on Oracle in `/opt/omnissiah/staging/scene-verification/`. Source passages and full generated narratives are excluded from the GitHub test summary.

## Free query handling, staged only

`staging/natural-queries/` contains a separate `QueryRewriter`, an immutable `QueryPlan`, retriever integration, and prompt integration. It performs no LLM/network calls. Supported aliases include Ferrus, Istvan, Sharokin, Kakaponi and Slaneshi; unknown names and the user's original question are preserved. Comparison/why instructions are selected internally. It never assumes a requested comparison is a duel or silently chooses a book.

The staged matcher uses whole words, preventing `v` from matching Vulkan or `fire` from matching firedrake. It counts numeral variants once, respects a zero stitching window, and applies the relevance floor to one-candidate results. Twenty-one unit/regression tests passed on Oracle; syntax compiled. Production has not received these staged files.

A separate seven-query corpus sample showed spelling normalization and comparison rewriting can help, but do not solve retrieval by themselves. The generic reranker also assigned negative scores to relevant Laer-blade material. The current zero cutoff is not calibrated truth confidence and must be revisited against scene-specific evidence. The earlier comment calling zero a natural probability boundary is not a sound guarantee.

## Query endpoints

All are POST routes and accept JSON with a `query` string.

| Endpoint | Output | Purpose |
|---|---|---|
| `/query/inspect` | JSON, no generation | Raw FAISS/BM25/grounding/stitching diagnostics; does not show final reranking/floor selection. |
| `/query/explore` | JSON | Analytical/lore answer; currently primarily configured for object exploration. |
| `/query/focus` | JSON | Narrow answer with at most 1–3 source groups. |
| `/query/narrate` | SSE | Broad scene reconstruction, with evidence and completion events. |
| `/query/focus/stream` | SSE | Streamed narrow answer. |
| `/query/ghost` | JSON, no generation | Scene-index lookup only; Oracle's missing index means unmatched is expected. |

There is no bare `/query` endpoint on this deployment.

## Commands inside Oracle SSH

List the seven cases:

```bash
/opt/omnissiah/app/.venv/bin/python /opt/omnissiah/staging/scene-verification/verify_scenes.py --case list
```

Run all original human queries through retrieval without LLM usage:

```bash
/opt/omnissiah/app/.venv/bin/python /opt/omnissiah/staging/scene-verification/verify_scenes.py --mode inspect
```

Run one generated scene, showing readable streamed text instead of raw SSE markers:

```bash
/opt/omnissiah/app/.venv/bin/python /opt/omnissiah/staging/scene-verification/verify_scenes.py --mode narrate --case lorgar-magnus
```

Run a normal JSON answer:

```bash
/opt/omnissiah/app/.venv/bin/python /opt/omnissiah/staging/scene-verification/verify_scenes.py --mode focus --case sharrowkyn-lucius
```

The CLI defaults to the original short/misspelled wording. `--expanded` uses an explicit diagnostic query for comparison; `--book-filter "Angel Exterminatus"` adds a known book constraint. These options are engineering probes, not requirements for end users. Test one generated case at a time to conserve the free quota.

## Next engineering work

Fix candidate selection, source-aware stitching and reranker truncation/calibration using these scene cases as acceptance criteria. Validate complete supported scene coverage and correct identities, not only positive scores or completed transport. Review generated Lorgar answers against their exact source passages. Then seek approval for the tested production patch. Git commits on the review branch do not themselves deploy code to Oracle or alter Lenovo.
