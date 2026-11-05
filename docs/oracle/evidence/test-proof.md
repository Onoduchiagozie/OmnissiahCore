# Oracle test proof — 5 November 2026

Rerun recorded at 5 November 2026, 10:52 WAT (2026-11-05 09:52:18 UTC).

## Follow this order

1. Read the query examples: the short input is preserved and search wording is expanded inside the app.
2. Read the exact test-runner output below. These are staging unit/mock tests.
3. Check the scene results: five evidence refusals and two unverified generated narratives. The application is not yet proven accurate for your requested scenes.

## 21 query-handling checks — actual output

Command:

```bash
cd /opt/omnissiah/app && .venv/bin/python /opt/omnissiah/staging/natural-queries/test_queries.py -v
```

Exit code: 0

```text
test_comparison_instructions_automatic (__main__.QueryTests) ... ok
test_existing_full_name_not_duplicated (__main__.QueryTests) ... ok
test_first_shot_remains (__main__.QueryTests) ... ok
test_injected_aliases (__main__.QueryTests) ... ok
test_no_global_numeral_rewrite (__main__.QueryTests) ... ok
test_question_intent_preserved (__main__.QueryTests) ... ok
test_short_comparison (__main__.QueryTests) ... ok
test_soul_and_daemon_spelling (__main__.QueryTests) ... ok
test_spelling_and_numerals (__main__.QueryTests) ... ok
test_unknown_name_preserved (__main__.QueryTests) ... ok
test_user_text_not_promoted_to_system_instruction (__main__.QueryTests) ... ok
test_users_scene_name_typos (__main__.QueryTests) ... ok
test_versus_period (__main__.QueryTests) ... ok
test_why_guidance_preserves_user_question (__main__.QueryTests) ... ok
test_book_filter_not_overridden (__main__.RetrievalTests) ... ok
test_search_uses_plan_and_zero_stitching (__main__.RetrievalTests) ... ok
test_single_candidate_cannot_bypass_floor (__main__.RetrievalTests) ... ok
test_single_letter_does_not_match_inside_word (__main__.RetrievalTests) ... ok
test_spelling_variants (__main__.RetrievalTests) ... ok
test_terms_are_whole_words (__main__.RetrievalTests) ... ok
test_variants_count_once (__main__.RetrievalTests) ... ok

----------------------------------------------------------------------
Ran 21 tests in 0.004s

OK
[Config] Loaded profile: 'dell_query' | role: query
```

## 9 OpenAI adapter checks — actual output

Command:

```bash
cd /opt/omnissiah/app && .venv/bin/python /opt/omnissiah/staging/openai-migration/test_openai.py -v
```

Exit code: 0. All OpenAI HTTP responses were mocked; no live OpenAI request was made.

```text
test_account_quota_not_retried (__main__.Tests) ... ok
test_bounded_exhaustion (__main__.Tests) ... ok
test_fatal_auth (__main__.Tests) ... ok
test_partial_stream_never_switches (__main__.Tests) ... ok
test_stream_incomplete_and_missing_end (__main__.Tests) ... ok
test_stream_success_with_fallback (__main__.Tests) ... ok
test_sync_fallback (__main__.Tests) ... ok
test_sync_incomplete (__main__.Tests) ... ok
test_timeout_fallback (__main__.Tests) ... ok

----------------------------------------------------------------------
Ran 9 tests in 0.007s

OK
```

## Actual staged query examples

### ferrus vs fulgrim

- Original preserved: `ferrus vs fulgrim`
- Internal search: `Compare Ferrus Manus and fulgrim.`
- Search keywords: `Ferrus Manus fulgrim`
- Hidden response instructions: Answer as a comparison of the named subjects using the supplied evidence. Explain supported similarities, differences, and relevant relationships. Do not assume the user meant only a duel or invent a hypothetical winner.

### sharokin vs lucius

- Original preserved: `sharokin vs lucius`
- Internal search: `Compare Sharrowkyn and lucius.`
- Search keywords: `Sharrowkyn lucius`
- Hidden response instructions: Answer as a comparison of the named subjects using the supplied evidence. Explain supported similarities, differences, and relevant relationships. Do not assume the user meant only a duel or invent a hypothetical winner.

### what happened on istvan five?

- Original preserved: `what happened on istvan five?`
- Internal search: `what happened on Isstvan V?`
- Search keywords: `happened Isstvan V`
- Hidden response instructions: Answer the original question directly using the supplied evidence. For a broad question give a coherent overview; respect any scope the user supplied.

### kakaponi of the 3rd legion attacked the 5th legion

- Original preserved: `kakaponi of the 3rd legion attacked the 5th legion`
- Internal search: `Kakophoni of the Emperor's Children attacked the White Scars`
- Search keywords: `Kakophoni Emperor's Children attacked White Scars`
- Hidden response instructions: Answer the original question directly using the supplied evidence. For a broad question give a coherent overview; respect any scope the user supplied.

### why did horus turn traitor?

- Original preserved: `why did horus turn traitor?`
- Internal search: `why did horus turn traitor?`
- Search keywords: `horus turn traitor`
- Hidden response instructions: Answer the why-question using causes supported by the supplied evidence. Distinguish stated motives from inference; do not invent missing causes.


## Independent agent review

Counts and assertions reviewed. Passing local/mock tests do not prove corpus accuracy, live OpenAI compatibility, activation, or production behavior.

1. Book-filter test uses zero stitching and one fake hit; it does not check source boundaries of neighboring chunks.
2. Retrieval tests bypass embeddings, FAISS/BM25, real reranker inference and realistic multi-candidate ranking.
3. All nine OpenAI tests set retries_per_model=0; backoff, total deadlines and retry counts above zero need separate tests.
4. Phrase-preservation and system-marker-placement assertions are narrow checks, not answer-quality or security validation.

## Activation checked on Oracle

- Live provider: gemini
- Live model: gemini-3.6-flash
- Live output allowance: 8192
- Query rewriter deployed: false
- OpenAI adapter deployed: false
- Production changes during this evidence run: none

## Prior live scene results

| Case | Result | Characters |
|---|---|---:|
| loken-jubal | Insufficient evidence | 47 |
| horus-invisibles | Insufficient evidence | 47 |
| kakophoni-white-scars | Insufficient evidence | 47 |
| lorgar-magnus | Generated; source review pending | 6270 |
| sea-of-souls | Generated; source review pending | 5507 |
| white-scars-daemon | Insufficient evidence | 34 |
| sharrowkyn-lucius | Insufficient evidence | 47 |

## Next engineering task

Repair raw-passage reranking and source-aware stitching in staging, then evaluate the seven requested scenes before production activation.

The test counts above are not evidence that these seven scenes are accurately answered.

## Tested-file SHA-256 fingerprints

- `natural-queries/query_rewriter.py`: `a5744adc7d0c49c1bf93fac7582c276a875f14ba47eb1a9b07969520b4300850`
- `natural-queries/retriever.py`: `e7149241c70271f575d2c8fba8c78c0f0ef26f8c96402da623ee6efc22f4ddea`
- `natural-queries/prompt.py`: `5e33f5799cd7b26bb9c9ee2644143b8cadaeb18d5753f997c0ec1ea94f3358f8`
- `natural-queries/test_queries.py`: `df02519e57896cbf6af04456cb80acaad9849cee5c0ce7d366cbc0b90a599576`
- `openai-migration/openai_provider.py`: `ae96d800d982e8d01282dd351f3d6b4e6bb9634e5d82b5b3dda7a6d875a905c8`
- `openai-migration/agent.py`: `39a58d14d6bc4b51cf71f9c2a5812705d245ae7020d23f5f817f6ac7115ca0c5`
- `openai-migration/test_openai.py`: `3778df9b511c3889f68fe7a269bfc083a255b327c8213ed426a3bcf8764b449a`
