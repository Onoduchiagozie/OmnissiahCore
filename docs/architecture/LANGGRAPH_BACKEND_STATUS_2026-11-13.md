# LangGraph backend work — 13 November 2026

## Objective

Break the user's question into searches, select evidence from the requested
sources, assess coverage, retry within a finite budget, and use the selected
passages for the original question. Backend first. No push or deployment
until real-corpus tests and acceptance.

## Current checkpoint

- The existing RetrievalGraph is wired into normal retriever calls only when
  `retrieval.langgraph_retrieval` is enabled in private configuration. It is
  not enabled in the checked-in defaults or on the hosted Oracle release.
- Planning now adds a compact search made only from the user's own words.
- If a book filter or explicit source list is supplied and first-pass evidence
  is insufficient, the second pass can search already-loaded metadata inside
  those sources. Candidate memory is bounded; scanning still traverses metadata
  and can be slow. This is lexical recall, not proof of relevance.
- Do not add raw chunks back after stitching has removed overlapping material.
- Source filters, retry budget and these changes have tests with fake indexes.
- The existing evaluator accepts `--book-filter` and repeatable `--source`
  arguments and includes the Magnus, Fulgrim and Nuceria examples.

## What is not verified

Oracle's earlier live release returned no sources for the filtered Magnus
Focus and Wide Net checks, and an insufficient answer for the Horus Explorer
check. Its `/query/inspect` showed pre-reranking candidates; that is not proof
of the passages used in the final answer. These observations do not establish
that this recovery will fix the cases.

The running Oracle process uses about 6 GB RSS on an 11.7 GB machine. A
second full index/model process risks memory pressure on the live service.
The production Python environment did not have LangGraph installed when
checked. Do not install packages into it, restart it, or load a second full
runtime there without a safe isolated test setup. The evaluator runs without
generation and requires the real index, compatible models, cached BM25 and
LangGraph in a separate environment. Keep source passages and settings private.

Compare baseline and graph passages for the actual user cases, first with the
exact book filters and again without them. Review source text, scene identity,
chronology, false positives and timing. Only then test answer generation and
normal API paths. A graph's lexical coverage result and passing unit tests are
not factual verification. A bounded retry does not guarantee the best answer.

No source has been pushed, no production configuration or service changed,
and the new local changes are not yet ready for deployment.
