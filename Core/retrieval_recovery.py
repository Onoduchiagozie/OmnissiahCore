"""Bounded evidence recovery over the existing OmnissiahCore hybrid indexes.

This module is dependency-free; the retriever supplies FAISS, BM25,
rank fusion, cross-encoder and source-aware stitching. The original query is
preserved for ranking and answer generation. No Warhammer facts are embedded.
The experiment is opt-in and must pass a live-corpus A/B evaluation.
"""
from __future__ import annotations

import re
from collections import defaultdict
from hashlib import sha256

_CAUSE = re.compile(r"^\s*(?:why\s+(?:did|does|do|was|were|is|are)\s+|what\s+(?:made|caused|led to)\s+)", re.I)


def is_explanatory(query: str) -> bool:
    return bool(_CAUSE.match(query)) or bool(re.search(r"\b(?:reasons?|causes?|motivations?)\b", query, re.I))


def plan_queries(query: str, limit: int = 4) -> list[str]:
    """Generic evidence-seeking variants. Never replaces the user's question."""
    original = query.strip()
    if not original:
        return []
    core = re.sub(r"[?!.]+$", "", original).strip()
    variants = [original]
    if is_explanatory(core):
        subject = _CAUSE.sub("", core, count=1).strip() or core
        variants.extend([
            f"causes and motivations behind {subject}",
            f"events and influences leading to {subject}",
            f"explanations and reasons for {subject}",
        ])
    else:
        variants.extend([
            f"events and context surrounding {core}",
            f"account of {core}",
            f"characters and actions involved in {core}",
        ])
    unique, seen = [], set()
    for item in variants:
        key = re.sub(r"\W+", " ", item).strip().casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:max(1, min(limit, 6))]


def _identity(chunk: dict) -> tuple:
    """Source-aware identity, retaining provenance of duplicate editions."""
    source = chunk.get("source", "")
    chunk_id = chunk.get("chunk_id")
    if chunk_id is not None:
        return source, str(chunk_id)
    digest = sha256(chunk.get("text", "").encode("utf-8", "replace")).hexdigest()
    return source, digest


def recover_evidence(
    retriever,
    query: str,
    baseline: list[dict],
    *,
    top_k: int,
    candidate_pool: int,
    stitching_window: int,
    book_filter: str | None = None,
    source_filter: list[str] | None = None,
    max_queries: int = 4,
    on_stage=None,
) -> list[dict]:
    """Try complementary hybrid searches, then rerank raw passages.

    Retains some legacy evidence: raw-first ranking alone previously caused
    genuine-scene regressions. Reranker logits are not treated as truth scores.
    """
    queries = plan_queries(query, max_queries)
    if len(queries) < 2:
        return baseline[:top_k]
    if on_stage:
        on_stage("evidence_recovery")

    pool = min(100, max(40, candidate_pool))
    evidence: dict[tuple, dict] = {}
    scores = defaultdict(float)
    source_set = set(source_filter) if source_filter else None

    for variant_number, variant in enumerate(queries):
        dense = retriever._faiss_search(retriever._enrich_query(variant), k=pool)
        sparse = retriever._bm25_search(variant, k=pool) if retriever.bm25 else []
        merged = retriever._rrf_merge(dense, sparse, k=pool)
        for rank, candidate in enumerate(merged):
            source = candidate.get("source", "")
            if book_filter and book_filter.casefold() not in source.casefold():
                continue
            if source_set is not None and source not in source_set:
                continue
            identity = _identity(candidate)
            evidence.setdefault(identity, dict(candidate))
            weight = 1.5 if variant_number == 0 else 1.0
            scores[identity] += weight / (60 + rank + 1)

    if not evidence:
        return baseline[:top_k]

    ordered = sorted(evidence, key=lambda key: scores[key], reverse=True)
    ordered = ordered[:min(len(ordered), 64)]
    if retriever.reranker is not None:
        pairs = [[query, evidence[key].get("text", "")] for key in ordered]
        raw_scores = retriever.reranker.predict(pairs)
        for key, value in zip(ordered, raw_scores):
            evidence[key]["recovery_raw_rerank_score"] = float(value)
        raw_order = sorted(
            ordered,
            key=lambda k: evidence[k].get("recovery_raw_rerank_score", float("-inf")),
            reverse=True,
        )
        raw_rank = {key: rank for rank, key in enumerate(raw_order)}
        prior_rank = {
            key: rank for rank, key in enumerate(
                sorted(ordered, key=lambda k: scores[k], reverse=True)
            )
        }
        ordered.sort(
            key=lambda key: 1 / (60 + raw_rank[key] + 1) + 1 / (60 + prior_rank[key] + 1),
            reverse=True,
        )

    keep_legacy = min(len(baseline), max(0, (top_k + 1) // 2))
    final = [dict(chunk) for chunk in baseline[:keep_legacy]]
    used = {_identity(chunk) for chunk in final}
    shortlist = [evidence[key] for key in ordered if key not in used]
    stitched = retriever._stitch_chunks(
        shortlist[:max(top_k * 4, 12)],
        window=stitching_window,
    )
    for chunk in stitched:
        if len(final) >= top_k:
            break
        identity = _identity(chunk)
        if identity not in used:
            final.append(chunk)
            used.add(identity)
    for chunk in baseline[keep_legacy:]:
        if len(final) >= top_k:
            break
        identity = _identity(chunk)
        if identity not in used:
            final.append(dict(chunk))
            used.add(identity)
    return final
