"""Corpus-wide, query-derived event anchor recall.

FAISS/BM25 can miss a specific event in a large lore corpus even when the
right book is indexed. This is a bounded lexical fallback over ALREADY loaded
chunk metadata, NOT a replacement index and NOT a lore-specific book lookup.

Search uses named subjects and places explicitly provided in the question.
It retains multiple source perspectives; no book is hardcoded or auto-trusted.
"""
import re
from collections import Counter, defaultdict
from math import sqrt
from typing import Optional

from Core.retrieval_quality import is_archive_boilerplate

_NAME = re.compile(r"\b[A-Z][A-Za-z'-]{2,}\b")
_INTRO = {"tell", "narrate", "recount", "explain", "describe", "show",
          "take", "walk", "give", "what", "why", "where", "which", "how",
          "who", "when", "the", "full", "whole", "complete", "entire"}
_WORD = re.compile(r"\b[a-z][a-z'-]{2,}\b")


def named_event_anchors(query: str) -> tuple[str, list[str]]:
    """Return subject and up to 3 named anchors supplied by the user.

    Explicit places following 'on/in/at' are examined first. For example,
    'Horus ... on Davin ... Chaos' yields subject='horus', anchors=['davin',
    'chaos']. No extra lore names are generated.
    """
    found = []
    for match in _NAME.finditer(query):
        word = match.group().lower()
        if word not in _INTRO and word not in found:
            found.append(word)
    if len(found) < 2:
        return (found[0] if found else ""), []
    subject = found[0]
    others = found[1:]
    place_priority = []
    for name in others:
        if re.search(r"\b(?:on|in|at|near|towards?|from)\s+"
                     + re.escape(name) + r"\b", query, re.I):
            place_priority.append(name)
    ordered = place_priority + [name for name in others if name not in place_priority]
    return subject, ordered[:3]


def _token_positions(text: str, token: str, limit: int = 20) -> list[int]:
    # This is only called after cheap substring pre-filtering.
    return [m.start() for m in re.finditer(r"\b" + re.escape(token) + r"\b", text)][:limit]


def scan_event_anchors(
    metadata: list[dict],
    query: str,
    *,
    limit: int = 48,
    max_sources_per_anchor: int = 5,
    per_source: int = 5,
    book_filter: Optional[str] = None,
    source_filter: Optional[list[str]] = None,
) -> tuple[list[dict], dict]:
    """Recall real subject/event co-occurrence passages across the corpus.

    A source must repeatedly contain the SUBJECT and EVENT together to gain
    source support. A passing mention in an unrelated later novel will not
    automatically beat many direct scenes in the most relevant book.

    Metadata is already resident in the retriever. No embeddings, file reads,
    cache builds, or Gemini calls occur.
    """
    subject, anchors = named_event_anchors(query)
    diagnostics = {"subject": subject, "anchors": anchors,
                   "sources_by_anchor": {}, "scanned_chunks": len(metadata)}
    if not subject or not anchors or limit <= 0:
        return [], diagnostics

    allowed_sources = set(source_filter or []) if source_filter else None
    counts = {anchor: Counter() for anchor in anchors}
    best = {anchor: defaultdict(list) for anchor in anchors}

    for idx, chunk in enumerate(metadata):
        source = chunk.get("source", "")
        if book_filter and book_filter.lower() not in source.lower():
            continue
        if allowed_sources is not None and source not in allowed_sources:
            continue
        body = chunk.get("text") or ""
        low = body.lower()
        if subject not in low or is_archive_boilerplate(low):
            continue

        matching_anchors = [a for a in anchors if a in low]
        if not matching_anchors:
            continue
        positions_subject = _token_positions(low, subject)
        if not positions_subject:
            continue
        for anchor in matching_anchors:
            positions_anchor = _token_positions(low, anchor)
            if not positions_anchor:
                continue
            nearest = min(abs(s - a) for s in positions_subject
                          for a in positions_anchor)
            # Prevent a long chunk that mentions both names hundreds of
            # words apart from being treated as a direct scene.
            if nearest > 850:
                continue
            counts[anchor][source] += 1
            proximity = 7 if nearest <= 160 else 4 if nearest <= 350 else 1
            score = proximity + 2 * len(positions_anchor) + len(positions_subject)
            rows = best[anchor][source]
            rows.append((score, -nearest, idx))
            rows.sort(reverse=True)
            del rows[per_source:]

    results = []
    seen = set()
    for anchor in anchors:
        supported_sources = counts[anchor].most_common(max_sources_per_anchor)
        diagnostics["sources_by_anchor"][anchor] = supported_sources[:8]
        # Reserve slots across source perspectives, not only one edition.
        chosen_by_source = []
        for source, evidence_count in supported_sources:
            candidates = best[anchor][source]
            chosen_by_source.append((source, evidence_count, candidates))
        # Round-robin for source coverage; high-support sources first.
        for row in range(per_source):
            for source, evidence_count, candidates in chosen_by_source:
                if row >= len(candidates):
                    continue
                score, neg_distance, idx = candidates[row]
                chunk = metadata[idx]
                identity = (source, chunk.get("chunk_id", idx))
                if identity in seen:
                    continue
                seen.add(identity)
                results.append({
                    "chunk_id": chunk.get("chunk_id", idx),
                    "source": source,
                    "text": chunk.get("text", ""),
                    "chapter": chunk.get("chapter", "unknown"),
                    "file_type": chunk.get("file_type", "pdf"),
                    "event_anchor": anchor,
                    "event_source_support": evidence_count,
                    "event_anchor_proximity": -neg_distance,
                    "event_anchor_score": score,
                })
                if len(results) >= limit:
                    diagnostics["candidate_count"] = len(results)
                    return results, diagnostics

    diagnostics["candidate_count"] = len(results)
    return results, diagnostics
