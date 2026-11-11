"""Query-aware retrieval helpers. No model, corpus, or network dependencies."""

import re

_CAUSAL_QUERY = re.compile(
    r"\bwhy\b|\bwhat (?:caused|made|led|motivated|convinced)\b|"
    r"\breasons? (?:for|behind)\b|\bwhat was (?:the )?cause\b",
    re.IGNORECASE,
)
_QUERY_WORDS = {"why", "what", "when", "how", "where", "who", "which", "after", "before", "the", "a", "an", "did"}
_CAUSAL_CUES = re.compile(
    r"\b(?:because|therefore|since|as a result|for this reason|the reason|"
    r"motiv(?:e|es|ated|ation|ations)|ambition|resentment|jealousy|fear|fears|"
    r"deceiv\w*|decept\w*|manipulat\w*|persuad\w*|convinc\w*|"
    r"corrupt\w*|vision\w*|revealed|revelation|decid\w*|chose|"
    r"believ\w*|distrust\w*|betray\w*|conspir\w*|suspicion|doubt\w*)\b",
    re.IGNORECASE,
)
_UNANSWERED = re.compile(
    r"\b(?:could not|cannot|did not|does not|unable to)\b.{0,65}\b(?:understand|know|explain|fathom|conceive)\b"
    r"|\b(?:wondered|wondering)\b.{0,30}\bwhy\b",
    re.IGNORECASE | re.DOTALL,
)


def is_causal_question(query: str) -> bool:
    return bool(_CAUSAL_QUERY.search(query))


def explanation_rewrites(query: str) -> list[str]:
    """Explore different lexical formulations without asserting lore-specific facts."""
    if not is_causal_question(query):
        return []
    subjects = [
        x for x in re.findall(r"\b[A-Z][a-zA-Z'-]{2,}\b", query)
        if x.lower() not in _QUERY_WORDS
    ]
    if not subjects:
        return []
    subject = subjects[0]
    return [
        f"{subject} reasons motives for the decision to betray or reject an allegiance",
        f"{subject} deception manipulation doubts visions and the choice to rebel",
    ]


def interleave_unique(*rankings: list[dict], limit: int) -> list[dict]:
    """Diversify candidate retrieval while retaining per-query order."""
    seen = set()
    output = []
    for rank in range(max((len(items) for items in rankings), default=0)):
        for items in rankings:
            if rank >= len(items):
                continue
            hit = items[rank]
            identity = (hit.get("source", ""), hit.get("chunk_id", hit.get("text", "")))
            if identity in seen:
                continue
            seen.add(identity)
            output.append(hit)
            if len(output) >= limit:
                return output
    return output


def rank_causal_evidence(query: str, chunks: list[dict]) -> list[dict]:
    """Prefer passages expressing reasons, not merely reporting an event.

    Lexical signals are weak heuristics, not proof of a causal claim.
    Cross-encoder reranking remains the semantic ranking stage when enabled.
    """
    if not is_causal_question(query):
        return chunks
    subjects = [
        x.lower() for x in re.findall(r"\b[A-Z][a-zA-Z'-]{2,}\b", query)
        if x.lower() not in _QUERY_WORDS
    ]
    primary = subjects[0] if subjects else ""
    ranked = []
    for position, hit in enumerate(chunks):
        body = hit.get("text", "") or ""
        in_scope = not primary or bool(re.search(r"\b" + re.escape(primary) + r"\b", body, re.I))
        cue_score = min(len(_CAUSAL_CUES.findall(body)), 4) if in_scope else 0
        unanswered = bool(_UNANSWERED.search(body))
        score = cue_score + (0.4 if in_scope else -2.0) - (1.5 if unanswered else 0)
        ranked.append((score, -position, hit))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [hit for _, __, hit in ranked]


def cap_causal_context(query: str, chunks: list[dict],
                       total_chars: int = 18000, per_chunk_chars: int = 4500) -> list[dict]:
    """Bound evidence for why/cause answers without changing battle narration."""
    if not is_causal_question(query):
        return chunks
    total_chars = max(1000, total_chars)
    per_chunk_chars = max(500, per_chunk_chars)
    remaining = total_chars
    selected = []
    for doc in chunks:
        if remaining < 500:
            break
        body = (doc.get("text") or "").strip()
        if not body:
            continue
        ceiling = min(remaining, per_chunk_chars)
        if len(body) > ceiling:
            body = body[:ceiling].rsplit(" ", 1)[0].rstrip() or body[:ceiling]
        selected.append({**doc, "text": body})
        remaining -= len(body)
    return selected


# Long-form requests require retrieval across multiple stages, not only a
# single query embedding. These probes only reuse words supplied by the user:
# no named lore event or supposed cause is injected by the retriever.
_CHRONICLE_INTENT = re.compile(
    r"\b(?:complete|full|whole|entire)\s+(?:story|chronicle|history|account)\b"
    r"|\b(?:tell|narrate|recount|chronicle|trace)\b.{0,65}\b(?:story|history|journey|fall|rise|arc)\b"
    r"|\bfrom\b.{5,150}\b(?:through|until|to)\b",
    re.IGNORECASE,
)
_INTRO_WORDS = _QUERY_WORDS | {
    "tell", "narrate", "recount", "chronicle", "trace", "describe",
    "give", "explain", "complete", "full", "whole", "entire",
}
_FACET_STOP = {
    "his", "her", "their", "its", "from", "through", "until", "into",
    "with", "the", "and", "then", "after", "before", "over", "onto",
    "all", "some", "story", "complete", "event", "events", "eventual",
    "growing", "later", "early", "final", "finally", "entire", "on",
    "of", "to", "a", "an", "in", "at", "for", "by", "or",
}


def is_chronicle_request(query: str) -> bool:
    """Identify requests for an arc across time, distinct from single scenes."""
    return bool(_CHRONICLE_INTENT.search(query))


def chronicle_probes(query: str) -> list[str]:
    """Build up to three chronology facets using only the user's own words."""
    if not is_chronicle_request(query):
        return []
    names = [
        word for word in re.findall(r"\b[A-Z][a-zA-Z'-]{2,}\b", query)
        if word.lower() not in _INTRO_WORDS
    ]
    subject = names[0] if names else ""
    timeline = re.search(
        r"\bfrom\s+(.+?)\s+(?:through|until|to)\s+(.+?)(?:[.!?]|$)",
        query, flags=re.IGNORECASE,
    )
    if timeline:
        first, last = timeline.groups()
        stages = re.split(r"\s+and\s+(?=(?:the|his|her|their|its)\s+)", first)
        stages.append(last)
    else:
        # These are neutral evidence probes, not claims about what happened.
        stages = ["motives and background", "turning point", "outcome and consequences"]

    output = []
    for stage in stages:
        words = [
            w for w in re.findall(r"[a-zA-Z'-]+", stage)
            if w.lower() not in {"his", "her", "their", "the", "a", "an"}
        ]
        if not words:
            continue
        probe = (subject + " " + " ".join(words)).strip()
        if probe.casefold() not in {p.casefold() for p in output}:
            output.append(probe)
        if len(output) == 3:
            break
    return output


def _chronicle_subject(query: str) -> str:
    names = [
        word for word in re.findall(r"\b[A-Z][a-zA-Z'-]{2,}\b", query)
        if word.lower() not in _INTRO_WORDS
    ]
    return names[0].lower() if names else ""


def _facet_score(probe: str, passage: dict, subject: str) -> int:
    """Demand content evidence; filenames alone cannot satisfy an event facet."""
    body = (passage.get("text") or "").lower()
    if not body:
        return 0
    terms = [
        w.lower() for w in re.findall(r"[a-zA-Z'-]+", probe)
        if w.lower() not in _FACET_STOP and w.lower() != subject
    ]
    if not terms:
        return 0
    matches = 0
    for word in terms:
        # Simple plural/inflection support; avoid matching arbitrary substrings.
        root = word[:-1] if word.endswith("s") and len(word) > 5 else word
        if re.search(r"\b" + re.escape(root) + r"[a-z]*\b", body):
            matches += 1
    if not matches:
        return 0
    subject_present = bool(subject and re.search(r"\b" + re.escape(subject) + r"\b", body))
    return matches * 2 + (3 if subject_present else 0)


def _dedupe_chronicle(hits: list[dict]) -> list[dict]:
    """Suppress near-identical passages from PDF/EPUB editions of the same text."""
    selected = []
    seen = set()
    for hit in hits:
        normalized = "".join(re.findall(r"[a-z0-9]+", (hit.get("text") or "").lower()))
        signature = normalized[:160] if len(normalized) >= 160 else normalized
        if not signature or signature in seen:
            continue
        seen.add(signature)
        selected.append(hit)
    return selected


def prioritize_chronicle_candidates(query: str, hits: list[dict], limit: int) -> list[dict]:
    """Reserve candidate capacity for requested stages before cross-encoding."""
    if not is_chronicle_request(query):
        return hits[:limit]
    probes = chronicle_probes(query)
    subject = _chronicle_subject(query)
    ordered = []
    seen = set()
    for probe in probes:
        ranked = sorted(
            enumerate(hits),
            key=lambda pair: (_facet_score(probe, pair[1], subject), -pair[0]),
            reverse=True,
        )
        added = 0
        for _, hit in ranked:
            if _facet_score(probe, hit, subject) == 0:
                break
            identity = (hit.get("source"), hit.get("chunk_id"))
            if identity in seen:
                continue
            seen.add(identity)
            ordered.append(hit)
            added += 1
            if added == 3 or len(ordered) >= limit:
                break
    for hit in hits:
        identity = (hit.get("source"), hit.get("chunk_id"))
        if identity not in seen and len(ordered) < limit:
            ordered.append(hit)
            seen.add(identity)
    return _dedupe_chronicle(ordered)[:limit]


def select_chronicle_evidence(query: str, ranked_hits: list[dict], top_k: int) -> list[dict]:
    """Guarantee grounded stage coverage when available; retain varied supporting evidence.

    Low-scoring chunks do not qualify merely for containing a location name.
    Every reserved stage hit must have a subject + facet overlap, and a
    cross-encoder logit >= -1 if a reranker was used.
    """
    docs = _dedupe_chronicle(ranked_hits)
    if not is_chronicle_request(query):
        return docs[:top_k]
    subject = _chronicle_subject(query)
    chosen = []
    used = set()
    for probe in chronicle_probes(query):
        for hit in docs:
            identity = (hit.get("source"), hit.get("chunk_id"))
            if identity in used or _facet_score(probe, hit, subject) < 5:
                continue
            if hit.get("rerank_score", 0.0) < -1.0:
                continue
            chosen.append(hit)
            used.add(identity)
            break
        if len(chosen) >= top_k:
            return chosen
    for hit in docs:
        identity = (hit.get("source"), hit.get("chunk_id"))
        if identity not in used and hit.get("rerank_score", 0.0) >= 0.0:
            chosen.append(hit)
            used.add(identity)
        if len(chosen) >= top_k:
            break
    return chosen


def cap_chronicle_context(chunks: list[dict],
                          total_chars: int = 40000,
                          per_chunk_chars: int = 7000) -> list[dict]:
    """Protect the LLM context budget while keeping longer connected scenes."""
    remaining = total_chars
    result = []
    for chunk in chunks:
        if remaining < 500:
            break
        body = (chunk.get("text") or "").strip()
        if not body:
            continue
        max_chars = min(per_chunk_chars, remaining)
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0].rstrip()
        result.append({**chunk, "text": body})
        remaining -= len(body)
    return result
