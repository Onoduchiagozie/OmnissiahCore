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
