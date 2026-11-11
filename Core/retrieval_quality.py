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


def compact_chronicle_probes(query: str) -> list[str]:
    """Concentrate corpus lookup on the actual named subject and stage.

    'Horus events on Davin' -> 'Horus Davin'. The query itself supplies
    all names and locations. These are retrieval probes, not claims.
    """
    subject = _chronicle_subject(query)
    output = []
    for probe in chronicle_probes(query):
        terms = [
            w for w in re.findall(r"[a-zA-Z'-]+", probe)
            if w.lower() not in _FACET_STOP and w.lower() != subject
        ]
        if not terms:
            continue
        compact = " ".join(([subject] if subject else []) + terms[:3])
        if compact.casefold() != probe.casefold() and compact not in output:
            output.append(compact)
    return output


_BOILERPLATE = re.compile(
    r"more (?:horus heresy|warhammer) (?:books|stories) from blacklibrary"
    r"|blacklibrary\.com|\\bisbn\\b"
    r"|all rights reserved|copyright \\d{4}",
    re.IGNORECASE,
)


def is_archive_boilerplate(text: str) -> bool:
    """Reject marketing/index pages, which are not scene evidence."""
    return bool(_BOILERPLATE.search(text or ""))


def _chronicle_subject(query: str) -> str:
    names = [
        word for word in re.findall(r"\b[A-Z][a-zA-Z'-]{2,}\b", query)
        if word.lower() not in _INTRO_WORDS
    ]
    return names[0].lower() if names else ""


def _facet_score(probe: str, passage: dict, subject: str) -> int:
    """Demand content evidence; filenames alone cannot satisfy an event facet."""
    body = (passage.get("text") or "").lower()
    if not body or is_archive_boilerplate(body):
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
    # Give local event evidence more weight than an isolated reference
    # to the subject hundreds of words away from a place/event.
    proximity = 0
    if subject_present:
        for word in terms:
            root = word[:-1] if word.endswith("s") and len(word) > 5 else word
            if re.search(
                r"\b" + re.escape(subject) + r"\b.{0,240}\b" + re.escape(root)
                + r"[a-z]*\b|\b" + re.escape(root)
                + r"[a-z]*\b.{0,240}\b" + re.escape(subject) + r"\b",
                body, flags=re.DOTALL,
            ):
                proximity = 3
                break
    return matches * 2 + (3 if subject_present else 0) + proximity


def chronicle_facet_score(probe: str, passage: dict, query: str) -> int:
    """Public scoring function shared by stage selection and cross-encoder."""
    return _facet_score(probe, passage, _chronicle_subject(query))


def _dedupe_chronicle(hits: list[dict]) -> list[dict]:
    """Suppress near-identical passages from PDF/EPUB editions of the same text."""
    selected = []
    seen = set()
    for hit in hits:
        if is_archive_boilerplate(hit.get("text", "")):
            continue
        normalized = "".join(re.findall(r"[a-z0-9]+", (hit.get("text") or "").lower()))
        signature = normalized[:160] if len(normalized) >= 160 else normalized
        if not signature or signature in seen:
            continue
        seen.add(signature)
        selected.append(hit)
    return selected


def prioritize_chronicle_candidates(query: str, hits: list[dict], limit: int) -> list[dict]:
    """Fairly reserve candidate capacity for each requested stage."""
    if not is_chronicle_request(query):
        return hits[:limit]
    hits = _dedupe_chronicle(hits)
    probes = chronicle_probes(query)
    subject = _chronicle_subject(query)
    ranked_facets = []
    for probe in probes:
        matches = [
            (idx, hit) for idx, hit in enumerate(hits)
            if _facet_score(probe, hit, subject) >= 3
        ]
        matches.sort(
            key=lambda pair: (_facet_score(probe, pair[1], subject), -pair[0]),
            reverse=True,
        )
        ranked_facets.append([hit for _, hit in matches])

    chosen = []
    used = set()
    # Round-robin: each timeline stage gets first refusal before an
    # earlier stage can consume several candidate slots.
    # Keep several candidates per stage for semantic reranking. Three
    # candidates per stage were too few: tangential references displaced
    # the primary novel before the cross-encoder ever saw its passages.
    stage_capacity = min(10, max(3, limit // max(1, len(probes) * 2)))
    for _ in range(stage_capacity):
        for ranked in ranked_facets:
            for hit in ranked:
                identity = (hit.get("source"), hit.get("chunk_id"))
                if identity not in used:
                    chosen.append(hit)
                    used.add(identity)
                    break
            if len(chosen) >= limit:
                return chosen

    for hit in hits:
        identity = (hit.get("source"), hit.get("chunk_id"))
        if identity not in used:
            chosen.append(hit)
            used.add(identity)
        if len(chosen) >= limit:
            break
    return chosen

def select_chronicle_evidence(query: str, ranked_hits: list[dict],
                              top_k: int) -> list[dict]:
    """Stage-specific relevance before broad support, without forced quotas."""
    docs = _dedupe_chronicle(ranked_hits)
    if not is_chronicle_request(query):
        return docs[:top_k]
    subject = _chronicle_subject(query)
    chosen = []
    used = set()
    for probe in chronicle_probes(query):
        options = [
            hit for hit in docs
            if (hit.get("source"), hit.get("chunk_id")) not in used
            and _facet_score(probe, hit, subject) >= 5
            and (
                hit.get("facet_rerank_scores", {}).get(probe, -99) >= 0.0
                if "facet_rerank_scores" in hit else
                hit.get("rerank_score", 0.0) >= -1.0
            )
        ]
        options.sort(
            key=lambda hit: (
                hit.get("facet_rerank_scores", {}).get(
                    probe, hit.get("rerank_score", 0.0)),
                _facet_score(probe, hit, subject),
            ),
            reverse=True,
        )
        if options:
            pick = options[0]
            chosen.append(pick)
            used.add((pick.get("source"), pick.get("chunk_id")))
        if len(chosen) >= top_k:
            return chosen

    # Broader supporting passages still belong in the story, but must
    # actually contain the requested subject and pass semantic screening.
    for hit in docs:
        identity = (hit.get("source"), hit.get("chunk_id"))
        body = (hit.get("text") or "").lower()
        subject_present = not subject or bool(
            re.search(r"\b" + re.escape(subject) + r"\b", body)
        )
        if identity not in used and subject_present and hit.get("rerank_score", 0.0) >= 0.0:
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
