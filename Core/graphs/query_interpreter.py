"""Deterministic, local-first question interpretation (no network or model calls).

This stage is deliberately NOT a full NER model or a source of lore facts.
It identifies intent, optionally normalizes known aliases, and extracts
candidate anchor words. Retrieval and evidence checks must validate meaning.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from collections.abc import Mapping

_STOPWORDS = frozenset(
    "a an and are as at be been by can did do does for from had has have "
    "how in into is it its me of on or our the their them there these this "
    "those to was were what when where which who why with would you your "
    "tell explain describe show story scene about happened happen".split()
)
_CAUSAL = re.compile(
    r"^\\s*(?:why\\b|what\\s+(?:caused|made|led)\\b|"
    r"how\\s+(?:did|does|do)\\b.*\\b(?:become|turn|fall|betray)\\b)",
    re.I,
)
_COMPARISON = re.compile(r"\\b(?:versus|vs\\.?|compare|difference between)\\b", re.I)
_SCENE = re.compile(r"\\b(?:duel|fight|battle|attack|confront|narrate|scene)\\b", re.I)

@dataclass(frozen=True)
class QueryInterpretation:
    original: str
    normalized: str
    intent: str
    anchors: tuple[str, ...]


def interpret(query: str, aliases: Mapping[str, str] | None = None) -> QueryInterpretation:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a nonempty string.")
    cleaned = re.sub(r"\\s+", " ", unicodedata.normalize("NFKC", query)).strip()
    normalized = cleaned
    # Alias mapping is supplied by a trusted corpus lexicon, never LLM guesses.
    for alias, canonical in sorted((aliases or {}).items(), key=lambda kv: -len(kv[0])):
        if not alias or not canonical:
            continue
        normalized = re.sub(
            r"(?<!\\w)" + re.escape(alias) + r"(?!\\w)",
            lambda _: canonical,
            normalized,
            flags=re.I,
        )
    if _CAUSAL.search(normalized):
        intent = "explanation"
    elif _COMPARISON.search(normalized):
        intent = "comparison"
    elif _SCENE.search(normalized):
        intent = "scene"
    else:
        intent = "lookup"
    tokens = re.findall(r"[\\w']+", normalized.casefold())
    anchors = tuple(dict.fromkeys(
        t for t in tokens if len(t) > 2 and t not in _STOPWORDS
    ))[:12]
    return QueryInterpretation(cleaned, normalized, intent, anchors)
