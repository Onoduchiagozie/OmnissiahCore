"""Deterministic, source-independent NLP for lore retrieval planning.

No external model calls, paid APIs or baked-in lore facts.
Aliases are optional and must come from a curated/index-derived lexicon.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

_SPACE = re.compile(r"\s+")
_WHY = re.compile(
    r"^why\s+(?:did|does|do|was|were|is|are)\s+"
    r"(?P<subject>.+?)\s+"
    r"(?P<action>turn|become|became|betray|betrayed|join|joined|"
    r"fall|fell|attack|attacked|leave|left|reject|rejected|"
    r"kill|killed|fight|fought|abandon|abandoned)\b(?P<rest>.*)$",
    re.I,
)
_CAUSAL = re.compile(r"^(?:why\b|what\s+(?:caused|made|led)\b)", re.I)
_COMPARISON = re.compile(r"\b(?:versus|vs\.?|compare|difference between)\b", re.I)
_SCENE = re.compile(r"\b(?:battle|fight|duel|confront|attacked|killed|rescued|saved|scene)\b", re.I)
_NON_WORD = re.compile(r"[^a-z0-9]+", re.I)
_PLANNING_STOP = frozenset("the and for with from this that when while what where why did does was were are into about which who how their they his her its trying want tell describe explain recount scene events event happened".split())


@dataclass(frozen=True)
class QueryPlan:
    original: str
    normalized: str
    intent: str
    subject: str
    action: str
    variants: tuple[str, ...]


def _dedupe(queries: list[str], limit: int) -> tuple[str, ...]:
    result, seen = [], set()
    for query in queries:
        clean = _SPACE.sub(" ", query).strip()
        key = _NON_WORD.sub(" ", clean.casefold()).strip()
        if key and key not in seen:
            result.append(clean)
            seen.add(key)
        if len(result) >= limit:
            break
    return tuple(result)


def interpret(query: str, aliases: Mapping[str, str] | None = None,
              max_queries: int = 4) -> QueryPlan:
    """Conservative interpretation; always preserves the actual user question."""
    original = _SPACE.sub(" ", query).strip()
    normalized = original
    for alias, target in (aliases or {}).items():
        if alias and target:
            normalized = re.sub(r"\b" + re.escape(alias) + r"\b",
                                lambda _: target, normalized, flags=re.I)
    normalized = normalized.strip()
    core = normalized.rstrip(" ?!.")

    if _CAUSAL.search(core):
        intent = "causal"
    elif _COMPARISON.search(core):
        intent = "comparison"
    elif _SCENE.search(core):
        intent = "scene"
    else:
        intent = "lookup"

    match = _WHY.match(core) if intent == "causal" else None
    subject = match.group("subject").strip() if match else ""
    action = (match.group("action") + match.group("rest")).strip() if match else ""
    target = core or normalized
    queries = [original]
    if normalized != original:
        queries.append(normalized)

    # Search the user's meaningful words independently of question phrasing.
    # Never invent names, books or events while decomposing a request.
    anchors = [word for word in re.findall(r"[A-Za-z0-9']+", core)
               if len(word) >= 3 and word.casefold() not in _PLANNING_STOP]
    if len(anchors) >= 3:
        queries.append(" ".join(anchors[:8]))

    if intent == "causal":
        if subject:
            queries.extend([
                f"{subject} motivations and causes for {action}",
                f"events and influences leading to {subject} {action}",
                f"{subject} reasons and circumstances behind {action}",
            ])
            if re.search(r"\btraitor|betray|rebellion\b", action, re.I):
                queries.append(f"{subject} betrayal rebellion motivations")
        else:
            queries.extend([
                f"causes and motivations underlying {target}",
                f"events and circumstances leading to {target}",
                f"explanation and reasons for {target}",
            ])
    elif intent == "comparison":
        queries.extend([
            f"relationship and differences concerning {target}",
            f"interactions and conflict involving {target}",
            f"history and context of {target}",
        ])
    elif intent == "scene":
        queries.extend([
            f"events and participants in {target}",
            f"confrontation and sequence of actions {target}",
            f"account of {target}",
        ])
    else:
        queries.extend([
            f"history and context of {target}",
            f"details and explanation of {target}",
            f"events connected to {target}",
        ])
    return QueryPlan(original, normalized, intent, subject, action,
                     _dedupe(queries, max(1, min(max_queries, 6))))


def recovery_queries(plan: QueryPlan, gaps: list[str], limit: int = 2) -> tuple[str, ...]:
    """Second pass searches for missing relationships without inventing lore."""
    base = plan.normalized.rstrip(" ?!.")
    if plan.intent == "causal":
        subject = plan.subject or base
        action = plan.action or base
        queries = [
            f"what happened before {subject} {action}",
            f"the decisions events and influences that changed {subject}",
            f"{subject} turning points leading to {action}",
        ]
    elif plan.intent == "scene":
        queries = [
            f"chronology of the encounter {base}",
            f"participants actions and outcome of {base}",
        ]
    else:
        queries = [
            f"earlier background and later consequences of {base}",
            f"other accounts and references to {base}",
        ]
    existing = {_NON_WORD.sub(" ", q.casefold()).strip() for q in plan.variants}
    remaining = [q for q in queries if _NON_WORD.sub(" ", q.casefold()).strip() not in existing]
    return _dedupe(remaining, max(1, min(limit, 4)))
