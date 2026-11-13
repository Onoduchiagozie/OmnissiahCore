"""Bounded LangGraph evidence retrieval for OmnissiahCore.

Opt-in only: no production runtime imports or service behaviour change until
retrieval_cfg['langgraph_retrieval'] is explicitly enabled.
The graph does NOT call an LLM. Existing agent/providers compose the answer
after retrieval, retaining the established API and SSE contracts.

Nodes: interpret -> plan -> retrieve -> rank -> stitch -> assess
                                          ^                 |
                                          |-- recover <-----|
All results are evidence CANDIDATES; deterministic coverage is not proof that
an answer is true. Human-verified benchmark remains the deployment gate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from heapq import heappush, heapreplace
import re
from typing import Any, TypedDict

from Core.retrieval_nlp import QueryPlan, interpret, recovery_queries
from Core.retrieval_quality import is_archive_boilerplate

_QUERY_STOP = frozenset("the and for with from this that when while what where why did does was were are into about which who how their they his her its trying want tell describe explain recount scene events event happened".split())
_TOKEN = re.compile(r"[a-z0-9]{3,}")


def _filtered_candidates(metadata, query, book_filter, source_filter, limit):
    """Find lexical anchors inside explicitly selected sources on retry.

    Only the selected book(s) are text-scanned. Scores are retrieval hints,
    not proof of relevance. The metadata list itself is still traversed.
    """
    terms = set(_TOKEN.findall(query.casefold())) - _QUERY_STOP
    if not terms or not (book_filter or source_filter):
        return []
    allowed = set(source_filter or [])
    book = (book_filter or "").casefold()
    matches = []
    minimum = min(2, len(terms))
    for index, item in enumerate(metadata):
        source = str(item.get("source", ""))
        if (book and book not in source.casefold()) or (allowed and source not in allowed):
            continue
        body = str(item.get("text") or "")
        if not body or is_archive_boilerplate(body.casefold()):
            continue
        found = terms.intersection(_TOKEN.findall(body.casefold()))
        if len(found) >= minimum:
            row = (len(found), -index, item)
            if len(matches) < limit:
                heappush(matches, row)
            elif row[:2] > matches[0][:2]:
                heapreplace(matches, row)
    return [dict(item) for _, _, item in sorted(matches, key=lambda x: (x[0], x[1]), reverse=True)]


class EvidenceState(TypedDict, total=False):
    original_query: str
    plan: QueryPlan
    search_queries: list[str]
    used_queries: list[str]
    candidates: list[dict]
    ranked: list[dict]
    selected_evidence: list[dict]
    evidence_gaps: list[str]
    evidence_coverage: str
    pass_number: int
    max_passes: int
    candidate_pool: int
    top_k: int
    stitching_window: int
    book_filter: str | None
    source_filter: list[str] | None
    diagnostic: list[dict]


def _doc_id(doc: dict) -> tuple[str, str]:
    source = str(doc.get("source", ""))
    chunk_id = doc.get("chunk_id")
    if chunk_id is not None:
        return source, str(chunk_id)
    return source, sha256(str(doc.get("text", "")).encode("utf-8", "replace")).hexdigest()


def _word_present(word: str, text: str) -> bool:
    if not word:
        return False
    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.I) is not None


def _causal_leads(subject: str, passages: list[dict]) -> tuple[str, ...]:
    """Find recurring names near the subject in the strongest passages."""
    if not subject:
        return ()
    names = Counter()
    excluded = set(subject.casefold().split()) | _QUERY_STOP
    for passage in passages[:2]:
        body = " ".join(str(passage.get("text", "")).split())
        for match in re.finditer(r"\b" + re.escape(subject) + r"\b", body, re.I):
            window = body[max(0, match.start() - 180):match.end() + 180]
            for name in re.findall(r"(?<![.!?]\s)\b[A-Z][a-z]{3,}\b", window):
                if name.casefold() not in excluded:
                    names[name] += 1
        if names:
            break
    return tuple(name for name, count in names.most_common(1) if count >= 2)


def _covered(plan: QueryPlan, passages: list[dict]) -> tuple[bool, list[str]]:
    """A conservative search-retry heuristic, NOT a factual entailment model."""
    if not passages:
        return False, ["no_selected_passages"]
    joined = "\n".join(str(x.get("text", "")) for x in passages)
    gaps = []
    if plan.subject and not _word_present(plan.subject, joined):
        gaps.append("missing_subject")
    if plan.intent == "causal":
        # Generic linguistic markers; no lore/event-specific facts.
        markers = re.compile(
            r"\b(?:because|reason|cause|caused|led\s+to|"
            r"convinced|persuaded|manipulated|influenced|"
            r"choice|chose|decision|motivated|resentment)\b",
            re.I,
        )
        aligned = [
            p for p in passages
            if (not plan.subject or _word_present(plan.subject, str(p.get("text", ""))))
            and markers.search(str(p.get("text", "")))
        ]
        if not aligned:
            gaps.append("missing_causal_context")
        if plan.subject and plan.action:
            action_terms = set(_TOKEN.findall(plan.action.casefold())) - _QUERY_STOP
            betrayal = bool(re.fullmatch(r"turn(?:ed)? traitor", plan.action, re.I))
            def connects_outcome(sentence: str) -> bool:
                if not (_word_present(plan.subject, sentence) and markers.search(sentence)):
                    return False
                if betrayal:
                    # The subject must be the actor. Being betrayed and
                    # describing what followed a betrayal do not explain it.
                    actor = re.escape(plan.subject)
                    return bool(re.search(
                        rf"\b{actor}\s+(?:(?:had|then|finally)\s+)*"
                        r"(?:betrayed|rebelled|turned\s+traitor|chose\s+rebellion)\b",
                        sentence, re.I,
                    ))
                return bool(action_terms) and all(_word_present(term, sentence) for term in action_terms)

            if not any(connects_outcome(sentence)
                       for passage in aligned
                       for sentence in re.split(r"[.!?]+", str(passage.get("text", "")))):
                gaps.append("missing_outcome_link")
    elif plan.intent in {"scene", "lookup", "comparison"}:
        words = [
            w for w in re.findall(r"[a-z]{4,}", plan.normalized.lower())
            if w not in _QUERY_STOP | {"using", "scene", "battle", "fight", "entire"}
        ]
        distinct = list(dict.fromkeys(words))
        overlap = max(
            (sum(_word_present(w, str(p.get("text", ""))) for w in distinct)
             for p in passages),
            default=0,
        )
        if distinct and overlap < min(2, len(distinct)):
            gaps.append("missing_scene_alignment")
            gaps.extend(f"missing_term:{word}" for word in distinct
                        if not _word_present(word, joined))
    return not gaps, gaps


class RetrievalGraph:
    """Stateful control flow with existing in-process retrieval model objects."""

    def __init__(self, retriever, *, aliases: dict[str, str] | None = None):
        self.retriever = retriever
        self.aliases = aliases or {}
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph retrieval is opt-in. Install the pinned optional "
                "dependency from requirements-langgraph.txt in a test venv."
            ) from exc

        graph = StateGraph(EvidenceState)
        graph.add_node("interpret", self._interpret)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rank_raw", self._rank_raw)
        graph.add_node("stitch", self._stitch)
        graph.add_node("assess", self._assess)
        graph.add_node("recover", self._recover)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "interpret")
        graph.add_edge("interpret", "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "rank_raw")
        graph.add_edge("rank_raw", "stitch")
        graph.add_edge("stitch", "assess")
        graph.add_conditional_edges(
            "assess",
            self._route,
            {"recover": "recover", "finish": "finish"},
        )
        graph.add_edge("recover", "retrieve")
        graph.add_edge("finish", END)
        self.graph = graph.compile()

    @staticmethod
    def _log(state: EvidenceState, step: str, **details) -> list[dict]:
        # Keep a simple, bounded per-request trace. No secrets or model calls.
        return (state.get("diagnostic", []) + [{"step": step, **details}])[-24:]

    def _interpret(self, state: EvidenceState) -> dict:
        plan = interpret(state["original_query"], self.aliases)
        return {"plan": plan, "diagnostic": self._log(
            state, "interpret", intent=plan.intent, subject=plan.subject,
        )}

    def _plan(self, state: EvidenceState) -> dict:
        queries = list(state["plan"].variants)
        return {
            "search_queries": queries, "used_queries": [],
            "pass_number": 0, "candidates": [],
            "diagnostic": self._log(state, "plan", variant_count=len(queries)),
        }

    def _retrieve(self, state: EvidenceState) -> dict:
        by_id: dict[tuple, dict] = {
            _doc_id(p): dict(p) for p in state.get("candidates", [])
        }
        fused = defaultdict(float)
        for key, candidate in by_id.items():
            fused[key] = float(candidate.get("discovery_weight", 0.0))
        queries = state["search_queries"]
        pool = max(4, min(int(state["candidate_pool"]), 64))
        sources = set(state.get("source_filter") or [])
        book_filter = (state.get("book_filter") or "").casefold()

        for q_index, query in enumerate(queries):
            dense = self.retriever._faiss_search(self.retriever._enrich_query(query), k=pool)
            sparse = (self.retriever._bm25_search(query, k=pool)
                      if self.retriever.bm25 else [])
            combined = self.retriever._rrf_merge(dense, sparse, k=pool)
            for rank, doc in enumerate(combined):
                source = str(doc.get("source", ""))
                if book_filter and book_filter not in source.casefold():
                    continue
                if sources and source not in sources:
                    continue
                key = _doc_id(doc)
                if key not in by_id:
                    by_id[key] = dict(doc)
                fused[key] += (1.35 if q_index == 0 and state["pass_number"] == 0
                               else 1.0) / (60.0 + rank + 1)

        local_hits = []
        if state["pass_number"] > 0 and (book_filter or sources):
            local_hits = _filtered_candidates(
                getattr(self.retriever, "metadata", []), state["original_query"],
                state.get("book_filter"), state.get("source_filter"), pool,
            )
            for rank, doc in enumerate(local_hits):
                key = _doc_id(doc)
                if key not in by_id:
                    by_id[key] = dict(doc)
                fused[key] += 1.0 / (61.0 + rank)

        ranked_ids = sorted(by_id, key=lambda k: fused[k], reverse=True)[:96]
        candidates = []
        for key in ranked_ids:
            chunk = dict(by_id[key])
            chunk["discovery_weight"] = fused[key]
            candidates.append(chunk)
        return {
            "candidates": candidates,
            "used_queries": state.get("used_queries", []) + queries,
            "diagnostic": self._log(
                state, "retrieve", pass_number=state["pass_number"],
                query_count=len(queries), candidate_count=len(candidates),
                source_fallback_count=len(local_hits),
            ),
        }

    def _rank_raw(self, state: EvidenceState) -> dict:
        docs = [dict(c) for c in state.get("candidates", [])[:64]]
        if not docs:
            return {"ranked": [], "diagnostic": self._log(state, "rank_raw", count=0)}
        discovery_order = sorted(
            range(len(docs)), key=lambda i: docs[i]["discovery_weight"], reverse=True
        )
        discovery_rank = {i: rank for rank, i in enumerate(discovery_order)}
        if self.retriever.reranker is not None:
            scores = self.retriever.reranker.predict([
                [state["original_query"], d.get("text", "")] for d in docs
            ])
            for doc, score in zip(docs, scores):
                doc["raw_rerank_logit"] = float(score)
            rerank_order = sorted(
                range(len(docs)), key=lambda i: docs[i].get("raw_rerank_logit", float("-inf")),
                reverse=True,
            )
            rerank_rank = {i: rank for rank, i in enumerate(rerank_order)}
            for i, doc in enumerate(docs):
                # Reciprocal rank fusion of semantic ranking and search ranks.
                # A negative cross-encoder logit is NOT grounds to declare missing evidence.
                doc["combined_evidence_rank"] = (
                    1 / (60 + discovery_rank[i] + 1)
                    + 1 / (60 + rerank_rank[i] + 1)
                )
        else:
            for i, doc in enumerate(docs):
                doc["combined_evidence_rank"] = 1 / (60 + discovery_rank[i] + 1)
        docs.sort(key=lambda d: d["combined_evidence_rank"], reverse=True)
        return {
            "ranked": docs,
            "diagnostic": self._log(
                state, "rank_raw", count=len(docs),
                cross_encoder=self.retriever.reranker is not None,
            ),
        }

    def _stitch(self, state: EvidenceState) -> dict:
        selected_count = max(1, min(int(state["top_k"]), 20))
        raw = state.get("ranked", [])
        # Rank short raw chunks first, expand same-source neighbours afterwards.
        stitched = self.retriever._stitch_chunks(
            raw[:min(len(raw), selected_count * 3)],
            window=max(0, min(int(state["stitching_window"]), 6)),
        )
        if state["plan"].intent == "causal":
            # Keep the ranking within each group, while giving passages that
            # connect the subject and outcome first claim on the context budget.
            stitched = sorted(stitched, key=lambda doc: _covered(state["plan"], [doc])[0],
                              reverse=True)
        selected = [dict(d) for d in stitched[:selected_count]]
        return {
            "selected_evidence": selected,
            "diagnostic": self._log(state, "stitch", result_count=len(selected)),
        }

    def _assess(self, state: EvidenceState) -> dict:
        covered, gaps = _covered(state["plan"], state.get("selected_evidence", []))
        return {
            "evidence_gaps": gaps,
            "evidence_coverage": "heuristic_pass" if covered else "incomplete",
            "diagnostic": self._log(state, "assess", coverage=covered, gaps=gaps),
        }

    def _route(self, state: EvidenceState) -> str:
        if (state.get("evidence_gaps")
                and int(state.get("pass_number", 0)) + 1 < state["max_passes"]):
            return "recover"
        return "finish"

    def _recover(self, state: EvidenceState) -> dict:
        next_queries = list(recovery_queries(
            state["plan"], state.get("evidence_gaps", []),
            limit=2,
            leads=_causal_leads(state["plan"].subject, state.get("selected_evidence", []))
            if "missing_outcome_link" in state.get("evidence_gaps", []) else (),
        ))
        existing = set(q.casefold() for q in state.get("used_queries", []))
        next_queries = [q for q in next_queries if q.casefold() not in existing]
        return {
            "search_queries": next_queries,
            "pass_number": state.get("pass_number", 0) + 1,
            "diagnostic": self._log(
                state, "recover", new_query_count=len(next_queries), queries=next_queries,
            ),
        }

    def _finish(self, state: EvidenceState) -> dict:
        return {
            "diagnostic": self._log(
                state, "finish",
                selected_count=len(state.get("selected_evidence", [])),
                evidence_coverage=state.get("evidence_coverage", "unknown"),
                # Human source verification is still required.
                factual_accuracy_verified=False,
            )
        }

    def run(self, query: str, *, top_k: int = 6, candidate_pool: int = 20,
            stitching_window: int = 2, book_filter: str | None = None,
            source_filter: list[str] | None = None,
            max_passes: int = 2) -> EvidenceState:
        if not query or not query.strip():
            return {"original_query": query, "selected_evidence": [],
                    "evidence_gaps": ["empty_question"], "diagnostic": []}
        return self.graph.invoke({
            "original_query": query,
            "top_k": max(1, min(top_k, 20)),
            "candidate_pool": max(4, min(candidate_pool, 64)),
            "stitching_window": max(0, min(stitching_window, 6)),
            "max_passes": max(1, min(max_passes, 2)),
            "book_filter": book_filter,
            "source_filter": source_filter,
            "diagnostic": [],
        }, config={"recursion_limit": 28})
