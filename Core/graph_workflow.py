"""Opt-in, source-grounded LangGraph workflow for OmnissiahCore.

No FAISS/model imports at module import time; real retriever and LLM are injected.
This deliberately does not replace /query/narrate, /query/focus or existing SSE.
"""
from __future__ import annotations

import re
from typing import Callable, TypedDict


NO_EVIDENCE = "The Archives do not clearly record enough here."
_STOP = frozenset({
    "a", "about", "an", "and", "are", "as", "at", "be", "between", "by", "can",
    "compare", "could", "describe", "did", "do", "does", "during", "explain",
    "for", "from", "give", "had", "has", "have", "how", "i", "in", "is", "it",
    "me", "of", "on", "or", "please", "say", "tell", "than", "that", "the", "their",
    "these", "they", "this", "to", "us", "versus", "vs", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "would", "you", "best",
})
_ALIAS = {
    "ferrus": "Ferrus Manus", "fulgrimm": "Fulgrim", "istvan": "Isstvan",
    "sharokin": "Sharrowkyn", "kakaponi": "Kakophoni", "slaneshi": "Slaaneshi",
    "sea of sauls": "Sea of Souls", "3rd legion": "Emperor's Children",
}


class EvidenceState(TypedDict, total=False):
    query: str
    book_filter: str | None
    source_filter: list[str] | None
    top_k: int
    candidate_pool: int | None
    stitching_window: int | None
    intent: str
    plan: list[str]
    subjects: list[str]
    attempt: int
    chunks: list[dict]
    sufficient: bool
    reason: str
    response: str
    status: str
    trace: list[dict]
    generate_enabled: bool
    verification: str


def normalize_query(question: str) -> str:
    text = re.sub(r"\s+", " ", question).strip()
    for alias, target in sorted(_ALIAS.items(), key=lambda pair: -len(pair[0])):
        # 'Ferrus Manus' must not become 'Ferrus Manus Manus'.
        suffix = r"(?!\s+Manus\b)" if alias == "ferrus" else ""
        text = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)" + suffix,
                      lambda _: target, text, flags=re.IGNORECASE)
    return text


def words(text: str) -> list[str]:
    return [s.casefold() for s in re.findall(r"[\w']+", text)
            if len(s) > 2 and s.casefold() not in _STOP]


def build_plan(query: str) -> tuple[str, list[str], list[str]]:
    normalized = normalize_query(query)
    if not normalized:
        raise ValueError("Query cannot be blank")
    compare_match = re.search(r"\b(?:vs\.?|versus)\b", normalized, re.IGNORECASE)
    if compare_match:
        left = normalized[:compare_match.start()].strip(" ?.,:;")
        right = normalized[compare_match.end():].strip(" ?.,:;")
        subjects = [left, right] if left and right else []
        intent = "comparison"
        primary = f"Compare {left} and {right}" if subjects else normalized
    elif re.search(r"\b(?:compare|difference between)\b", normalized, re.IGNORECASE):
        intent, subjects, primary = "comparison", [], normalized
    elif re.search(r"\bwhy\b", normalized, re.IGNORECASE):
        intent, subjects, primary = "explanation", [], normalized
    else:
        intent, subjects, primary = "lookup", [], normalized
    sparse = " ".join(words(normalized))
    # Deterministic, bounded variants. Avoid hallucinated lore expansions.
    variants = list(dict.fromkeys(x for x in (primary, normalized, sparse) if x))
    return intent, subjects, variants[:3]


def _allowed(chunk: dict, book_filter: str | None,
             source_filter: list[str] | None) -> bool:
    source = chunk.get("source")
    text = chunk.get("text")
    if not isinstance(source, str) or not source.strip():
        return False
    if not isinstance(text, str) or len(text.strip()) < 48:
        return False
    if book_filter and book_filter.casefold() not in source.casefold():
        return False
    if source_filter is not None and source not in source_filter:
        return False
    return True


def merge_evidence(current: list[dict], incoming: list[dict], *,
                   book_filter: str | None, source_filter: list[str] | None,
                   limit: int) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple] = set()
    for chunk in [*current, *incoming]:
        if not isinstance(chunk, dict) or not _allowed(chunk, book_filter, source_filter):
            continue
        # Store bounded snippets from the start so a repeat search doesn't
        # duplicate the same long stitched passage after budget trimming.
        item = {**chunk, "text": chunk["text"][:2400]}
        key = (item["source"], item.get("chapter"),
               re.sub(r"\s+", " ", item["text"]).strip().casefold())
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def prioritize_subjects(chunks: list[dict], subjects: list[str]) -> list[dict]:
    """For a comparison, reserve early context space for each subject."""
    preferred: list[dict] = []
    for subject in subjects:
        terms = set(words(subject))
        for chunk in chunks:
            if terms and terms.issubset(set(words(chunk["text"]))) and chunk not in preferred:
                preferred.append(chunk)
                break
    return preferred + [chunk for chunk in chunks if chunk not in preferred]


def bound_context(chunks: list[dict], *, max_chars: int = 16000,
                  per_chunk: int = 2400) -> list[dict]:
    """Limit context size without altering the original retriever records.

    This is an approximate character budget, NOT a tokenizer-accurate budget.
    The model's actual context window and prompt template still matter.
    """
    limited: list[dict] = []
    remaining = max_chars
    for chunk in chunks:
        if remaining < 48:
            break
        text = chunk.get("text", "")
        keep = min(len(text), per_chunk, remaining)
        if keep < 48:
            continue
        limited.append({**chunk, "text": text[:keep]})
        remaining -= keep
    return limited


def assess_evidence(query: str, intent: str, subjects: list[str], chunks: list[dict]) -> tuple[bool, str]:
    if not chunks:
        return False, "no_valid_source_passages"
    corpus_words = set(words(" ".join(c["text"] for c in chunks)))
    if intent == "comparison" and len(subjects) == 2:
        for subject in subjects:
            terms = set(words(subject))
            if not terms or not terms.issubset(corpus_words):
                return False, "missing_comparison_subject"
        return True, "both_subjects_present"
    if intent == "explanation":
        # A name match is not evidence for WHY something happened. Require
        # both a query anchor and an explanatory signal in the SAME passage.
        # This remains a lexical prefilter, NOT a causal fact verifier.
        causal = re.compile(
            r"\b(?:because|caused?|causing|convinced?|manipulat(?:ed|ion)|"
            r"corrupt(?:ed|ion)|motiv(?:ated|ation|e)|ambition|resentment|"
            r"fear(?:ed|ing|s)?|distrust|deceiv(?:ed|ing)|betray(?:al|ed)?|"
            r"rebel(?:lion|led)?|tempt(?:ed|ation)|influenc(?:ed|e)|"
            r"prompt(?:ed)?|dr(?:ove|iven)|reason|vision(?:s)?|"
            r"led\s+to|due\s+to|as\s+a\s+result)\b",
            re.IGNORECASE,
        )
        anchors = set(words(normalize_query(query)))
        for chunk in chunks:
            passage = chunk.get("text", "")
            if causal.search(passage) and anchors.intersection(words(passage)):
                return True, "explanatory_language_present_unverified"
        return False, "no_explanatory_evidence"
    # Lexical gate is deliberately modest: a passage can support a concept
    # without repeating every word of the question. It is NOT fact checking.
    anchors = set(words(normalize_query(query)))
    if anchors and not anchors.intersection(corpus_words):
        return False, "query_has_no_lexical_grounding"
    return True, "lexically_grounded_passages"


class EvidenceGraph:
    """Runs a bounded plan -> retrieve -> grade -> retry -> generate workflow.

    retrieve: accepts (query, **search_kwargs), returns list[dict] with source/text.
    generate: accepts (original_question, supported_chunks, intent), returns text.
    The original question and metadata filters are never rewritten for generation.
    """

    def __init__(self, retrieve: Callable, generate: Callable, *,
                 verify: Callable | None = None, max_attempts: int = 3):
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        from langgraph.graph import START, END, StateGraph

        self._retrieve = retrieve
        self._generate = generate
        self._verify = verify
        self.max_attempts = max_attempts
        builder = StateGraph(EvidenceState)
        builder.add_node("plan", self._plan)
        builder.add_node("retrieve", self._search)
        builder.add_node("grade", self._grade)
        builder.add_node("generate", self._answer)
        builder.add_node("verify", self._validate)
        builder.add_node("incomplete", self._incomplete)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges("grade", self._decision,
                                      {"retry": "retrieve", "answer": "generate", "stop": "incomplete"})
        builder.add_edge("generate", "verify")
        builder.add_edge("verify", END)
        builder.add_edge("incomplete", END)
        self.graph = builder.compile()

    def _plan(self, state: EvidenceState) -> dict:
        intent, subjects, variants = build_plan(state["query"])
        return {"intent": intent, "subjects": subjects, "plan": variants,
                "attempt": 0, "chunks": [], "trace": [], "status": "searching",
                "verification": "not_run"}

    def _search(self, state: EvidenceState) -> dict:
        attempt = state["attempt"]
        variant = state["plan"][min(attempt, len(state["plan"]) - 1)]
        kwargs = {"top_k": state.get("top_k", 8),
                  "book_filter": state.get("book_filter"),
                  "source_filter": state.get("source_filter"),
                  "candidate_pool": state.get("candidate_pool"),
                  "stitching_window": state.get("stitching_window")}
        try:
            incoming = self._retrieve(variant, **kwargs)
            if not isinstance(incoming, list):
                raise TypeError("Retriever must return a list of passages")
            chunks = merge_evidence(state["chunks"], incoming,
                                    book_filter=kwargs["book_filter"],
                                    source_filter=kwargs["source_filter"],
                                    limit=max(1, kwargs["top_k"]) * 2 + len(incoming))
            # Reconsider the combined evidence BEFORE applying the character budget.
            # Otherwise a full first search permanently excludes recovered passages.
            anchors = set(words(normalize_query(state["query"])))
            chunks.sort(key=lambda chunk: (
                assess_evidence(state["query"], state["intent"], [], [chunk])[0],
                len(anchors.intersection(words(chunk["text"]))),
            ), reverse=True)
            if state.get("subjects"):
                chunks = prioritize_subjects(chunks, state["subjects"])
            chunks = bound_context(chunks)
            trace = [*state["trace"], {"step": "retrieval", "attempt": attempt + 1,
                                        "query": variant, "hits": len(incoming),
                                        "kept": len(chunks)}]
            return {"attempt": attempt + 1, "chunks": chunks, "trace": trace}
        except Exception as exc:
            return {"attempt": attempt + 1, "status": "retrieval_failed",
                    "reason": type(exc).__name__, "trace": [*state["trace"],
                    {"step": "retrieval", "attempt": attempt + 1, "error": type(exc).__name__}]}

    def _grade(self, state: EvidenceState) -> dict:
        if state.get("status") == "retrieval_failed":
            return {"sufficient": False}
        sufficient, reason = assess_evidence(state["query"], state["intent"],
                                              state["subjects"], state["chunks"])
        return {"sufficient": sufficient, "reason": reason,
                "trace": [*state["trace"], {"step": "grade", "sufficient": sufficient,
                                              "reason": reason}]}

    def _decision(self, state: EvidenceState) -> str:
        if state.get("status") == "retrieval_failed":
            return "stop"
        if state["sufficient"]:
            return "answer"
        if state["attempt"] < min(self.max_attempts, len(state["plan"])):
            return "retry"
        return "stop"

    def _answer(self, state: EvidenceState) -> dict:
        if not state.get("generate_enabled", True):
            return {"status": "inspection_complete", "response": "",
                    "trace": [*state["trace"], {"step": "generation", "status": "skipped"}]}
        try:
            response = self._generate(state["query"], list(state["chunks"]), state["intent"])
            if not isinstance(response, str) or not response.strip() or "[ERROR]" in response:
                raise ValueError("LLM did not produce a successful answer")
            return {"status": "ok", "response": response.strip(),
                    "trace": [*state["trace"], {"step": "generation", "status": "ok"}]}
        except Exception as exc:
            return {"status": "generation_failed", "response": "Generation failed; retry later.",
                    "reason": type(exc).__name__, "trace": [*state["trace"],
                    {"step": "generation", "status": "failed", "error": type(exc).__name__}]}

    def _validate(self, state: EvidenceState) -> dict:
        # Do not confuse retrieval lexical overlap with fact-level verification.
        # The optional verifier is independent of generation and fails closed.
        if state.get("status") != "ok" or self._verify is None:
            return {"verification": "not_run"}
        try:
            verdict = self._verify(state["query"], list(state["chunks"]),
                                   state["response"], state["intent"])
            if verdict is True:
                return {"verification": "passed", "trace": [*state["trace"],
                        {"step": "verification", "status": "passed"}]}
            return {"verification": "rejected", "status": "verification_failed",
                    "response": NO_EVIDENCE, "trace": [*state["trace"],
                    {"step": "verification", "status": "rejected"}]}
        except Exception as exc:
            return {"verification": "error", "status": "verification_failed",
                    "response": NO_EVIDENCE, "reason": type(exc).__name__,
                    "trace": [*state["trace"],
                    {"step": "verification", "status": "error", "error": type(exc).__name__}]}

    @staticmethod
    def _incomplete(state: EvidenceState) -> dict:
        if state.get("status") == "retrieval_failed":
            return {"response": "Retrieval unavailable; retry later."}
        return {"status": "insufficient_evidence", "response": NO_EVIDENCE}

    def run(self, query: str, *, book_filter: str | None = None,
            source_filter: list[str] | None = None, top_k: int = 8,
            candidate_pool: int | None = None, stitching_window: int | None = None,
            generate_enabled: bool = True) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be 1..20")
        state = self.graph.invoke({"query": query.strip(), "book_filter": book_filter,
                                   "source_filter": source_filter, "top_k": top_k,
                                   "candidate_pool": candidate_pool,
                                   "stitching_window": stitching_window,
                                   "generate_enabled": generate_enabled},
                                  config={"recursion_limit": 20})
        return {"query": query, "response": state["response"],
                "status": state["status"], "verification": state.get("verification", "not_run"),
                "attempts": state["attempt"],
                "intent": state["intent"], "chunks_used": len(state["chunks"]),
                "sources": [{"source": c["source"], "chapter": c.get("chapter", "?"),
                             "stitch_range": c.get("stitch_range", "")}
                            for c in state["chunks"]],
                "trace": state["trace"], "evidence_grade": state.get("reason", "unknown"),
                "passages": ([{"source": c["source"], "chapter": c.get("chapter", "?"),
                              "excerpt": c["text"][:500]} for c in state["chunks"]]
                             if not generate_enabled else [])}
