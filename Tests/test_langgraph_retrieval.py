"""Tests for actual LangGraph control flow with fake indexes (no Gemini).

Run in a test environment after installing requirements-langgraph.txt:
    python -m unittest discover -s Tests -p test_langgraph_retrieval.py -v

These tests prove routing/invariants, NOT lore accuracy. Real corpus A/B
verification remains required prior to enabling the production feature.
"""
import unittest

from Core.retrieval_nlp import interpret, recovery_queries

try:
    from Core.graphs.retrieval_graph import RetrievalGraph
    import langgraph  # noqa: F401
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False


class FakeRanker:
    def predict(self, pairs):
        return [4.0 if "because" in text.lower() else -2.0 for _, text in pairs]


class FakeHybrid:
    def __init__(self, allow_recovery=True):
        self.bm25 = True
        self.reranker = FakeRanker()
        self.queries = []
        self.allow_recovery = allow_recovery

    def _enrich_query(self, query):
        return query

    def _faiss_search(self, query, k):
        self.queries.append(query)
        if ("what happened before" in query.lower()
                and self.allow_recovery):
            return [{
                "source": "False Gods", "chunk_id": 24,
                "text": "Horus was changed because his advisors manipulated his decision.",
            }]
        return [{
            "source": "Horus Rising", "chunk_id": 11,
            "text": "Horus led an expedition during the Great Crusade.",
        }]

    def _bm25_search(self, query, k):
        return []

    def _rrf_merge(self, dense, sparse, k):
        return [dict(d) for d in dense[:k]]

    def _stitch_chunks(self, candidates, window):
        return [dict(x) for x in candidates]


class NLPTests(unittest.TestCase):
    def test_long_question_produces_short_search_using_only_user_terms(self):
        plan = interpret("Magnus crashes the wards on the Imperial Palace while trying to warn the Emperor")
        self.assertIn("Magnus crashes wards Imperial Palace warn Emperor", plan.variants)
    def test_why_question_extracts_subject_without_facts(self):
        plan = interpret("why did horus turn traitor")
        self.assertEqual(plan.intent, "causal")
        self.assertEqual(plan.subject.lower(), "horus")
        self.assertEqual(plan.action.lower(), "turn traitor")
        self.assertEqual(plan.variants[0], "why did horus turn traitor")
        self.assertTrue(all("Davin" not in x and "Erebus" not in x for x in plan.variants))

    def test_alias_is_optional_and_word_bounded(self):
        plan = interpret("sharokin vs lucius", aliases={"sharokin": "Sharrowkyn"})
        self.assertIn("Sharrowkyn", plan.normalized)
        self.assertEqual(plan.original, "sharokin vs lucius")
        self.assertEqual(plan.intent, "comparison")

    def test_recovery_does_not_duplicate_original(self):
        plan = interpret("why did horus turn traitor")
        later = recovery_queries(plan, ["missing_causal_context"])
        self.assertTrue(later)
        self.assertTrue(set(q.casefold() for q in later).isdisjoint(
            set(q.casefold() for q in plan.variants)))


@unittest.skipUnless(HAS_GRAPH, "LangGraph optional dependency not installed")
class GraphTests(unittest.TestCase):
    def test_filtered_retry_searches_selected_source_after_global_miss(self):
        fake = FakeHybrid()
        fake.metadata = [
            {"source": "other.pdf", "chunk_id": 1, "text": "Magnus broke the wards in the palace."},
            {"source": "selected.pdf", "chunk_id": 2, "text": "Magnus broke the wards in the palace."},
            {"source": "selected.pdf", "chunk_id": 3, "text": "An unrelated dinner was served."},
        ]
        result = RetrievalGraph(fake).run("Magnus broke the wards", book_filter="selected.pdf")
        self.assertEqual(result["pass_number"], 1)
        self.assertEqual([p["chunk_id"] for p in result["selected_evidence"]], [2])
        retry = [step for step in result["diagnostic"]
                 if step["step"] == "retrieve" and step["pass_number"] == 1][0]
        self.assertEqual(retry["source_fallback_count"], 1)

    def test_retry_honors_exact_source_list_and_rejects_other_sources(self):
        fake = FakeHybrid()
        fake.metadata = [
            {"source": "wanted.pdf", "chunk_id": 1, "text": "Magnus broke the wards in the palace."},
            {"source": "wanted-extra.pdf", "chunk_id": 2, "text": "Magnus broke the wards in the palace."},
        ]
        result = RetrievalGraph(fake).run("Magnus broke the wards", source_filter=["wanted.pdf"])
        self.assertEqual([p["source"] for p in result["selected_evidence"]], ["wanted.pdf"])

    def test_retry_does_not_scan_metadata_without_an_explicit_filter(self):
        fake = FakeHybrid()
        fake.metadata = [{"source": "book.pdf", "chunk_id": 2,
                          "text": "Horus chose rebellion because he was manipulated."}]
        result = RetrievalGraph(fake).run("why did horus turn traitor")
        retry = [step for step in result["diagnostic"]
                 if step["step"] == "retrieve" and step["pass_number"] == 1][0]
        self.assertEqual(retry["source_fallback_count"], 0)

    def test_stitch_does_not_reintroduce_raw_overlapping_chunks(self):
        fake = FakeHybrid()
        fake._stitch_chunks = lambda candidates, window: [dict(candidates[0])]
        fake._faiss_search = lambda query, k: [
            {"source": "book.pdf", "chunk_id": i, "text": "Magnus broke the wards"}
            for i in (1, 2, 3)]
        result = RetrievalGraph(fake).run("Magnus broke the wards", top_k=3)
        self.assertEqual(len(result["selected_evidence"]), 1)

    def test_bounded_retry_recovers_causal_passage(self):
        fake = FakeHybrid()
        graph = RetrievalGraph(fake)
        state = graph.run("why did horus turn traitor", top_k=3,
                          candidate_pool=8, stitching_window=1, max_passes=2)
        self.assertEqual(state["pass_number"], 1)
        self.assertTrue(any("because" in p["text"] for p in state["selected_evidence"]))
        self.assertIn("recover", [x["step"] for x in state["diagnostic"]])
        self.assertTrue(all(x.get("factual_accuracy_verified") is False
                            for x in state["diagnostic"] if x["step"] == "finish"))

    def test_retry_budget_is_enforced(self):
        fake = FakeHybrid()
        state = RetrievalGraph(fake).run(
            "why did horus turn traitor", top_k=2, candidate_pool=8,
            max_passes=1,
        )
        self.assertEqual(state["pass_number"], 0)
        self.assertNotIn("recover", [x["step"] for x in state["diagnostic"]])

    def test_book_filter_never_leaks_unselected_book(self):
        fake = FakeHybrid()
        state = RetrievalGraph(fake).run(
            "why did horus turn traitor", book_filter="Betrayer",
            max_passes=2,
        )
        self.assertEqual(state["selected_evidence"], [])
        self.assertIn("no_selected_passages", state["evidence_gaps"])

    def test_existing_match_completes_without_recovery(self):
        fake = FakeHybrid()
        state = RetrievalGraph(fake).run(
            "who was Horus", candidate_pool=8, max_passes=2,
        )
        self.assertEqual(state["pass_number"], 0)
        self.assertEqual(len(state["selected_evidence"]), 1)


if __name__ == "__main__":
    unittest.main()
