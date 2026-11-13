"""Exercise the integrated selection methods without downloading ML models."""
import ast
import re
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import Core.retrieval_quality as quality
from Core.event_recall import scan_event_anchors


TREE = ast.parse((Path(__file__).resolve().parents[2] / "Core/retriever.py").read_text())
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "OmnissiahRetriever")


def method(name, settings):
    method_node = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(vars(quality), retrieval_cfg=settings, RERANK_RELEVANCE_FLOOR=0,
                     scan_event_anchors=scan_event_anchors)
    exec(compile(ast.Module(body=[method_node], type_ignores=[]), "<actual-retriever>", "exec"), namespace)
    return namespace[name]


class ReconciledRetrievalTests(unittest.TestCase):
    def test_quality_path_respects_filters_before_rerank_and_zero_window(self):
        hits = [
            {"source": "wanted.pdf", "chunk_id": 1, "text": "Aren chose this because of duty."},
            {"source": "wrong.pdf", "chunk_id": 2, "text": "Aren chose this because of duty."},
        ]
        fake = Mock()
        fake._retrieve_candidate_sets.return_value = ([], [], hits)
        fake._apply_query_grounding.side_effect = lambda q, docs: docs
        fake.reranker = True
        # Supply two allowed raw chunks to exercise the actual rerank branch.
        hits.append({"source": "wanted.pdf", "chunk_id": 3, "text": "Aren explained the reason."})
        def rank(q, docs, top_k):
            self.assertEqual([d["source"] for d in docs], ["wanted.pdf"] * 2)
            self.assertTrue(all("stitch_range" not in d for d in docs))
            return [dict(docs[0], rerank_score=2), dict(docs[1], rerank_score=-1)]
        fake._rerank.side_effect = rank
        fake._stitch_chunks.side_effect = lambda docs, window: docs
        settings = {"top_k": 2, "candidate_pool": 10, "stitching_window": 2}
        result = method("_search_quality", settings)(
            fake, "Why did Aren choose this?", book_filter="wanted",
            source_filter=["wanted.pdf"], stitching_window=0)
        self.assertEqual([d["chunk_id"] for d in result], [1])
        fake._stitch_chunks.assert_called_once()
        self.assertEqual(fake._stitch_chunks.call_args.kwargs["window"], 0)

    def test_quality_selection_never_runs_legacy_recovery_too(self):
        fake = Mock()
        fake._search_quality.return_value = ["selected"]
        settings = {"top_k": 2, "candidate_pool": 10, "stitching_window": 2,
                    "retrieval_quality": True, "adaptive_retrieval": True}
        self.assertEqual(method("search", settings)(fake, "query"), ["selected"])
        fake._search_base.assert_not_called()

    def test_graph_precedence_prevents_nested_quality_pipeline(self):
        import Core.graphs.retrieval_graph as graphs
        settings = {"top_k": 2, "candidate_pool": 10, "stitching_window": 2,
                    "retrieval_quality": True, "langgraph_retrieval": True}
        fake = Mock()
        with patch.object(graphs, "RetrievalGraph") as graph:
            graph.return_value.run.return_value = {"selected_evidence": ["graph"]}
            self.assertEqual(method("search", settings)(fake, "query"), ["graph"])
        fake._search_quality.assert_not_called()
        fake._search_base.assert_not_called()

    def test_causal_rewrites_do_not_invent_rebellion_for_unrelated_questions(self):
        probes = quality.explanation_rewrites("Why did Yarrick defend Hades Hive?")
        self.assertTrue(all("Hades Hive" in p for p in probes))
        self.assertFalse(any(word in " ".join(probes) for word in ("betray", "rebel", "Chaos")))
        self.assertEqual(len(quality.explanation_rewrites("why did horus turn traitor")), 2)

    def test_word_grounding_does_not_match_turn_inside_returned(self):
        node = next(n for n in CLASS.body if getattr(n, "name", "") == "_term_matches")
        namespace = {"re": re, "TERM_VARIANTS": {}}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<actual-grounding>", "exec"), namespace)
        matches = namespace["_term_matches"]
        self.assertFalse(matches(None, "turn", "he returned to the palace"))
        self.assertTrue(matches(None, "turn", "he turned away"))
