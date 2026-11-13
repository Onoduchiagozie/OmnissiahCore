"""Regression tests for opt-in evidence recovery; no corpus or paid LLM used.

Run: python -m unittest discover -s Tests -p test_retrieval_recovery.py -v
"""
import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "Core" / "retrieval_recovery.py"
spec = importlib.util.spec_from_file_location("retrieval_recovery", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeReranker:
    def predict(self, pairs):
        return [2.0 if "cause" in text.lower() else -3.0 for _, text in pairs]


class FakeRetriever:
    def __init__(self):
        self.bm25 = True
        self.reranker = FakeReranker()
        self.variants = []

    def _enrich_query(self, query):
        return query

    def _faiss_search(self, query, k):
        self.variants.append(query)
        if "motivations behind" in query:
            return [{
                "chunk_id": 8, "source": "book-a",
                "text": "The causes behind the event are explained here.",
            }]
        return [{
            "chunk_id": 7, "source": "book-a",
            "text": "The name is mentioned, without explanation.",
        }]

    def _bm25_search(self, query, k):
        return []

    def _rrf_merge(self, dense, sparse, k):
        return dense[:k]

    def _stitch_chunks(self, hits, window):
        return list(hits)


class RecoveryTests(unittest.TestCase):
    def test_explanatory_query_uses_generic_variants(self):
        variants = module.plan_queries("why did horus turn traitor")
        self.assertEqual(variants[0], "why did horus turn traitor")
        self.assertEqual(len(variants), 4)
        self.assertIn("causes and motivations", variants[1])
        self.assertFalse(any("Davin" in v or "False Gods" in v for v in variants))

    def test_blank_query(self):
        self.assertEqual(module.plan_queries("   "), [])

    def test_recovery_finds_evidence_initial_query_missed(self):
        retriever = FakeRetriever()
        out = module.recover_evidence(
            retriever, "why did horus turn traitor",
            baseline=[], top_k=2, candidate_pool=20, stitching_window=1,
        )
        self.assertEqual(len(out), 2)
        self.assertTrue(any("causes" in p["text"].lower() for p in out))
        self.assertEqual(len(retriever.variants), 4)

    def test_source_filter_never_leaks_other_book(self):
        retriever = FakeRetriever()
        out = module.recover_evidence(
            retriever, "why did horus turn traitor",
            baseline=[], top_k=3, candidate_pool=20,
            stitching_window=1, book_filter="book-b",
        )
        self.assertEqual(out, [])

    def test_original_results_not_silently_dropped(self):
        retriever = FakeRetriever()
        base = [{
            "chunk_id": 99, "source": "book-b",
            "text": "important base scene",
        }]
        out = module.recover_evidence(
            retriever, "why did horus turn traitor",
            baseline=base, top_k=3, candidate_pool=20, stitching_window=1,
        )
        self.assertEqual(out[0]["chunk_id"], 99)


if __name__ == "__main__":
    unittest.main()
