"""Focused regression checks for the staged retrieval corrections.

Run from repository root:
    python -m unittest discover -s Tests -p test_retrieval_invariants.py -v

Only standard library is required. These are code-level invariant tests;
real FAISS/BM25/LLM accuracy must be tested separately on Oracle.
"""
import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "Core" / "retriever.py").read_text(encoding="utf-8-sig"))
RETRIEVER = next(item for item in TREE.body if isinstance(item, ast.ClassDef)
                 and item.name == "OmnissiahRetriever")


def compile_method(method_name, globals_dict):
    method = next(item for item in RETRIEVER.body if isinstance(item, ast.FunctionDef)
                  and item.name == method_name)
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(globals_dict)
    exec(compile(module, "<retriever-extracted>", "exec"), namespace)
    return namespace[method_name]


class RetrievalInvariants(unittest.TestCase):
    def test_stitch_does_not_cross_sources(self):
        stitch = compile_method("_stitch_chunks", {"re": re})
        chunks = [
            {"chunk_id": 100, "source": "Horus Rising", "text": "first scene"},
            {"chunk_id": 101, "source": "False Gods", "text": "different book"},
            {"chunk_id": 102, "source": "Horus Rising", "text": "next same-source text"},
        ]
        fake = type("RetrieverFake", (), {})()
        fake.metadata = chunks
        fake._id_to_idx = {x["chunk_id"]: i for i, x in enumerate(chunks)}
        result = stitch(fake, [chunks[0]], window=2)
        self.assertEqual(len(result), 1)
        self.assertNotIn("different book", result[0]["text"])
        self.assertIn("first scene", result[0]["text"])
        self.assertIn("next same-source text", result[0]["text"])

    def test_zero_window_retains_raw_passage(self):
        stitch = compile_method("_stitch_chunks", {"re": re})
        fake = type("RetrieverFake", (), {})()
        hits = [{"chunk_id": 2, "source": "book", "text": "raw"}]
        self.assertIs(stitch(fake, hits, window=0), hits)

    def test_bge_m3_query_prefix_experiment_is_opt_in(self):
        fake = object()
        normal = compile_method("_enrich_query", {
            "retrieval_cfg": {},
            "embedding_cfg": {"model": "BAAI/bge-m3"},
        })
        raw = compile_method("_enrich_query", {
            "retrieval_cfg": {"bge_m3_raw_queries": True},
            "embedding_cfg": {"model": "BAAI/bge-m3"},
        })
        self.assertTrue(normal(fake, "why did Horus turn traitor").startswith("Represent this sentence"))
        self.assertEqual(raw(fake, "why did Horus turn traitor"), "why did Horus turn traitor")

    def test_inspect_exposes_selected_hits(self):
        method = next(item for item in RETRIEVER.body if isinstance(item, ast.FunctionDef)
                      and item.name == "inspect")
        body = ast.unparse(method)
        self.assertIn('"selected_hits"', body)
        self.assertIn("self.search(", body)


if __name__ == "__main__":
    unittest.main()
