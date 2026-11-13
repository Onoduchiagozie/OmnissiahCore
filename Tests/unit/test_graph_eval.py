"""Pure non-network evaluation tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.graph_eval import compare


class RetrievalAuditTests(unittest.TestCase):
    def test_disjoint_sources_are_reported_but_not_scored_as_accuracy(self):
        baseline = {"inspection": {"stitched_hits": [{"source": "A.pdf"}]}}
        graph = {"sources": [{"source": "B.pdf"}], "status": "inspection_complete", "attempts": 2}
        row = compare(baseline, graph, "Who met whom?")
        self.assertEqual(row["source_overlap_jaccard"], 0.0)
        self.assertTrue(row["requires_human_source_review"])
        self.assertNotIn("accuracy", row)

    def test_identical_source_overlap_is_one(self):
        baseline = {"inspection": {"stitched_hits": [{"source": "A.pdf"}]}}
        graph = {"sources": [{"source": "A.pdf"}], "attempts": 1}
        self.assertEqual(compare(baseline, graph, "Question")["source_overlap_jaccard"], 1)

    def test_no_results_have_no_defined_overlap(self):
        self.assertIsNone(compare({}, {}, "Question")["source_overlap_jaccard"])
