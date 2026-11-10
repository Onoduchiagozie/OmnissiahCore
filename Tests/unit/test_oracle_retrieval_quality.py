"""Regression checks for Oracle's causal retrieval helpers; no model downloads."""
import unittest

from Core.retrieval_quality import (
    cap_causal_context, explanation_rewrites, interleave_unique,
    is_causal_question, rank_causal_evidence,
)


class OracleRetrievalQualityTests(unittest.TestCase):
    def test_classifies_why_and_cause(self):
        self.assertTrue(is_causal_question("Why did Horus turn traitor?"))
        self.assertTrue(is_causal_question(
            "What caused Horus to reject the Emperor after being wounded on Davin?"
        ))
        self.assertFalse(is_causal_question("Describe the battle on Isstvan V"))

    def test_rewrites_do_not_assert_unretrieved_lore(self):
        probes = explanation_rewrites("What caused Horus to betray the Emperor?")
        self.assertEqual(len(probes), 2)
        self.assertTrue(all("Horus" in probe for probe in probes))
        self.assertFalse(any("Erebus" in probe for probe in probes))

    def test_interleaves_query_probes_without_duplicate_ids(self):
        dense = [{"source": "a", "chunk_id": 1}, {"source": "a", "chunk_id": 2}]
        expanded = [{"source": "a", "chunk_id": 1}, {"source": "b", "chunk_id": 3}]
        output = interleave_unique(dense, expanded, limit=3)
        self.assertEqual([c["chunk_id"] for c in output], [1, 2, 3])

    def test_prefers_explanation_over_unanswered_question(self):
        unknown = {"chunk_id": 1, "text": "Dorn did not know why Horus had turned on the Emperor."}
        cause = {"chunk_id": 2, "text": "Horus became distrustful because deception and fear shaped his decision."}
        output = rank_causal_evidence("Why did Horus rebel?", [unknown, cause])
        self.assertEqual(output[0]["chunk_id"], 2)

    def test_does_not_change_non_causal_queries(self):
        docs = [{"text": "some lore"}]
        self.assertIs(rank_causal_evidence("Describe the duel", docs), docs)
        self.assertIs(cap_causal_context("Describe the duel", docs), docs)

    def test_caps_context_without_mutating_original(self):
        docs = [{"chunk_id": i, "text": "Horus because of doubt. " * 500} for i in range(8)]
        result = cap_causal_context("Why did Horus rebel?", docs, total_chars=5000, per_chunk_chars=2000)
        self.assertLessEqual(sum(len(c["text"]) for c in result), 5000)
        self.assertEqual(docs[0]["text"], "Horus because of doubt. " * 500)

    def test_handles_empty_evidence(self):
        self.assertEqual(cap_causal_context("Why did he defect?", []), [])
        self.assertEqual(rank_causal_evidence("Why did Horus rebel?", []), [])

if __name__ == "__main__":
    unittest.main()
