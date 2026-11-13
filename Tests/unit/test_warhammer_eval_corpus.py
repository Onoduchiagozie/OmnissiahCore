"""100-query evaluation dataset integrity and automatic voice accuracy.

Fixture is 70 narrator, 30 Explorer; never force traffic into quotas.
"""
import json
import unittest
from collections import Counter
from pathlib import Path

from Core.response_style import infer_response_style, resolve_response_options


DATA = Path(__file__).resolve().parent.parent / "data" / "warhammer_eval_70_30.json"


class WarhammerEvalCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = json.loads(DATA.read_text(encoding="utf-8"))
        cls.questions = cls.corpus["questions"]

    def test_exact_70_30_mix(self):
        styles = Counter(row["expected_style"] for row in self.questions)
        self.assertEqual(len(self.questions), 100)
        self.assertEqual(styles, {"narrator": 70, "explorer": 30})
        self.assertEqual(self.corpus["summary"]["narrative"], 70)
        self.assertEqual(self.corpus["summary"]["explorer"], 30)

    def test_unique_questions_and_ids(self):
        self.assertEqual(len({row["id"] for row in self.questions}), 100)
        self.assertEqual(len({row["query"].casefold() for row in self.questions}), 100)

    def test_all_questions_have_style_scope_and_provenance(self):
        for row in self.questions:
            with self.subTest(id=row["id"]):
                self.assertIn(row["expected_style"], ("narrator", "explorer"))
                self.assertIn(row["retrieval_scope"], ("focus", "wide"))
                self.assertIn(row["origin"], ("user-history", "reddit-inspired", "original"))
                if row["origin"] == "reddit-inspired":
                    self.assertTrue(row["reddit_url"].startswith("https://www.reddit.com/"))

    def test_auto_style_matches_every_realistic_query(self):
        errors = []
        for row in self.questions:
            observed = infer_response_style(row["query"])
            if observed != row["expected_style"]:
                errors.append((row["id"], observed, row["expected_style"], row["query"]))
        self.assertFalse(errors, f"Voice misclassification:\n{errors}")

    def test_scope_is_independent_of_voice(self):
        for row in self.questions:
            observed = resolve_response_options(
                row["query"], endpoint_mode="adaptive",
                requested_scope=row["retrieval_scope"],
            )
            self.assertEqual(observed, (row["expected_style"], row["retrieval_scope"]))


if __name__ == "__main__":
    unittest.main()
