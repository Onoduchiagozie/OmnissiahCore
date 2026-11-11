"""Validate 100 *new* questions without using Horus as a test subject.

Model-free checks are ROUTING tests only. They are not evidence or generation
quality proof. Run evidence in the real Oracle pipeline separately.
"""
import json
from pathlib import Path
import unittest

from Core.response_style import infer_response_style, resolve_response_options

PATH = Path(__file__).resolve().parents[1] / "data" / "warhammer_eval_nonhorus_70_30.json"


class NonHorusEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = json.loads(PATH.read_text(encoding="utf-8"))["cases"]

    def test_exact_70_30_and_no_horus(self):
        self.assertEqual(len(self.questions), 100)
        self.assertEqual(
            sum(q["expected_style"] == "narrator" for q in self.questions), 70
        )
        self.assertEqual(
            sum(q["expected_style"] == "explorer" for q in self.questions), 30
        )
        self.assertTrue(all("horus" not in q["query"].lower() for q in self.questions))

    def test_all_unique_with_broad_coverage(self):
        self.assertEqual(len({q["id"] for q in self.questions}), 100)
        self.assertEqual(
            len({q["query"].casefold() for q in self.questions}), 100
        )
        self.assertGreaterEqual(len({q["category"] for q in self.questions}), 10)
        self.assertTrue(all(q["origin"] == "original" for q in self.questions))

    def test_all_100_auto_routes(self):
        mistakes = [
            (q["id"], q["query"], infer_response_style(q["query"]))
            for q in self.questions
            if infer_response_style(q["query"]) != q["expected_style"]
        ]
        self.assertEqual(mistakes, [])

    def test_scope_is_independent_of_style(self):
        for q in self.questions:
            style, scope = resolve_response_options(
                q["query"], endpoint_mode="adaptive",
                requested_style="auto", requested_scope=q["retrieval_scope"],
            )
            self.assertEqual(style, q["expected_style"], q["id"])
            self.assertEqual(scope, q["retrieval_scope"], q["id"])

    def test_explicit_voice_override(self):
        for q in self.questions:
            override = (
                "explorer" if q["expected_style"] == "narrator" else "narrator"
            )
            self.assertEqual(
                resolve_response_options(
                    q["query"], endpoint_mode="adaptive",
                    requested_style=override,
                    requested_scope=q["retrieval_scope"],
                ),
                (override, q["retrieval_scope"]),
            )


if __name__ == "__main__":
    unittest.main()
