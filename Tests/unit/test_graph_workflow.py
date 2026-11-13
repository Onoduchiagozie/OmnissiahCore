"""Real LangGraph control-flow tests with injected corpus/model test doubles."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Core.graph_workflow import (  # noqa: E402
    NO_EVIDENCE, EvidenceGraph, assess_evidence, build_plan,
    merge_evidence, normalize_query, bound_context, prioritize_subjects,
)

# Integration checks must fail if the real runtime is missing.
import langgraph.graph  # noqa: E402,F401


def chunk(source="Fulgrim.pdf", text=None, **extras):
    return {"source": source, "chapter": "Chapter 1",
            "text": text or "Fulgrim confronted Ferrus Manus as the battle reached its end, and the witnesses watched.",
            **extras}


class RecoveryBudgetTests(unittest.TestCase):
    def test_explanation_recovery_displaces_full_irrelevant_context(self):
        calls = []
        def retrieve(query, **kwargs):
            calls.append(query)
            if len(calls) == 1:
                return [chunk(source=f"chapter-{i}.pdf",
                              text="Fulgrim watched the chamber. " + "Silent stones lined the floor. " * 100)
                        for i in range(8)]
            return [chunk(source="cause.pdf",
                          text="Fulgrim changed because he was manipulated by the voice in the blade, which influenced his decisions.")]
        graph = EvidenceGraph(retrieve, lambda q, chunks, intent: chunks[0]["source"])
        result = graph.run("Why did Fulgrim change?", top_k=8)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["response"], "cause.pdf")


class PlanningTests(unittest.TestCase):
    def test_normalization_never_doubles_manus(self):
        self.assertEqual(normalize_query("Ferrus Manus vs Fulgrimm"),
                         "Ferrus Manus vs Fulgrim")

    def test_plan_is_bounded_and_keeps_comparison_intent(self):
        intent, subjects, queries = build_plan("Ferrus vs Fulgrim")
        self.assertEqual(intent, "comparison")
        self.assertEqual(subjects, ["Ferrus Manus", "Fulgrim"])
        self.assertTrue(1 <= len(queries) <= 3)

    def test_reason_is_not_rewritten_as_a_duel(self):
        intent, _, variants = build_plan("Why did Fulgrim change?")
        self.assertEqual(intent, "explanation")
        self.assertNotIn("duel", " ".join(variants).casefold())

    def test_query_blank_rejected(self):
        with self.assertRaises(ValueError):
            build_plan("   ")

    def test_deduplicates_sources_and_preserves_book_filter(self):
        first = chunk()
        merged = merge_evidence([first], [dict(first), chunk("Other.pdf")],
                                book_filter="Fulgrim", source_filter=None, limit=5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "Fulgrim.pdf")

    def test_source_filter_requires_exact_name(self):
        merged = merge_evidence([], [chunk("Fulgrim.pdf"), chunk("Fulgrim-extra.pdf")],
                                book_filter=None, source_filter=["Fulgrim.pdf"], limit=5)
        self.assertEqual([c["source"] for c in merged], ["Fulgrim.pdf"])

    def test_rejects_short_and_invalid_source_metadata(self):
        merged = merge_evidence([], [{"source": "", "text": "t" * 100},
                                    {"source": "A.pdf", "text": "short"}],
                                book_filter=None, source_filter=None, limit=5)
        self.assertEqual(merged, [])

    def test_bounded_context_limits_total_text(self):
        docs = [chunk(source=f"Book_{i}.pdf", text="Fulgrim " * 1200) for i in range(10)]
        limited = bound_context(docs)
        self.assertLessEqual(sum(len(c["text"]) for c in limited), 16000)
        self.assertTrue(all(len(c["text"]) <= 2400 for c in limited))

    def test_long_passage_deduplicates_across_searches(self):
        text = "Ferrus Manus and Fulgrim " * 300
        first = merge_evidence([], [chunk(text=text)], book_filter=None, source_filter=None, limit=10)
        second = merge_evidence(first, [chunk(text=text)], book_filter=None, source_filter=None, limit=10)
        self.assertEqual(len(second), 1)

    def test_anchors_are_preserved_when_many_chapters_returned(self):
        docs = [chunk(source=f"Ferrus_{i}.pdf", text="Ferrus Manus marched with the Iron Hands " * 80)
                for i in range(8)]
        docs.append(chunk("Fulgrim.pdf", text="Fulgrim marched with the Emperor's Children " * 80))
        priority = prioritize_subjects(docs, ["Ferrus Manus", "Fulgrim"])
        limited = bound_context(priority)
        self.assertEqual(assess_evidence("Ferrus vs Fulgrim", "comparison",
                                         ["Ferrus Manus", "Fulgrim"], limited)[0], True)

    def test_comparison_requires_both_subjects_in_evidence(self):
        first = chunk(text="Ferrus Manus is described fighting alongside the Iron Hands on the battlefield.")
        self.assertEqual(assess_evidence("Ferrus vs Fulgrim", "comparison",
                                         ["Ferrus Manus", "Fulgrim"], [first])[0], False)
        second = chunk("Different.pdf", text="Fulgrim watched the Emperor's Children prepare for the conflict and regroup.")
        self.assertEqual(assess_evidence("Ferrus vs Fulgrim", "comparison",
                                         ["Ferrus Manus", "Fulgrim"], [first, second])[0], True)


class HorusRegressionTests(unittest.TestCase):
    QUESTION = "Why did Horus turn traitor?"

    def test_unrelated_horus_passage_cannot_trigger_generation(self):
        passage = chunk("Horus Rising.pdf", text=(
            "Horus commanded his captains to prepare their ships and review "
            "the strategy for the Great Crusade. Officers gathered around "
            "their tactical charts before the assembly began."))
        called = []
        result = EvidenceGraph(
            lambda q, **kw: [passage],
            lambda q, docs, intent: called.append(True) or "UNSUPPORTED ANSWER"
        ).run(self.QUESTION)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertGreaterEqual(result["attempts"], 1)
        self.assertLessEqual(result["attempts"], 3)
        self.assertEqual(called, [])

    def test_causally_relevant_horus_passage_reaches_generation(self):
        passage = chunk("False Gods.pdf", text=(
            "Erebus manipulated Horus while he lay dying after the wound from "
            "the Anathame on Davin. Visions in the Serpent Lodge convinced "
            "Horus that his father might abandon the primarchs, and Horus "
            "chose rebellion through ambition, resentment and fear."))
        result = EvidenceGraph(
            lambda q, **kw: [passage],
            lambda q, docs, intent: "Horus rebelled under Erebus's manipulation."
        ).run(self.QUESTION)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["intent"], "explanation")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["evidence_grade"], "explanatory_language_present_unverified")


class GraphExecutionTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.generations = []

    def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [chunk()]

    def generate(self, query, chunks, intent):
        self.generations.append((query, chunks, intent))
        return "Ferrus Manus and Fulgrim were connected by the events described."

    def graph(self, retrieve=None, generate=None, max_attempts=3):
        return EvidenceGraph(retrieve or self.retrieve, generate or self.generate,
                             max_attempts=max_attempts)

    def test_success_uses_original_question_and_filters(self):
        result = self.graph().run("Ferrus vs Fulgrimm", top_k=4,
                                  book_filter="Fulgrim", source_filter=["Fulgrim.pdf"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(self.generations[0][0], "Ferrus vs Fulgrimm")
        self.assertEqual(self.calls[0][1]["source_filter"], ["Fulgrim.pdf"])
        self.assertEqual(result["chunks_used"], 1)

    def test_empty_retrieval_never_calls_llm(self):
        result = self.graph(retrieve=lambda q, **kw: []).run("What is the unknown relic?")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["response"], NO_EVIDENCE)
        self.assertEqual(self.generations, [])
        self.assertLessEqual(result["attempts"], 3)

    def test_comparison_only_one_side_never_calls_llm(self):
        result = self.graph(retrieve=lambda q, **kw: [chunk(text="Ferrus Manus faced the Iron Hands and made a decision in battle.")]).run(
            "Ferrus versus Fulgrim")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(self.generations, [])

    def test_retry_adds_missing_subject_and_then_generates(self):
        def retrieval(q, **kw):
            self.calls.append(q)
            if len(self.calls) == 1:
                return [chunk(text="Ferrus Manus was present with the Iron Hands on the battlefield.")]
            return [chunk("Other.pdf", text="Fulgrim was present with his legion during the conflict and aftermath.")]
        result = self.graph(retrieve=retrieval).run("Ferrus vs Fulgrim")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["chunks_used"], 2)

    def test_retry_can_recover_when_first_pass_is_full(self):
        passes = [0]
        def retrieval(q, **kw):
            passes[0] += 1
            if passes[0] == 1:
                return [chunk(f"Ferrus_{i}.pdf", text="Ferrus Manus went to war alongside the Iron Hands " * 30)
                        for i in range(10)]
            return [chunk("Fulgrim.pdf", text="Fulgrim prepared for war alongside the Emperor's Children " * 30)]
        result = self.graph(retrieve=retrieval).run("Ferrus vs Fulgrim", top_k=8)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["attempts"], 2)
        self.assertLessEqual(result["chunks_used"], 16)

    def test_retriever_ignores_filtered_out_evidence(self):
        result = self.graph(retrieve=lambda q, **kw: [chunk("Unexpected.pdf")]).run(
            "Ferrus vs Fulgrim", source_filter=["Fulgrim.pdf"])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(self.generations, [])

    def test_max_attempts_one_really_stops(self):
        result = self.graph(retrieve=lambda q, **kw: [], max_attempts=1).run("Who was missing?")
        self.assertEqual(result["attempts"], 1)

    def test_retriever_crash_is_controlled_and_no_generation(self):
        def bad(q, **kw):
            raise RuntimeError("private server path")
        result = self.graph(retrieve=bad).run("Ferrus")
        self.assertEqual(result["status"], "retrieval_failed")
        self.assertEqual(result["attempts"], 1)
        self.assertNotIn("private server path", str(result))
        self.assertEqual(self.generations, [])

    def test_llm_exception_is_controlled(self):
        def bad(q, chunks, intent):
            raise RuntimeError("sensitive model details")
        result = self.graph(generate=bad).run("Ferrus Manus")
        self.assertEqual(result["status"], "generation_failed")
        self.assertNotIn("sensitive model details", str(result))

    def test_llm_error_string_is_not_success(self):
        result = self.graph(generate=lambda q, chunks, intent: "[ERROR] Gemini HTTP 503: overloaded").run("Fulgrim")
        self.assertEqual(result["status"], "generation_failed")

    def test_limits_are_validated(self):
        with self.assertRaises(ValueError):
            self.graph().run("Fulgrim", top_k=0)
        with self.assertRaises(ValueError):
            EvidenceGraph(self.retrieve, self.generate, max_attempts=99)

    def test_fails_open_to_nothing_when_passage_is_unrelated(self):
        unrelated = chunk(text="The Ultramarines gathered on Calth and watched their fleet enter the sector.")
        result = self.graph(retrieve=lambda q, **kw: [unrelated]).run("What happened to Angron?")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(self.generations, [])




class VerificationAndInspectionTests(unittest.TestCase):
    def setUp(self):
        self.source = chunk()
        self.calls = {"generate": 0, "verify": 0}

    def make(self, verifier=None):
        def generate(question, chunks, intent):
            self.calls["generate"] += 1
            return "Ferrus Manus confronted Fulgrim in the supplied battle account."
        def verify(question, chunks, answer, intent):
            self.calls["verify"] += 1
            return verifier(question, chunks, answer, intent)
        return EvidenceGraph(lambda question, **kw: [self.source], generate,
                             verify=verify if verifier else None)

    def test_inspection_without_llm_or_verifier(self):
        result = self.make(lambda *x: True).run("Ferrus vs Fulgrim", generate_enabled=False)
        self.assertEqual(result["status"], "inspection_complete")
        self.assertEqual(result["verification"], "not_run")
        self.assertEqual(result["response"], "")
        self.assertTrue(result["passages"][0]["excerpt"])
        self.assertEqual(self.calls["generate"], 0)
        self.assertEqual(self.calls["verify"], 0)

    def test_verifier_approves_and_reports_passed(self):
        result = self.make(lambda *x: True).run("Ferrus vs Fulgrim")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["verification"], "passed")
        self.assertEqual(self.calls["verify"], 1)
        self.assertEqual(result["passages"], [])

    def test_verifier_rejects_and_hides_unsupported_answer(self):
        result = self.make(lambda *x: False).run("Ferrus vs Fulgrim")
        self.assertEqual(result["status"], "verification_failed")
        self.assertEqual(result["verification"], "rejected")
        self.assertEqual(result["response"], NO_EVIDENCE)
        self.assertEqual(self.calls["verify"], 1)

    def test_verifier_non_boolean_truthy_value_is_rejected(self):
        result = self.make(lambda *x: "true").run("Ferrus vs Fulgrim")
        self.assertEqual(result["status"], "verification_failed")

    def test_verifier_exception_is_fail_closed_without_internal_message(self):
        def fail(*x):
            raise RuntimeError("secret private prompt content")
        result = self.make(fail).run("Ferrus vs Fulgrim")
        self.assertEqual(result["status"], "verification_failed")
        self.assertEqual(result["verification"], "error")
        self.assertNotIn("secret private prompt", str(result))

    def test_no_evidence_skips_generation_and_verification(self):
        graph = EvidenceGraph(lambda query, **kw: [],
                              lambda *x: self.calls.__setitem__("generate", 99),
                              verify=lambda *x: self.calls.__setitem__("verify", 99))
        result = graph.run("Ferrus vs Fulgrim")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(self.calls["generate"], 0)
        self.assertEqual(self.calls["verify"], 0)

    def test_inspection_excerpt_has_fixed_upper_bound(self):
        self.source = chunk(text="Fulgrim and Ferrus Manus fought in battle. " * 300)
        result = self.make().run("Ferrus vs Fulgrim", generate_enabled=False)
        self.assertTrue(all(len(x["excerpt"]) <= 500 for x in result["passages"]))


if __name__ == "__main__":
    print("Graph runtime:", "REAL LangGraph" if REAL_LANGGRAPH else "TEST DOUBLE (LangGraph not installed)")
    unittest.main()
