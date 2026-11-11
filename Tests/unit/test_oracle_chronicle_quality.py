"""Model-free regression tests for chronology-aware RAG retrieval."""
import unittest

from Core.retrieval_quality import (
    cap_chronicle_context,
    chronicle_probes,
    compact_chronicle_probes,
    is_archive_boilerplate,
    is_chronicle_request,
    prioritize_chronicle_candidates,
    select_chronicle_evidence,
)


QUERY = (
    "Tell the complete story of Horus falling to Chaos, from his growing "
    "doubts and the events on Davin through his eventual betrayal."
)


def hit(cid, passage, source="fiction.pdf", score=0.8):
    return {
        "chunk_id": cid,
        "source": source,
        "text": passage,
        "rerank_score": score,
    }


class ChronicleQualityTests(unittest.TestCase):
    def test_recognizes_long_story_not_single_battle(self):
        self.assertTrue(is_chronicle_request(QUERY))
        self.assertFalse(is_chronicle_request("Describe the battle of Isstvan V"))
        self.assertFalse(is_chronicle_request("Why did Horus turn traitor?"))

    def test_probes_use_only_query_supplied_facts(self):
        probes = chronicle_probes(QUERY)
        self.assertEqual(
            probes,
            ["Horus growing doubts", "Horus events on Davin", "Horus eventual betrayal"],
        )
        self.assertFalse(any("Erebus" in p or "Serpent Lodge" in p for p in probes))

    def test_candidate_selection_preserves_all_timeline_stages(self):
        candidates = [
            hit(1, "Generic Horus overview." * 20),
            hit(2, "Horus had doubts about his destiny and his father's decision." * 10),
            hit(3, "Horus had doubts about his destiny and his father's decision." * 9 + "x"),
            hit(4, "Horus was on Davin and faced the events on Davin." * 10),
            hit(5, "Horus weighed his eventual betrayal with his legion." * 10),
        ]
        chosen = prioritize_chronicle_candidates(QUERY, candidates, limit=3)
        ids = {x["chunk_id"] for x in chosen}
        self.assertIn(2, ids)
        self.assertIn(4, ids)
        self.assertIn(5, ids)

    def test_final_evidence_uses_explicit_stages_then_supporting_passage(self):
        candidates = [
            hit(1, "A broad unrelated account of distant events." * 8, score=3.0),
            hit(2, "Horus had doubts about his destiny." * 10, score=0.6),
            hit(3, "On Davin Horus made a fateful decision." * 10, score=-0.4),
            hit(4, "Horus considered betrayal and the consequences." * 10, score=0.5),
            hit(5, "Horus and his warriors advanced." * 10, score=0.8),
        ]
        chosen = select_chronicle_evidence(QUERY, candidates, top_k=4)
        ids = {x["chunk_id"] for x in chosen}
        self.assertIn(2, ids)
        self.assertIn(3, ids)
        self.assertIn(4, ids)
        self.assertIn(5, ids)
        self.assertNotIn(1, ids)
        self.assertEqual(len(chosen), 4)

    def test_deduplicates_pdf_and_epub_with_different_spacing(self):
        passage = "Horus reached Davin while his companions debated the tragedy. " * 8
        pdf = hit(1, passage, "False Gods.pdf")
        epub = hit(2, passage.replace(" ", "\n"), "False Gods.epub")
        chosen = select_chronicle_evidence(QUERY, [pdf, epub], top_k=2)
        self.assertEqual(len(chosen), 1)

    def test_compact_recall_probes_do_not_invent_lore(self):
        probes = [p.casefold() for p in compact_chronicle_probes(QUERY)]
        self.assertEqual(probes, ["horus doubts", "horus davin", "horus betrayal"])
        self.assertFalse(any("erebus" in p or "serpent lodge" in p for p in probes))

    def test_marketing_pages_are_not_event_evidence(self):
        ad = hit(
            7, "More Horus Heresy stories from blacklibrary.com. "
               "Book 1 Horus Rising Book 2 False Gods. "
               "Horus Davin betrayal and other summaries."
        )
        self.assertTrue(is_archive_boilerplate(ad["text"]))
        chosen = prioritize_chronicle_candidates(QUERY, [ad], limit=5)
        self.assertEqual(chosen, [])

    def test_direct_davin_source_beats_high_scoring_later_mentions(self):
        # Synthetic evidence: direct event should survive even when its
        # global cross-encoder score is lower than later references.
        facet = "Horus events on Davin"
        direct = hit(
            100,
            "Horus landed on Davin and confronted the rebels. " * 8,
            source="02. False Gods.pdf",
            score=0.3,
        )
        direct["facet_rerank_scores"] = {facet: 4.2}
        later = hit(
            101,
            "Horus was remembered by loyal soldiers in many distant wars. " * 8
            + "Horus had once fought on Davin before returning to battle.",
            source="46. Ruinstorm.pdf",
            score=6.4,
        )
        later["facet_rerank_scores"] = {facet: -1.7}
        candidates = [later, direct]
        chosen = select_chronicle_evidence(QUERY, candidates, top_k=2)
        self.assertEqual(chosen[0]["source"], "02. False Gods.pdf")

    def test_chronicle_limit_caps_long_context_and_keeps_source(self):
        chunks = [
            hit(i, "A detailed battle scene. " * 500)
            for i in range(6)
        ]
        clipped = cap_chronicle_context(chunks, total_chars=8000, per_chunk_chars=2500)
        self.assertLessEqual(sum(len(x["text"]) for x in clipped), 8000)
        self.assertEqual(clipped[0]["source"], "fiction.pdf")
        self.assertEqual(len(chunks[0]["text"]), len("A detailed battle scene. " * 500))


if __name__ == "__main__":
    unittest.main()
