"""Model-free tests for whole-corpus, named-event retrieval fallback."""
import unittest

from Core.event_recall import named_event_anchors, scan_event_anchors

HORUS_QUERY = (
    "Tell the complete story of Horus falling to Chaos, "
    "from his growing doubts and the events on Davin through his eventual betrayal."
)


def item(source, text, cid):
    return {
        "source": source,
        "text": text,
        "chunk_id": cid,
        "chapter": "test",
        "file_type": "pdf",
    }


class EventAnchorRecallTests(unittest.TestCase):
    def test_prioritizes_user_named_location_over_generic_theme(self):
        self.assertEqual(
            named_event_anchors(HORUS_QUERY), ("horus", ["davin", "chaos"])
        )

    def test_no_lore_facts_injected_when_event_not_named(self):
        subject, anchors = named_event_anchors("Narrate what happened to Corax")
        self.assertEqual((subject, anchors), ("corax", []))

    def test_recall_finds_direct_event_even_when_global_top_hits_miss(self):
        # Simulate a common failure: global top results favor later wars,
        # while the central book has many first-hand named-event passages.
        corpus = [
            item("late-war.pdf", "Horus later remembered Davin while fighting.", 1),
            item("late-war.pdf", "The warriors had heard of Horus and Davin.", 2),
            item("unrelated.pdf", "Fulgrim spoke of Chaos while Horus listened.", 3),
        ]
        for i in range(4, 13):
            corpus.append(item(
                "early-event.pdf",
                f"Horus travelled to Davin and faced another crisis in part {i}.",
                i,
            ))
        candidates, report = scan_event_anchors(
            corpus, HORUS_QUERY, limit=12, max_sources_per_anchor=3,
            per_source=5,
        )
        self.assertGreater(report["sources_by_anchor"]["davin"][0][1], 2)
        self.assertEqual(report["sources_by_anchor"]["davin"][0][0], "early-event.pdf")
        self.assertIn("early-event.pdf", [hit["source"] for hit in candidates[:3]])
        self.assertTrue(all("event_anchor" in hit for hit in candidates))

    def test_irrelevant_book_ad_is_not_a_candidate(self):
        corpus = [
            item("advert.pdf",
                 "More Horus Heresy stories from blacklibrary.com. Horus on Davin.",
                 1),
            item("actual.pdf", "Horus went to Davin and fought.", 2),
        ]
        candidates, report = scan_event_anchors(
            corpus, HORUS_QUERY, limit=5
        )
        self.assertNotIn("advert.pdf", [hit["source"] for hit in candidates])
        self.assertTrue(any(h["source"] == "actual.pdf" for h in candidates))

    def test_book_filter_honored_inside_corpus_scan(self):
        corpus = [
            item("A.pdf", "Horus went to Davin.", 1),
            item("B.pdf", "Horus fought on Davin.", 2),
        ]
        hits, _ = scan_event_anchors(
            corpus, HORUS_QUERY, book_filter="B.pdf"
        )
        self.assertEqual({h["source"] for h in hits}, {"B.pdf"})

    def test_source_filter_honored_inside_corpus_scan(self):
        corpus = [
            item("A.pdf", "Horus reached Davin.", 1),
            item("B.pdf", "Horus came to Davin.", 2),
        ]
        hits, _ = scan_event_anchors(
            corpus, HORUS_QUERY, source_filter=["A.pdf"]
        )
        self.assertEqual({h["source"] for h in hits}, {"A.pdf"})

    def test_generic_subject_works_without_horus_specific_logic(self):
        query = "Tell the full story of Aren from the siege on Meridia to his exile."
        corpus = [
            item("early.pdf", f"Aren defended Meridia on the ramparts {i}.", i)
            for i in range(6)
        ]
        hits, diag = scan_event_anchors(corpus, query)
        self.assertEqual(diag["subject"], "aren")
        self.assertEqual(diag["anchors"], ["meridia"])
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
