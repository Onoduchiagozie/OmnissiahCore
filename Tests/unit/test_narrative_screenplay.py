import json
import unittest

from Core.narrative_screenplay import compose_screenplay


class NarrativeScreenplayTests(unittest.TestCase):
    def setUp(self):
        self.passages = [{
            "source": "Original test scene",
            "chapter": "The bridge",
            "chunk_id": 17,
            "text": "Mara told Ivo to wait. An unidentified voice in the fog said, ‘I hear you.’",
        }]

    def compose(self, generate):
        return compose_screenplay("What happened at the bridge?", "Mara asked Ivo to wait.",
                                  self.passages, generate)

    def test_preserves_order_unknown_speaker_and_provenance(self):
        beats = [
            {"kind": "dialogue", "text": "Wait.", "speaker": "Mara", "evidence_ids": ["p1"]},
            {"kind": "dialogue", "text": "I hear you.", "speaker": None, "evidence_ids": ["p1"]},
        ]

        def generate(_system, payload):
            self.assertEqual(json.loads(payload)["evidence"]["p1"]["text"],
                             self.passages[0]["text"])
            return json.dumps(beats)

        result = self.compose(generate)
        self.assertEqual(result["status"], "draft")
        self.assertEqual(result["beats"], beats)
        self.assertEqual(result["sources"]["p1"]["chunk_id"], 17)
        self.assertNotIn("text", result["sources"]["p1"])

    def test_rejects_unknown_citations_and_non_json(self):
        with self.assertRaisesRegex(ValueError, "must cite supplied evidence"):
            self.compose(lambda *_: json.dumps([
                {"kind": "narration", "text": "Mara spoke.", "speaker": None,
                 "evidence_ids": ["p2"]},
            ]))
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            self.compose(lambda *_: "[ERROR]")

    def test_empty_scene_reports_evidence_gap(self):
        result = self.compose(lambda *_: "[]")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["beats"], [])
