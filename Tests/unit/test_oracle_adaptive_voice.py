"""Model-free voice/scope routing regressions.

Check user requests control HOW an answer sounds independently of WHERE
the evidence comes from. No Gemini or index required.
"""
import unittest

from Core.response_style import infer_response_style, resolve_response_options


class AdaptiveVoiceTests(unittest.TestCase):
    def test_explanatory_questions_sound_human(self):
        self.assertEqual(infer_response_style("Why did Horus turn traitor?"), "explorer")
        self.assertEqual(infer_response_style("Who was Erebus?"), "explorer")
        self.assertEqual(infer_response_style("How did the Anathame work?"), "explorer")
        self.assertEqual(infer_response_style("Compare the Word Bearers and Black Templars."), "explorer")
        self.assertEqual(infer_response_style("Explain the Isstvan V battle."), "explorer")

    def test_storytelling_requests_sound_cinematic(self):
        self.assertEqual(
            infer_response_style("Tell the complete story of Horus falling to Chaos."),
            "narrator",
        )
        self.assertEqual(infer_response_style("Narrate the duel between Fulgrim and Ferrus."), "narrator")
        self.assertEqual(infer_response_style("Recount the final battle in a cinematic scene."), "narrator")

    def test_explicit_style_overrides_query_in_both_scopes(self):
        q = "Why did Horus turn traitor?"
        for scope in ("focus", "wide"):
            self.assertEqual(
                resolve_response_options(q, endpoint_mode="adaptive",
                                         requested_style="narrator",
                                         requested_scope=scope),
                ("narrator", scope),
            )
            self.assertEqual(
                resolve_response_options(q, endpoint_mode="adaptive",
                                         requested_style="explorer",
                                         requested_scope=scope),
                ("explorer", scope),
            )

    def test_auto_style_can_be_overridden_by_query_but_not_scope(self):
        self.assertEqual(
            resolve_response_options("Tell the full story of Davin.",
                                     endpoint_mode="adaptive",
                                     requested_scope="focus"),
            ("narrator", "focus"),
        )
        self.assertEqual(
            resolve_response_options("What happened on Davin?",
                                     endpoint_mode="adaptive",
                                     requested_scope="wide"),
            ("explorer", "wide"),
        )

    def test_existing_endpoints_keep_legacy_default(self):
        self.assertEqual(
            resolve_response_options("Why did Horus fall?", endpoint_mode="narrator"),
            ("narrator", "wide"),
        )
        self.assertEqual(
            resolve_response_options("Tell the story of Davin.", endpoint_mode="explorer"),
            ("explorer", "wide"),
        )
        self.assertEqual(
            resolve_response_options("Why did Horus fall?", endpoint_mode="focus"),
            ("narrator", "focus"),
        )

    def test_legacy_endpoint_can_explicitly_override_voice(self):
        self.assertEqual(
            resolve_response_options("Why did Horus fall?",
                                     endpoint_mode="focus",
                                     requested_style="auto"),
            ("explorer", "focus"),
        )
        self.assertEqual(
            resolve_response_options("Narrate Horus's return.",
                                     endpoint_mode="explorer",
                                     requested_style="narrator",
                                     requested_scope="focus"),
            ("narrator", "focus"),
        )


if __name__ == "__main__":
    unittest.main()
