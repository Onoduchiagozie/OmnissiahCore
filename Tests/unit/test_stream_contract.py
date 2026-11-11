"""Streaming terminal contract tests; no model, index, or key required."""
import unittest

from Core.stream_contract import has_answer_text, is_failure_frame


class TerminalContractTests(unittest.TestCase):
    def test_source_only_response_is_not_an_answer(self):
        self.assertFalse(has_answer_text(""))
        self.assertFalse(has_answer_text("  \n\t "))

    def test_real_answer_is_accepted(self):
        self.assertTrue(has_answer_text("The defenders of Terra held the line."))

    def test_provider_error_is_terminal(self):
        self.assertTrue(is_failure_frame("data: [ERROR] Gemini returned 503\n\n"))

    def test_failed_stage_is_terminal(self):
        self.assertTrue(is_failure_frame("data: __STAGE__:failed\n\n"))
        self.assertTrue(is_failure_frame("data: __STAGE__:error\n\n"))

    def test_normal_stages_and_sources_are_not_errors(self):
        self.assertFalse(is_failure_frame("data: __STAGE__:generating\n\n"))
        self.assertFalse(is_failure_frame("data: __STAGE__:complete\n\n"))
        self.assertFalse(is_failure_frame('data: __SOURCES__:[{"source":"Prospero.pdf"}]\n\n'))

    def test_text_and_done_frames_not_errors(self):
        self.assertFalse(is_failure_frame("data: The battle continued.\n\n"))
        self.assertFalse(is_failure_frame("data: [DONE]\n\n"))


if __name__ == "__main__":
    unittest.main()
