"""FastAPI terminal SSE contract: no provider calls or Warhammer corpus needed."""
import unittest
from Api.services.stream_contract import (
    validated_sse_frames, validate_generated_response, GenerationFailure,
)


def guard(*frames):
    return list(validated_sse_frames(iter(frames)))


class StreamContractTests(unittest.TestCase):
    def test_success_requires_answer_and_confirmed_complete(self):
        result = guard("data: __STAGE__:generating\n\n",
                       "data: On Prospero, the defenders advanced.\n\n",
                       'data: __SOURCES__:[{"source":"Prospero.pdf"}]\n\n',
                       "data: __STAGE__:complete\n\n")
        self.assertEqual(result[-2:], [
            "data: __STAGE__:complete\n\n", "data: [DONE]\n\n"])
        self.assertEqual(len(result), 5)

    def test_no_text_with_sources_is_error(self):
        result = guard('data: __SOURCES__:[{"source":"Nuceria.pdf"}]\n\n',
                       "data: __STAGE__:complete\n\n")
        self.assertIn("no answer text", result[-1])
        self.assertFalse(any("[DONE]" in x for x in result))

    def test_provider_error_never_becomes_success(self):
        result = guard("data: __STAGE__:generating\n\n",
                       "data: [ERROR] Gemini unavailable.\n\n",
                       "data: __STAGE__:complete\n\n")
        self.assertIn("[ERROR]", result[-1])
        self.assertFalse(any("[DONE]" in x for x in result))

    def test_failed_stage_is_terminal(self):
        result = guard("data: The legion was scattered.\n\n",
                       "data: __STAGE__:failed\n\n")
        self.assertIn("[ERROR]", result[-1])
        self.assertFalse(any("[DONE]" in x for x in result))

    def test_eof_after_partial_text_is_error(self):
        result = guard("data: Lorgar watched the storm.\n\n")
        self.assertIn("without confirmed completion", result[-1])
        self.assertEqual(result[0], "data: Lorgar watched the storm.\n\n")

    def test_multiline_answer_preserved(self):
        answer = "data: Russ approached.\ndata: Magnus answered.\n\n"
        result = guard(answer, "data: __STAGE__:complete\n\n")
        self.assertEqual(result[0], answer)
        self.assertEqual(result[-1], "data: [DONE]\n\n")

    def test_internal_done_not_a_success_shortcut(self):
        result = guard("data: An answer.\n\n", "data: [DONE]\n\n")
        self.assertIn("[ERROR]", result[-1])
        self.assertFalse(any("[DONE]" in x for x in result))

    def test_no_events_returns_error(self):
        result = guard()
        self.assertEqual(len(result), 1)
        self.assertIn("[ERROR]", result[0])

    def test_whitespace_not_answer(self):
        result = guard("data:   \n\n", "data: __STAGE__:complete\n\n")
        self.assertIn("no answer text", result[-1])

    def test_sources_before_text_do_not_complete_early(self):
        result = guard('data: __SOURCES__:[{"source":"Necrons.pdf"}]\n\n',
                       "data: The tomb stirred.\n\n",
                       "data: __STAGE__:complete\n\n")
        self.assertEqual(result[-1], "data: [DONE]\n\n")

    def test_sync_output_rejects_empty_answer(self):
        with self.assertRaisesRegex(GenerationFailure, "no answer"):
            validate_generated_response("   ")

    def test_sync_output_rejects_provider_error(self):
        with self.assertRaisesRegex(GenerationFailure, "generation failed"):
            validate_generated_response(" [ERROR] Gemini quota exhausted")

    def test_sync_output_preserves_successful_text(self):
        answer = "The Necron tomb stirred."
        self.assertEqual(validate_generated_response(answer), answer)

    def test_upstream_exception_is_not_swallowed(self):
        def upstream():
            yield "data: Partial.\n\n"
            raise RuntimeError("upstream offline")
        with self.assertRaisesRegex(RuntimeError, "upstream offline"):
            list(validated_sse_frames(upstream()))


if __name__ == "__main__":
    unittest.main()
