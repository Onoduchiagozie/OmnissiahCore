"""Provider completion/fallback regressions without live API calls."""
import os
import unittest
from unittest.mock import patch
from Core.gemini_provider import generate, ProviderError


class Response:
    def __init__(self, status, data=None, lines=None):
        self.status_code = status
        self.data = data or {}
        self.lines = lines or []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def json(self):
        return self.data
    def iter_lines(self, **kwargs):
        return iter(self.lines)


class ProviderTests(unittest.TestCase):
    config = {"model": "primary", "fallback_models": ["second", "last"],
              "retries_per_model": 0, "timeout": 60}

    def test_exhausted_primary_and_second_reach_last_configured_model(self):
        okay = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "complete"}]}}]}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}), \
             patch("Core.gemini_provider.requests.post", side_effect=[
                 Response(429), Response(503), Response(200, okay)]) as post:
            self.assertEqual("".join(generate(self.config, {})), "complete")
            self.assertEqual(post.call_count, 3)
            self.assertIn("/models/last:", post.call_args.args[0])

    def test_all_configured_models_failed_raises_error(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}), \
             patch("Core.gemini_provider.requests.post", return_value=Response(429)) as post:
            with self.assertRaises(ProviderError):
                list(generate(self.config, {}))
            self.assertEqual(post.call_count, 3)

    def test_incomplete_output_does_not_report_success_or_switch_after_text(self):
        line = b'data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}'
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}), \
             patch("Core.gemini_provider.requests.post", return_value=Response(200, lines=[line])) as post:
            stream = generate(self.config, {}, stream=True)
            self.assertEqual(next(stream), "partial")
            with self.assertRaises(ProviderError):
                next(stream)
            self.assertEqual(post.call_count, 1)

    def test_token_limited_json_never_returns_a_completed_answer(self):
        truncated = {"candidates": [{"finishReason": "MAX_TOKENS",
                                    "content": {"parts": [{"text": "partial"}]}}]}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}), \
             patch("Core.gemini_provider.requests.post", return_value=Response(200, truncated)):
            with self.assertRaises(ProviderError):
                list(generate(self.config, {}))


if __name__ == "__main__":
    unittest.main()
