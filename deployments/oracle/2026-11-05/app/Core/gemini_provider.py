"""Bounded Gemini retries; never switch models after publishing text."""
import json
import logging
import os
import time

import requests

log = logging.getLogger(__name__)
RETRYABLE = {429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    pass


def generate(config, payload, stream=False):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY is not configured.")
    models = list(dict.fromkeys([config["model"]] + config.get("fallback_models", [])))
    retries = max(0, min(int(config.get("retries_per_model", 1)), 2))
    base = config.get("api_base", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    started = time.monotonic()
    budget = min(float(config.get("timeout", 240)), 240)
    last_error = "no response"
    for model in models:
        for attempt in range(retries + 1):
            emitted = False
            finish_reason = None
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                raise ProviderError("Gemini retry time budget exhausted.")
            action = "streamGenerateContent?alt=sse" if stream else "generateContent"
            try:
                with requests.post(
                    f"{base}/models/{model}:{action}",
                    headers={"x-goog-api-key": key}, json=payload,
                    stream=stream, timeout=(min(10, remaining), min(60, remaining)),
                ) as response:
                    status = response.status_code
                    log.info("Gemini attempt model=%s attempt=%d status=%d", model, attempt + 1, status)
                    if status >= 400:
                        last_error = f"HTTP {status} from {model}"
                        if status not in RETRYABLE:
                            raise ProviderError(f"Gemini {last_error}; request not retried.")
                        if status == 429:
                            try:
                                details = response.json().get("error", {}).get("details", [])
                                delays = [float(d["retryDelay"].removesuffix("s")) for d in details if "retryDelay" in d]
                            except (ValueError, TypeError, AttributeError):
                                delays = []
                            retry_delay = max(delays, default=0)
                            if retry_delay > budget - (time.monotonic() - started):
                                last_error += f"; quota cooldown {int(retry_delay)} seconds"
                                break
                            if retry_delay and attempt < retries:
                                time.sleep(retry_delay)
                    else:
                        if stream:
                            for line in response.iter_lines(chunk_size=1):
                                if time.monotonic() - started > budget:
                                    raise ProviderError("Gemini stream time budget exhausted.")
                                if not line or not line.startswith(b"data:"):
                                    continue
                                raw = line[5:].strip()
                                if raw == b"[DONE]":
                                    break
                                data = json.loads(raw)
                                if data.get("error"):
                                    raise ProviderError("Gemini returned a stream error.")
                                candidates = data.get("candidates") or []
                                if candidates and candidates[0].get("finishReason"):
                                    finish_reason = candidates[0]["finishReason"]
                                    log.info("Gemini completion model=%s finish=%s usage=%s", model, finish_reason, data.get("usageMetadata", {}))
                                for token in _texts(data):
                                    emitted = True
                                    yield token
                        else:
                            data = response.json()
                            candidates = data.get("candidates") or []
                            finish_reason = candidates[0].get("finishReason") if candidates else None
                            log.info("Gemini completion model=%s finish=%s usage=%s", model, finish_reason, data.get("usageMetadata", {}))
                            _require_complete(finish_reason)
                            for token in _texts(data):
                                emitted = True
                                yield token
                        if emitted:
                            _require_complete(finish_reason)
                            return
                        raise ProviderError("Gemini returned no text; response may be blocked or token-limited.")
            except (requests.RequestException, ValueError) as exc:
                last_error = type(exc).__name__
                if emitted:
                    raise ProviderError("Gemini stream interrupted after text began; no model switch performed.") from None
                log.warning("Gemini transport failure model=%s kind=%s", model, last_error)
            if attempt < retries:
                time.sleep(min(2 ** attempt, max(0, budget - (time.monotonic() - started))))
    raise ProviderError(f"Gemini models exhausted: {last_error}.")


def _texts(data):
    candidates = data.get("candidates") or []
    if candidates:
        for part in candidates[0].get("content", {}).get("parts", []):
            if part.get("text") and not part.get("thought"):
                yield part["text"]


def _require_complete(reason):
    if reason == "STOP":
        return
    if reason == "MAX_TOKENS":
        raise ProviderError("Gemini answer incomplete: output-token limit reached. Partial text is not a completed answer.")
    if reason is None:
        raise ProviderError("Gemini response ended without a confirmed completion reason.")
    raise ProviderError(f"Gemini answer incomplete: finish reason {reason}.")
