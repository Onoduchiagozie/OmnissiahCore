"""Validate the real FastAPI SSE outcome before writing a terminal success.

The retrieval/generation runtime emits preformatted SSE events and a stage
marker. Router code must NEVER append [DONE] to an errored, incomplete, or
empty stream. This guard is provider-independent and uses no model/database.
"""
from collections.abc import Iterable, Iterator
import logging

logger = logging.getLogger(__name__)


def _event_data(frame: str) -> str:
    """Decode one already-encoded SSE event, including multi-line data."""
    lines = frame.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    data = []
    for line in lines:
        if line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if not data:
        raise ValueError("Stream runtime returned an event without SSE data")
    return "\n".join(data)


def validated_sse_frames(upstream: Iterable[str]) -> Iterator[str]:
    """Pass through data while enforcing one truthful terminal result.

    - Upstream's __STAGE__:complete is withheld until checks pass.
    - [ERROR] and failed stages are terminal; never append [DONE].
    - [DONE] is solely the API router's responsibility.
    - If the generator ends abruptly, a terminal error is produced.
    - Source metadata is never counted as generated text.
    """
    generated = False
    declared_complete = False
    for frame in upstream:
        data = _event_data(frame)
        if data == "[DONE]":
            # Unexpected internal DONE: it must not short-circuit validation.
            continue
        if "[ERROR]" in data:
            logger.error("Upstream generation failed: %s", data)
            yield "data: [ERROR] The service could not finish your request.\n\n"
            return
        if data.startswith("__STAGE__:"):
            stage = data.partition(":")[2].strip().lower()
            if stage in {"failed", "error"}:
                yield "data: [ERROR] Backend generation failed.\n\n"
                return
            if stage == "complete":
                declared_complete = True
                continue
            yield frame
            continue
        if data.startswith("__SOURCES__:"):
            yield frame
            continue
        if data.strip():
            generated = True
        yield frame

    if not declared_complete:
        yield "data: [ERROR] Backend stream ended without confirmed completion.\n\n"
    elif not generated:
        yield "data: [ERROR] Backend retrieved sources but returned no answer text.\n\n"
    else:
        yield "data: __STAGE__:complete\n\n"
        yield "data: [DONE]\n\n"


class GenerationFailure(ValueError):
    """The model did not provide a usable, successfully generated answer."""


def validate_generated_response(response: str) -> str:
    """Validate a synchronous model answer before returning HTTP 200."""
    if not isinstance(response, str) or not response.strip():
        raise GenerationFailure("Language model returned no answer.")
    if "[ERROR]" in response:
        raise GenerationFailure("Language model generation failed.")
    return response
