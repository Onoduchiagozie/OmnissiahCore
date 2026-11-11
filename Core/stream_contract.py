"""Small dependency-free terminal-contract helpers for streaming responses.

A retrieval result is not a generated answer; an upstream provider error
must not be followed by a successful SSE completion.
"""

def has_answer_text(text: str) -> bool:
    return bool((text or "").strip())


def is_failure_frame(frame: str) -> bool:
    return frame.startswith("data: [ERROR]") or frame.startswith(
        "data: __STAGE__:failed"
    ) or frame.startswith("data: __STAGE__:error")
