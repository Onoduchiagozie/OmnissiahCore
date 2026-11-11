"""Independent style and retrieval-scope selection for OmnissiahCore.

Voice is HOW the answer sounds. Scope is WHERE its evidence comes from.
Both are explicit API choices, and "auto" infers voice only from user wording.
No Warhammer-specific answer facts are embedded here.
"""
import re
from typing import Optional

STYLES = frozenset({"auto", "explorer", "narrator"})
SCOPES = frozenset({"focus", "wide"})

_NARRATE = re.compile(
    r"\b(narrate|recount|dramatize|dramatise|reenact|re-enact|"
    r"cinematic|storytelling|scene|chronicle|show me|"
    r"tell (?:me )?(?:the |a )?(?:complete |full |whole |entire )?story|"
    r"play out|bring (?:it|this) to life)\b"
    r"|\b(?:complete|full|whole|entire)\s+(?:story|chronicle|saga)\b"
    r"|\b(?:as if (?:i|we) (?:was|were) there)\b",
    re.IGNORECASE,
)
_EXPLAIN = re.compile(
    r"^\s*(?:why|what|who|when|where|how|which)\b"
    r"|\b(?:explain|compare|difference|define|meaning|"
    r"analysis|analyse|analyze|overview|summary|summarize|"
    r"motives|reasons?|significance|tell me about)\b",
    re.IGNORECASE,
)


def infer_response_style(query: str) -> str:
    """Natural questions get a human explanation; storytelling requests get cinematic narration.

    Explicit narration beats other general words such as 'what happened'.
    Unclear text defaults to explorer rather than manufacturing a scene.
    """
    if _NARRATE.search(query):
        return "narrator"
    if _EXPLAIN.search(query):
        return "explorer"
    return "explorer"


def resolve_response_options(
    query: str,
    *,
    endpoint_mode: str,
    requested_style: Optional[str] = None,
    requested_scope: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve (style, scope); endpoint legacy defaults are preserved."""
    default_style = {
        "narrator": "narrator",
        "focus": "narrator",
        "explorer": "explorer",
        "adaptive": "auto",
        "remembrancer": "auto",
    }.get(endpoint_mode, "auto")
    default_scope = "focus" if endpoint_mode == "focus" else "wide"
    style = requested_style or default_style
    scope = requested_scope or default_scope
    if style not in STYLES:
        raise ValueError(f"Unknown response style: {style}")
    if scope not in SCOPES:
        raise ValueError(f"Unknown retrieval scope: {scope}")
    return (infer_response_style(query) if style == "auto" else style, scope)
