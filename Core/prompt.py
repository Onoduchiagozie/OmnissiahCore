"""
OmnissiahCore - Core/prompt.py

Prompt builders for the LLM. These prompts are intentionally strict because
the retriever is only useful if the model stays grounded in retrieved evidence.
"""

import re
from Core.app_text import app_text

REMEMBRANCER_SYSTEM = app_text["prompts"]["remembrancer_system"]
USER_TEMPLATE = app_text["prompts"]["user_template"]

# --- PRODUCTION CLEAN-TEXT & ANTI-REPETITION MANDATE ---
# This wrapper forces the LLM to output pristine text optimized for Text-to-Speech engines,
# preventing markdown artifacts, loops, backslashes, or bulleted lists.
CLEAN_TEXT_MANDATE = (
    "\n\n[STRICT PRODUCTION RULES FOR OUTPUT FORMATTING]\n"
    "1. DO NOT use any Markdown formatting characters under any circumstances. This includes headers (#, ##, ###), "
    "bolding (**text**), bullet points (- or *), and blockquotes (>).\n"
    "2. All structural text must be structured into natural, clean, flowing prose paragraphs separated only by normal spaces or simple line breaks.\n"
    "3. DO NOT repeat points, facts, or descriptions. If a piece of evidence from the context has been addressed, "
    "do not restate it or summarize it again in a subsequent paragraph.\n"
    "4. The output must be completely readable by Text-to-Speech (TTS) audio software. Avoid technical lists or outlines. "
    "Write numbers or symbols out as full words if necessary.\n"
    "5. Do not output raw control characters, escape slashes, backslashes, or internal scaffolding marks. Simple. Clean. Direct."
)


def _format_context_block(chunks: list[dict], include_viewpoint: bool = False) -> str:
    """
    Shared context block builder.
    When include_viewpoint=True, prepends a viewpoint hint derived from
    the source/chapter metadata so the model knows whose eyes each passage is from.
    """
    if not chunks:
        return app_text["prompts"]["empty_context"]

    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown source")
        chapter = chunk.get("chapter", "unknown chapter")
        stitch_range = chunk.get("stitch_range", "")
        text = chunk.get("text", "").strip()
        file_type = chunk.get("file_type", "")

        header = f"[Passage {i} - {source}"
        if chapter and chapter != "unknown":
            header += f", {chapter}"
        if stitch_range:
            header += f" ({stitch_range})"
        if file_type:
            header += f" [{file_type.upper()}]"
        header += "]"

        if include_viewpoint:
            hint = _infer_viewpoint(source, chapter, text)
            if hint:
                header += f"\n[Viewpoint: {hint}]"

        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


def _infer_viewpoint(source: str, chapter: str, text: str) -> str:
    """
    Derives a plain-English viewpoint hint from chunk metadata and text content.
    Used to tell the model whose perspective each passage is from, so it can
    signal transitions in the narration naturally.
    """
    text_lower = text.lower()
    source_lower = source.lower()

    # Named primarchs
    primarchs = [
        "fulgrim", "ferrus manus", "horus", "lorgar", "perturabo",
        "mortarion", "angron", "magnus", "vulkan", "corax", "dorn",
        "guilliman", "lion", "russ", "curze", "alpharius", "omegon",
        "sanguinius", "jonson",
    ]
    for p in primarchs:
        if p in text_lower:
            return f"Primarch viewpoint - {p.title()}"

    # Named legions in source
    legions = {
        "emperor's children": "Emperor's Children",
        "iron hands": "Iron Hands",
        "world eaters": "World Eaters",
        "death guard": "Death Guard",
        "thousand sons": "Thousand Sons",
        "luna wolves": "Luna Wolves",
        "sons of horus": "Sons of Horus",
        "night lords": "Night Lords",
        "iron warriors": "Iron Warriors",
        "raven guard": "Raven Guard",
        "salamanders": "Salamanders",
        "word bearers": "Word Bearers",
        "alpha legion": "Alpha Legion",
    }
    for key, label in legions.items():
        if key in source_lower or key in text_lower:
            return f"{label} Astartes"

    # Location signals
    if any(w in text_lower for w in ["bridge", "deck", "void", "orbit", "ship", "vessel", "frigate", "cruiser", "battleship"]):
        return "shipboard / orbital viewpoint"
    if any(w in text_lower for w in ["trench", "mud", "rubble", "crater", "surface", "drop pod", "landed"]):
        return "ground-level Astartes viewpoint"
    if any(w in text_lower for w in ["command", "vox", "strategium", "tactical display"]):
        return "command level viewpoint"

    # Audio drama
    if "audio" in source_lower:
        return "audio drama perspective"

    return ""


def build_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """Standard remembrancer prompt — grounded answer, no perspective signalling."""
    context_block = _format_context_block(chunks, include_viewpoint=False)
    system_prompt = REMEMBRANCER_SYSTEM.format(context=context_block) + CLEAN_TEXT_MANDATE
    user_message = USER_TEMPLATE.format(query=query)
    return system_prompt, user_message


def build_narrate_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Narration prompt — weaves all chunks into one flowing account,
    signals perspective shifts naturally in prose when the viewpoint changes.
    Use this when you want a full scene reconstruction across multiple sources.
    """
    context_block = _format_context_block(chunks, include_viewpoint=True)
    system_prompt = app_text["prompts"]["narrate_system"].format(context=context_block) + CLEAN_TEXT_MANDATE
    user_message = app_text["prompts"]["narrate_user_template"].format(query=query)
    return system_prompt, user_message


def build_object_explorer_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """Alternate prompt for object-oriented analysis."""
    system = app_text["prompts"]["object_explorer_system"]
    if not chunks:
        context_block = app_text["prompts"]["empty_object_context"]
    else:
        parts = [
            f"[{c.get('source', '?')} / {c.get('chapter', '?')}]\n{c.get('text', '').strip()}"
            for c in chunks
        ]
        context_block = "\n\n---\n\n".join(parts)

    system_prompt = system.format(context=context_block) + CLEAN_TEXT_MANDATE
    return system_prompt, app_text["prompts"]["object_query_template"].format(query=query)


def format_debug(query: str, chunks: list[dict], response: str) -> str:
    """Pretty debug output for CLI verbose mode."""
    sep = "-" * 70
    source_lines = []
    for i, c in enumerate(chunks, 1):
        score = (
            c.get("rerank_score")
            or c.get("query_overlap_score")
            or c.get("rrf_score")
            or c.get("faiss_score")
            or 0.0
        )
        rng = c.get("stitch_range", "")
        overlap = c.get("query_overlap_terms", [])
        viewpoint = _infer_viewpoint(c.get("source", ""), c.get("chapter", ""), c.get("text", ""))
        line = f"  [{i}] {c.get('source', '?')} / {c.get('chapter', '?')}"
        if rng:
            line += f"  ({rng})"
        line += f"  score={score:.4f}"
        if viewpoint:
            line += f"  [{viewpoint}]"
        if overlap:
            line += f"  overlap={','.join(overlap[:6])}"
        source_lines.append(line)

    return (
        f"\n{sep}\n"
        f"QUERY:   {query}\n"
        f"{sep}\n"
        f"SOURCES ({len(chunks)} chunks retrieved + stitched):\n"
        + "\n".join(source_lines)
        + "\n"
        f"{sep}\n"
        f"RESPONSE:\n{response}\n"
        f"{sep}\n"
    )