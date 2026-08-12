"""
horus_stage3_weave.py — Stage 3: Light-Edit Weave & DB Write (local LM Studio)
Memoria (TheForge) / Cogitator (OmnissiahCore)

REWRITTEN around a chunked light-edit approach after discovering the
single-shot "chronicle" generation was silently summarizing scenes —
a 1,200-word passage with real dialogue collapsed to 4 sentences.

APPROACH:
  1. Strip extraction noise (TOC lines, catalog numbers, headers) from
     the START of stitched_text only, structurally (not by memorizing
     exact strings — see clean_extraction_noise), never touching the
     middle of the text.
  2. Split the cleaned text into ~1,000-1,200 word chunks on paragraph
     boundaries (never mid-sentence).
  3. Light-edit EACH CHUNK SEPARATELY — a narrow, easy task local models
     handle reliably: fix noise/awkward breaks, preserve every line of
     dialogue and detail. Never asked to compress.
  4. Word-retention safeguard PER CHUNK: if an edited chunk comes back
     under RETENTION_RATIO of the input chunk's word count, retry with a
     stronger instruction; if it still fails, keep the ORIGINAL chunk
     text verbatim rather than accept a lossy result. Content is never
     silently lost.
  5. Concatenate edited chunks -> final chronicle.
  6. One short separate call generates scene_name/teaser/query_prompt
     from a preview of the finished chronicle (kept short by design).

Run:
    python horus_stage3_weave.py --scenes-db battle_scenes.db --model "gemma-2-9b-it" --limit 5
    python horus_stage3_weave.py --status
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


CHECKPOINT_PATH = Path("stage3_checkpoint.json")
DEFAULT_URL = "http://localhost:1234"
TIMEOUT_DEFAULT = 300
CHUNK_TARGET_WORDS = 1100
RETENTION_RATIO = 0.85   # "bigger the better" — strict: edited chunk must keep 85%+ of original word count
MAX_CHUNK_RETRIES = 2


# ─────────────────────────────────────────────────────────────────────────────
#  NOISE STRIPPING — structural, boundary-only, never touches the middle
# ─────────────────────────────────────────────────────────────────────────────
_JUNK_LINE_PATTERNS = [
    re.compile(r'^\s*CONTENTS\b', re.IGNORECASE),
    re.compile(r'^\s*\d+\.\d+\s*\(\d{4}\.\d+\)\s*$'),          # "1.3 (2012.01)"
    re.compile(r'^\s*[\d\.\-–]+\s*$'),                          # bare numbers/dashes
    re.compile(r'^\s*[A-Z][A-Z\s&\'\-]{6,}$'),                  # long ALL-CAPS runs (titles/headers)
    re.compile(r'^\s*(ISBN|Copyright|All rights reserved)\b', re.IGNORECASE),
]

_SENTENCE_START_RE = re.compile(r'^["\u201c]?[A-Z][a-z]')


def _looks_like_junk_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if len(stripped) < 60 and any(p.match(stripped) for p in _JUNK_LINE_PATTERNS):
        return True
    return False


def clean_extraction_noise(text: str, window_chars: int = 800) -> str:
    """
    Walks forward from the start of the text, stripping junk-shaped lines,
    until it hits a line that actually looks like prose (capitalized word
    followed by lowercase text). Only operates within the first
    `window_chars` — bounded and conservative, so it can never eat real
    scene content further in.
    """
    head = text[:window_chars]
    rest = text[window_chars:]

    lines = head.split('\n')
    kept_from = 0
    for i, line in enumerate(lines):
        if _looks_like_junk_line(line):
            continue
        if _SENTENCE_START_RE.match(line.strip()) or len(line.strip()) > 80:
            kept_from = i
            break
        kept_from = i + 1
    else:
        kept_from = len(lines)

    cleaned_head = '\n'.join(lines[kept_from:])
    return (cleaned_head + rest).strip()


# ─────────────────────────────────────────────────────────────────────────────
#  CHUNKING — split on paragraph boundaries, never mid-sentence
# ─────────────────────────────────────────────────────────────────────────────
def split_into_chunks(text: str, target_words: int = CHUNK_TARGET_WORDS) -> list[str]:
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks, current, current_words = [], [], 0
    for para in paragraphs:
        para_words = len(para.split())
        if current and current_words + para_words > target_words:
            chunks.append('\n\n'.join(current))
            current, current_words = [], 0
        current.append(para)
        current_words += para_words

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  AUTHOR EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def split_author_title(book_title: str) -> tuple[str | None, str]:
    if '  ' in book_title:
        parts = book_title.split('  ', 1)
        author = parts[0].strip()
        title = parts[1].strip(' -–_')
        if 0 < len(author) < 40 and title:
            return author, title
    return None, book_title


def strip_reasoning(raw: str) -> str:
    return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()


# ─────────────────────────────────────────────────────────────────────────────
#  LIGHT-EDIT PROMPT (per chunk — narrow task, easy to get right)
# ─────────────────────────────────────────────────────────────────────────────
LIGHT_EDIT_SYSTEM_PROMPT = """You are a light copy-editor for Horus Heresy novel excerpts. \
You will receive a passage of narrative text, extracted from a physical book, which may contain \
minor extraction artefacts (broken line breaks, stray characters, odd spacing).

Your task is NARROW: lightly clean the passage. You are NOT summarizing, condensing, or \
retelling. Preserve EVERY sentence, EVERY line of dialogue, EVERY detail. Only:
  - Fix broken line breaks and spacing artefacts
  - Smooth an awkward sentence break caused by extraction, if one exists
  - Correct an obviously mangled word from bad text extraction

Do NOT cut any dialogue. Do NOT shorten descriptions. Do NOT remove any character's lines. \
Preserve the author's exact wording and voice wherever possible — this is light editing, not \
rewriting. The output should be close to the same length as the input.

Output ONLY the cleaned passage text. No commentary, no labels, no preamble."""


def light_edit_chunk(chat_endpoint: str, model: str, timeout: int, chunk: str,
                      author: str | None) -> str:
    author_note = f"\n\nAuthor: {author} — preserve their exact voice." if author else ""
    user_message = f"Lightly clean this passage. Preserve everything.{author_note}\n\n{chunk}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LIGHT_EDIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": -1,
        "stream": False,
    }
    r = requests.post(chat_endpoint, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}", response=r)
    content = r.json()['choices'][0]['message']['content']
    return strip_reasoning(content).strip()


def light_edit_with_safeguard(chat_endpoint: str, model: str, timeout: int, chunk: str,
                               author: str | None) -> tuple[str, bool]:
    """Returns (final_text, was_fallback). Falls back to the original chunk
    verbatim if the model can't produce an edit that retains enough content."""
    original_words = len(chunk.split())

    for attempt in range(MAX_CHUNK_RETRIES + 1):
        try:
            edited = light_edit_chunk(chat_endpoint, model, timeout, chunk, author)
        except Exception:
            continue

        edited_words = len(edited.split())
        if original_words == 0 or edited_words >= RETENTION_RATIO * original_words:
            return edited, False

    return chunk, True


# ─────────────────────────────────────────────────────────────────────────────
#  SHORT-FIELD GENERATION (scene_name/teaser/query_prompt) — labeled text
# ─────────────────────────────────────────────────────────────────────────────
SHORT_FIELDS_SYSTEM_PROMPT = """You are the Remembrancer of the Imperium, naming and framing \
Horus Heresy scenes for a database. Given a scene's chronicle text, produce exactly this \
structure with these EXACT labels on their own lines:

SCENE_NAME: [A specific, evocative name, as a historian would write it. Never generic.]

TEASER: [One sentence, PRESENT TENSE, following this pattern: name the specific named
         character(s) actually present in the text, state the threat or stakes they face,
         and end on tension WITHOUT revealing the outcome. This is a hook, not a summary.

         GOOD: "Cornered by a monstrous alien horror in the collapsing tunnels, Greel must
         fight for survival as betrayal closes in from his own company."
         GOOD: "Against a tide of desperate cultists, the Imperial Fists fight for the breach
         that will save thousands — if it can be held."
         BAD (flat statement, no tension, no named character): "A battle happens between two
         forces and one side wins after heavy fighting."
         BAD (reveals the ending): "The Luna Wolves win the battle without losses."]

QUERY_PROMPT: [One sentence a reader could send to a Warhammer lore AI to learn more about
               this specific scene or its participants.]

CRITICAL — GROUNDING RULE: Every named character, place, or faction you mention MUST appear
literally in the chronicle text below. Do NOT invent names, relationships, motivations, or
plot details not explicitly present. If you are unsure a detail is accurate, leave it out
rather than guess or infer it.

Output ONLY these three labeled sections. No commentary."""


# ─────────────────────────────────────────────────────────────────────────────
#  GROUNDING CHECK — catches fabricated names the model wasn't given
# ─────────────────────────────────────────────────────────────────────────────
_PROPER_NOUN_RE = re.compile(r'\b[A-Z][a-z]{2,}\b')

_COMMON_SENTENCE_STARTERS = {
    'The', 'A', 'An', 'His', 'Her', 'Their', 'They', 'This', 'That', 'When',
    'As', 'For', 'In', 'On', 'With', 'Against', 'Beneath', 'Cornered',
    'Amidst', 'Beyond', 'After', 'Before', 'Now', 'Only', 'One', 'It',
}


def find_ungrounded_names(generated_text: str, source_text: str) -> list[str]:
    """
    Extracts capitalized-word candidates (likely proper nouns) from generated
    text and returns any that don't appear anywhere in the source chronicle —
    i.e. names the model invented rather than pulled from the actual scene.
    Heuristic, not perfect, but catches the common fabrication pattern.
    """
    candidates = set(_PROPER_NOUN_RE.findall(generated_text)) - _COMMON_SENTENCE_STARTERS
    source_lower = source_text.lower()
    return [c for c in candidates if c.lower() not in source_lower]


def _call_short_fields(chat_endpoint: str, model: str, timeout: int, user_message: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SHORT_FIELDS_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.6,
        "max_tokens": -1,
        "stream": False,
    }
    r = requests.post(chat_endpoint, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}", response=r)
    content = strip_reasoning(r.json()['choices'][0]['message']['content'])

    result = {'scene_name': '', 'teaser': '', 'query_prompt': ''}
    current, buffer = None, []

    def flush(field, buf):
        result[field] = '\n'.join(buf).strip()

    for line in content.splitlines():
        s = line.strip()
        matched = False
        for label, field in [('SCENE_NAME:', 'scene_name'), ('TEASER:', 'teaser'),
                              ('QUERY_PROMPT:', 'query_prompt')]:
            if s.startswith(label):
                if current and buffer:
                    flush(current, buffer)
                current, buffer, matched = field, [s[len(label):].strip()], True
                break
        if not matched and current:
            buffer.append(line)
    if current and buffer:
        flush(current, buffer)

    return result


def generate_short_fields(chat_endpoint: str, model: str, timeout: int, book_title: str,
                           scene_type: str, chronicle: str, max_retries: int = 2) -> tuple[dict, list[str]]:
    """
    Returns (result, ungrounded_names_on_final_attempt). Retries if the model
    invents a name not present anywhere in the actual chronicle text — this
    is what catches hallucination regardless of model size, since it checks
    the OUTPUT against the SOURCE rather than trusting the model to behave.
    """
    _, clean_title = split_author_title(book_title)
    base_message = f"Book: {clean_title}\nScene type: {scene_type}\n\nChronicle:\n{chronicle[:6000]}"

    warning = ""
    for attempt in range(max_retries + 1):
        result = _call_short_fields(chat_endpoint, model, timeout, base_message + warning)
        check_text = f"{result['teaser']} {result['query_prompt']}"  # scene_name excluded: title language ≠ factual claim
        ungrounded = find_ungrounded_names(check_text, chronicle)

        if not ungrounded:
            return result, []

        warning = (f"\n\nWARNING: your previous attempt invented these names, which do NOT "
                   f"appear in the chronicle: {', '.join(ungrounded)}. Do not use any name "
                   f"that isn't literally present in the chronicle text above.")

    return result, ungrounded  # exhausted retries — caller decides whether to accept or flag


# ─────────────────────────────────────────────────────────────────────────────
#  LM STUDIO HELPERS + CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def get_active_model(models_endpoint: str, requested_model: str) -> str | None:
    try:
        r = requests.get(models_endpoint, timeout=10)
        ids = [m['id'] for m in r.json().get('data', []) if 'embed' not in m['id'].lower()]
        return requested_model if requested_model in ids else (ids[0] if ids else None)
    except Exception as e:
        print(f"  [warn] Could not query models: {e}")
        return None


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding='utf-8'))
    return {"last_scene_id": None, "woven": 0, "failed": 0, "fallback_chunks": 0, "last_updated": None}


def save_checkpoint(cp: dict):
    cp["last_updated"] = datetime.utcnow().isoformat()
    CHECKPOINT_PATH.write_text(json.dumps(cp, indent=2), encoding='utf-8')


def print_status(scenes_db: Path):
    conn = sqlite3.connect(str(scenes_db))
    total = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
    woven = conn.execute("SELECT COUNT(*) FROM scenes WHERE chronicle IS NOT NULL AND chronicle != ''").fetchone()[0]
    conn.close()
    cp = load_checkpoint()
    print("=" * 70)
    print("  Stage 3 Status")
    print("=" * 70)
    print(f"  Total scenes : {total:,}   Woven: {woven:,}   Remaining: {total - woven:,}")
    print(f"  Last checkpoint scene_id : {cp['last_scene_id']}")
    print(f"  Checkpoint woven/failed  : {cp['woven']}/{cp['failed']}")
    print(f"  Chunks that fell back to original text: {cp.get('fallback_chunks', 0)}")
    print(f"  Last updated             : {cp['last_updated']}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 3: chunked light-edit weaving (local LM Studio)")
    parser.add_argument('--scenes-db', type=Path, default=Path('battle_scenes.db'))
    parser.add_argument('--url', type=str, default=DEFAULT_URL)
    parser.add_argument('--model', type=str, default='local-model')
    parser.add_argument('--timeout', type=int, default=TIMEOUT_DEFAULT)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()

    if args.status:
        print_status(args.scenes_db)
        return

    lm_url = args.url.rstrip('/')
    chat_endpoint = f"{lm_url}/v1/chat/completions"
    models_endpoint = f"{lm_url}/v1/models"

    print("=" * 70)
    print("  Stage 3 — Chunked Light-Edit Weave  (local LM Studio)")
    print("=" * 70)

    print(f"\n[1/3]  Connecting to LM Studio at {lm_url} ...")
    model = get_active_model(models_endpoint, args.model)
    if not model:
        print("ERROR: No model found. Make sure LM Studio's local server is running.")
        sys.exit(1)
    print(f"       Model: {model}")
    if model != args.model:
        print(f"       [WARN] Requested '{args.model}' not loaded — using '{model}' instead")

    conn = sqlite3.connect(str(args.scenes_db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
        SELECT s.scene_id, s.stitched_text, s.scene_type, s.score, b.title AS book_title
        FROM scenes s JOIN books b ON s.book_id = b.book_id
        WHERE (s.chronicle IS NULL OR s.chronicle = '')
          AND s.stitched_text IS NOT NULL AND s.stitched_text != ''
        ORDER BY s.score DESC
    """
    if args.limit:
        query += f" LIMIT {args.limit}"
    rows = cur.execute(query).fetchall()

    print(f"\n[2/3]  {len(rows):,} scenes to weave this run")
    if not rows:
        print("\n  Nothing to do.")
        conn.close()
        return

    print(f"\n[3/3]  Weaving (chunked light-edit, retention safeguard={RETENTION_RATIO:.0%}) ...\n")

    cp = load_checkpoint()
    woven, failed, fallback_chunks = cp["woven"], cp["failed"], cp.get("fallback_chunks", 0)

    for i, row in enumerate(rows, start=1):
        scene = dict(row)
        author, _ = split_author_title(scene["book_title"])

        print(f"  [{i:>4}/{len(rows)}]  {scene['book_title'][:50]}  (scene_id={scene['scene_id']})")

        cleaned = clean_extraction_noise(scene["stitched_text"])
        chunks = split_into_chunks(cleaned)
        print(f"          {len(chunks)} chunk(s), {len(cleaned.split()):,} words total")

        edited_chunks = []
        for c_idx, chunk in enumerate(chunks):
            try:
                edited, was_fallback = light_edit_with_safeguard(chat_endpoint, model, args.timeout, chunk, author)
                edited_chunks.append(edited)
                if was_fallback:
                    fallback_chunks += 1
                    print(f"          chunk {c_idx+1}/{len(chunks)}: retention safeguard triggered — kept original text")
            except requests.exceptions.ConnectionError:
                print(f"\n  [CONNECTION LOST] LM Studio unreachable. Progress saved at scene_id={scene['scene_id']}.")
                save_checkpoint({"last_scene_id": scene['scene_id'], "woven": woven, "failed": failed,
                                  "fallback_chunks": fallback_chunks, "last_updated": None})
                conn.close()
                return
            except Exception as e:
                print(f"          chunk {c_idx+1}/{len(chunks)}: ERROR ({e}) — keeping original text")
                edited_chunks.append(chunk)
                fallback_chunks += 1

        chronicle = '\n\n'.join(edited_chunks)

        try:
            short_fields, ungrounded = generate_short_fields(chat_endpoint, model, args.timeout,
                                                               scene["book_title"], scene["scene_type"], chronicle)
            if ungrounded:
                print(f"          [WARN] possible fabricated names after retries: {', '.join(ungrounded)}")
        except Exception as e:
            print(f"          [WARN] short-field generation failed ({e}) — using placeholder title")
            short_fields = {'scene_name': scene['book_title'][:60], 'teaser': '', 'query_prompt': ''}

        cur.execute("""
            UPDATE scenes SET scene_name = ?, teaser = ?, chronicle = ?, query_prompt = ?, woven_at = ?
            WHERE scene_id = ?
        """, (short_fields['scene_name'], short_fields['teaser'], chronicle,
              short_fields['query_prompt'], datetime.utcnow().isoformat(), scene['scene_id']))
        conn.commit()
        woven += 1

        print(f"          → \"{short_fields['scene_name']}\"  ({len(chronicle.split()):,} words)")

        save_checkpoint({"last_scene_id": scene["scene_id"], "woven": woven, "failed": failed,
                          "fallback_chunks": fallback_chunks, "last_updated": None})

    print("\n" + "=" * 70)
    print("  Stage 3 — Run Complete")
    print("=" * 70)
    print(f"  Scenes woven          : {woven:,}")
    print(f"  Chunks using fallback  : {fallback_chunks:,}  (original text kept — model couldn't retain content)")
    print("=" * 70)
    conn.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Wall time: {(time.time() - t0) / 60:.1f} minutes")