"""
phase2_verify.py — LLM weave verification before the full Phase 2 run.

Pulls the top N highest-scoring scenes from clusters_raw.json, calls
LM Studio once per scene with a FRESH context window each time (no
conversation history carried between calls), and prints the raw LLM
output so you can read it and judge quality before committing anything
to the database.

NOTHING is written to battle_scenes.db during this script.

Usage:
    python phase2_verify.py                    # top 5 scenes
    python phase2_verify.py --n 10             # verify 10 scenes
    python phase2_verify.py --type confrontation
    python phase2_verify.py --url http://192.168.100.71:1234

Context length is set in LM Studio's UI, not passed as a parameter.
The script truncates stitched text so the full prompt stays within
MAX_INPUT_CHARS (configurable below) to avoid overflowing the context.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
# How many characters of stitched text to send to the LLM at most.
# Priority is always the FULL text — we send as much as fits.
# At ~3 chars/token, 60000 chars ≈ 20k tokens, leaving room for
# system prompt + generated output inside a 32k context window.
# Raise this if your model has a larger context set in LM Studio.
MAX_INPUT_CHARS = 60_000

DEFAULT_URL     = 'http://192.168.100.71:1234'
DEFAULT_MODEL   = 'google/gemma-4-e4b'
DEFAULT_TIMEOUT = 300

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n',       type=int, default=5)
parser.add_argument('--url',     type=str, default=DEFAULT_URL)
parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT)
parser.add_argument('--type',    type=str, default=None, dest='scene_type')
args = parser.parse_args()

CLUSTERS_PATH  = Path(__file__).parent / 'clusters_raw.json'
LM_STUDIO_URL  = args.url.rstrip('/')
CHAT_ENDPOINT  = f"{LM_STUDIO_URL}/v1/chat/completions"
MODELS_ENDPOINT = f"{LM_STUDIO_URL}/v1/models"

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the Remembrancer of the Imperium — a scholar charged with \
chronicling the battles and confrontations of the Warhammer 40,000 universe.

You will receive raw text fragments extracted from a Black Library novel. \
These fragments may contain minor encoding artefacts (stray numbers, unusual \
characters, formatting noise from PDF extraction). Ignore all artefacts and \
work only from the narrative content.

Your output MUST follow this EXACT structure with these EXACT labels:

SCENE_NAME: [A specific evocative name. Examples: "The Betrayal at Isstvan III", \
"Duel in the Vault of Molech". Never use generic names like "Battle Scene".]

TEASER: [One sentence. The hook that makes someone want to read this scene.]

CHRONICLE:
[3-4 paragraphs in the voice of Imperial historical record. Begin at the moment \
of action. Do not invent facts not in the fragments. Write clean prose only.]

QUERY_PROMPT: [One sentence phrased as a question a reader would ask to learn \
about this scene. Example: "What happened during the Luna Wolves assault on \
the Whisperhead Mountains?"]"""


def make_user_message(book_title: str, scene_type: str, stitched_text: str) -> str:
    type_hint = (
        "This is an intimate confrontation, duel, or guardian encounter."
        if scene_type == 'confrontation'
        else "This is a mass battle, siege, or large-scale engagement."
    )
    # Truncate stitched text to stay within context budget.
    # We send as much as possible — the full text is always priority.
    text = stitched_text[:MAX_INPUT_CHARS]
    truncated = len(stitched_text) > MAX_INPUT_CHARS
    suffix = f"\n\n[Note: {len(stitched_text) - MAX_INPUT_CHARS:,} chars truncated to fit context]" \
             if truncated else ""
    return (
        f"Book: {book_title}\n"
        f"Scene type: {type_hint}\n\n"
        f"FRAGMENTS:\n{text}{suffix}\n\n"
        f"Identify, name, and chronicle this scene."
    )


# ── LM Studio ─────────────────────────────────────────────────────────────────
def get_loaded_model() -> str | None:
    try:
        r = requests.get(MODELS_ENDPOINT, timeout=10)
        models = r.json().get('data', [])
        # Prefer the configured default model if it's in the list
        ids = [m['id'] for m in models]
        if DEFAULT_MODEL in ids:
            return DEFAULT_MODEL
        # Otherwise return the first non-embedding model
        for m in models:
            if 'embed' not in m['id'].lower():
                return m['id']
        return ids[0] if ids else None
    except Exception as e:
        print(f"  [warn] Could not query models: {e}")
        return None


def call_lm_studio(model: str, book_title: str, scene_type: str,
                   stitched_text: str, timeout: int) -> dict:
    """
    Single stateless LLM call. Exactly two messages — system + user.
    No num_ctx parameter (set context in LM Studio UI).
    No conversation history from any prior call.
    """
    payload = {
        "model":       model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_user_message(
                book_title, scene_type, stitched_text)},
        ],
        "temperature": 0.3,
        "max_tokens":  -1,    # let the model finish naturally
        "stream":      False,
    }
    t0 = time.time()
    response = requests.post(CHAT_ENDPOINT, json=payload, timeout=timeout)
    elapsed = time.time() - t0

    if response.status_code != 200:
        raise requests.HTTPError(
            f"{response.status_code}: {response.text[:300]}", response=response)

    data = response.json()
    content = data['choices'][0]['message']['content']
    return {
        'content':     content,
        'elapsed':     elapsed,
        'tokens_used': data.get('usage', {}),
    }


def parse_llm_output(content: str) -> dict:
    result = {'scene_name': '', 'teaser': '', 'chronicle': '', 'query_prompt': '', 'raw': content}
    lines = content.splitlines()
    current_field, buffer = None, []

    def flush(field, buf):
        result[field] = '\n'.join(buf).strip()

    for line in lines:
        s = line.strip()
        for label, field in [('SCENE_NAME:', 'scene_name'), ('TEASER:', 'teaser'),
                              ('CHRONICLE:', 'chronicle'), ('QUERY_PROMPT:', 'query_prompt')]:
            if s.startswith(label):
                if current_field and buffer:
                    flush(current_field, buffer)
                current_field = field
                buffer = [s[len(label):].strip()]
                break
        else:
            if current_field is not None:
                buffer.append(line)

    if current_field and buffer:
        flush(current_field, buffer)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Phase 2 Verify — LLM weave preview (no DB writes)")
    print("=" * 70)

    print(f"\nConnecting to LM Studio at {LM_STUDIO_URL} ...")
    model = get_loaded_model()
    if not model:
        print("ERROR: No model found. Make sure LM Studio is running with a model loaded.")
        sys.exit(1)

    if model != DEFAULT_MODEL:
        print(f"  [WARN] Expected '{DEFAULT_MODEL}' but found '{model}'")
        print(f"  Switch to {DEFAULT_MODEL} in LM Studio for best results.")
        print(f"  Continuing with: {model}")
    else:
        print(f"  Model  : {model} ✓")

    print(f"  MAX_INPUT_CHARS : {MAX_INPUT_CHARS:,}  (~{MAX_INPUT_CHARS//3:,} tokens of input)")
    print(f"  Timeout: {args.timeout}s per call")
    print(f"  Context length  : set in LM Studio UI (not passed via API)")

    print(f"\nLoading {CLUSTERS_PATH.name} ...")
    with open(CLUSTERS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_scenes = []
    for source_raw, book in data.items():
        for scene in book['scenes']:
            if args.scene_type and scene.get('scene_type') != args.scene_type:
                continue
            stitched = scene.get('stitched_text', '')
            if not stitched.strip():
                continue
            all_scenes.append({
                'book_title': book['title'],
                'score':      scene['cluster_score'],
                'scene_type': scene.get('scene_type', 'battle'),
                'span_start': scene['chunk_id_start'],
                'span_end':   scene['chunk_id_end'],
                'stitched':   stitched,
                'rank':       scene['rank'],
            })

    all_scenes.sort(key=lambda s: s['score'], reverse=True)
    to_verify = all_scenes[:args.n]

    print(f"  Total scenes: {len(all_scenes):,}  |  Verifying: {len(to_verify)}")

    for i, scene in enumerate(to_verify):
        print(f"\n{'='*70}")
        print(f"SCENE {i+1}/{len(to_verify)}: {scene['book_title']}")
        print(f"  type={scene['scene_type']}  score={scene['score']:.3f}  "
              f"span={scene['span_start']}-{scene['span_end']}")
        print(f"{'='*70}")

        stitched = scene['stitched']
        chars_sent = min(len(stitched), MAX_INPUT_CHARS)
        print(f"\n--- STITCHED INPUT: {len(stitched):,} chars total, "
              f"{chars_sent:,} sent to LLM ---")
        print(stitched[:400])
        if len(stitched) > 400:
            print(f"  ... [preview truncated — full {chars_sent:,} chars sent to LLM]")

        print(f"\n--- LLM OUTPUT ---")
        try:
            result = call_lm_studio(
                model=model,
                book_title=scene['book_title'],
                scene_type=scene['scene_type'],
                stitched_text=stitched,
                timeout=args.timeout,
            )
            parsed = parse_llm_output(result['content'])

            print(f"\nSCENE_NAME   : {parsed['scene_name']}")
            print(f"TEASER       : {parsed['teaser']}")
            print(f"\nCHRONICLE:\n{parsed['chronicle']}")
            print(f"\nQUERY_PROMPT : {parsed['query_prompt']}")
            print(f"\n[{result['elapsed']:.1f}s | {result['tokens_used']}]")

            missing = [f for f in ['scene_name','teaser','chronicle','query_prompt']
                       if not parsed[f]]
            print(f"  [{'OK — all fields parsed' if not missing else 'WARN — missing: ' + str(missing)}]")

        except requests.HTTPError as e:
            print(f"  [ERROR] HTTP {e}")
            print("  Common causes: model not loaded, or LM Studio rejected the request.")
        except requests.exceptions.Timeout:
            print(f"  [ERROR] Timed out after {args.timeout}s — try a smaller --n or longer --timeout")
        except requests.exceptions.ConnectionError:
            print(f"  [ERROR] Cannot connect to {LM_STUDIO_URL}")
            break
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\n{'='*70}")
    print("  Verify complete — nothing written to battle_scenes.db")
    print("  If output looks good: run phase2_build.py")
    print("=" * 70)


if __name__ == "__main__":
    main()