"""
phase2_build.py  —  Phase 2: LLM Weave & DB Write
OmnissiahCore Battle Scene Index Builder

Reads clusters_raw.json, calls LM Studio once per scene with a FRESH
context window each time (no history between calls), and writes the
woven result to battle_scenes.db immediately after each call.

Crash-safe: tracks completion PER SCENE, not per book. Every restart
skips exactly the scenes already written, regardless of whether their
parent book is fully complete. No scene is ever re-woven on restart.

Run:
    python phase2_build.py

Force re-weave everything (e.g. after prompt change):
    python phase2_build.py --reweave
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
CLUSTERS_PATH = BASE_DIR / "clusters_raw.json"
SQLITE_PATH   = BASE_DIR / "battle_scenes.db"

# ── LM Studio config ──────────────────────────────────────────────────────────
DEFAULT_URL   = "http://192.168.100.71:1234"
DEFAULT_MODEL = "google/gemma-4-e4b"
TIMEOUT       = 700

# Maximum characters of stitched_text sent to the LLM per scene.
# Full text is always priority — we send as much as fits.
MAX_INPUT_CHARS = 80_000

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--url',     type=str,  default=DEFAULT_URL)
parser.add_argument('--reweave', action='store_true',
                    help='Re-weave all scenes (clears DB scenes table first)')
args = parser.parse_args()

LM_STUDIO_URL   = args.url.rstrip('/')
CHAT_ENDPOINT   = f"{LM_STUDIO_URL}/v1/chat/completions"
MODELS_ENDPOINT = f"{LM_STUDIO_URL}/v1/models"

# ── DB Schema ─────────────────────────────────────────────────────────────────
SCHEMA = """
    CREATE TABLE IF NOT EXISTS books (
        book_id     TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        source_raw  TEXT NOT NULL UNIQUE,
        chunk_count INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS scenes (
        scene_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id       TEXT NOT NULL REFERENCES books(book_id),
        scene_key     TEXT NOT NULL UNIQUE,
        scene_name    TEXT,
        teaser        TEXT,
        chronicle     TEXT,
        query_prompt  TEXT,
        stitched_text TEXT,
        scene_type    TEXT DEFAULT 'battle',
        score         REAL NOT NULL,
        rank          INTEGER NOT NULL,
        woven_at      TEXT
    );

    CREATE TABLE IF NOT EXISTS scene_chunks (
        scene_id   INTEGER REFERENCES scenes(scene_id),
        chunk_id   INTEGER NOT NULL,
        chunk_rank INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS build_progress (
        source_raw   TEXT PRIMARY KEY,
        book_id      TEXT,
        chunk_count  INTEGER,
        scenes_found INTEGER DEFAULT 0,
        top_score    REAL    DEFAULT 0.0,
        phase1_done  INTEGER DEFAULT 0,
        phase2_done  INTEGER DEFAULT 0,
        skipped      INTEGER DEFAULT 0,
        skip_reason  TEXT,
        processed_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_scenes_book  ON scenes(book_id);
    CREATE INDEX IF NOT EXISTS idx_scenes_key   ON scenes(scene_key);
    CREATE INDEX IF NOT EXISTS idx_scenes_type  ON scenes(scene_type);
    CREATE INDEX IF NOT EXISTS idx_chunks_scene ON scene_chunks(scene_id);
"""

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the Remembrancer of the Imperium — a scholar charged \
with chronicling the battles and confrontations of the Warhammer 40,000 universe \
and its related settings (Warhammer Fantasy, Age of Sigmar, and other Black Library fiction).

You will receive raw text fragments extracted from a Black Library novel. \
These fragments may contain minor encoding artefacts (stray numbers, unusual \
characters, spacing noise from PDF extraction). Ignore all artefacts and \
work only from the narrative content.

Your output MUST follow this EXACT structure with these EXACT labels on their own lines:

SCENE_NAME: [A specific, evocative name. Name it as a historian would.
             Examples: "The Betrayal at Isstvan III", "Duel in the Vault of Molech",
             "The Last Stand of Shadrac". Never use generic names like "Battle Scene".]

TEASER: [One sentence. The hook shown on the scene card before the user clicks.
         Make it vivid and specific.]

CHRONICLE:
[8-10 substantial paragraphs, each at least 150 words. Weave the fragments into
one rich, immersive, flowing account. Begin at the moment of action and carry
through to the resolution. Describe the combatants, the environment, the stakes,
the turning points, and the aftermath as present in the fragments.
Do not invent facts not in the fragments, but expand on atmosphere, tension,
and physical detail that the fragments imply. Write clean prose only.]

QUERY_PROMPT: [One sentence a reader could send to a Warhammer lore AI to learn
               more about this specific scene or its participants.
               Example: "Tell me more about Karl's encounter with the beastmen
               in Haldedorf and what it reveals about his character."]"""


def make_user_message(book_title: str, scene_type: str, stitched_text: str) -> str:
    type_hint = (
        "This is an intimate confrontation, duel, or guardian encounter — "
        "not a mass battle. Chronicle it with focus on the individuals involved."
        if scene_type == 'confrontation'
        else
        "This is a mass battle, siege, boarding action, or large-scale military engagement."
    )
    text = stitched_text[:MAX_INPUT_CHARS]
    truncated = len(stitched_text) > MAX_INPUT_CHARS
    suffix = (
        f"\n\n[Note: {len(stitched_text)-MAX_INPUT_CHARS:,} additional chars omitted "
        f"to fit context. Chronicle what is here.]"
        if truncated else ""
    )
    return (
        f"Book: {book_title}\n"
        f"Scene type: {type_hint}\n\n"
        f"FRAGMENTS:\n{text}{suffix}\n\n"
        f"Name, weave, and chronicle this scene."
    )


# ── LM Studio ─────────────────────────────────────────────────────────────────
def get_active_model() -> str | None:
    try:
        r = requests.get(MODELS_ENDPOINT, timeout=10)
        models = r.json().get('data', [])
        ids = [m['id'] for m in models if 'embed' not in m['id'].lower()]
        if DEFAULT_MODEL in ids:
            return DEFAULT_MODEL
        return ids[0] if ids else None
    except Exception as e:
        print(f"  [warn] Could not query models: {e}")
        return None


def call_lm_studio(model: str, book_title: str, scene_type: str,
                   stitched_text: str) -> dict:
    """Stateless call — exactly two messages, no history, full context per scene."""
    payload = {
        "model":       model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_user_message(
                book_title, scene_type, stitched_text)},
        ],
        "temperature": 0.3,
        "max_tokens":  -1,
        "stream":      False,
    }
    t0 = time.time()
    r = requests.post(CHAT_ENDPOINT, json=payload, timeout=TIMEOUT)
    elapsed = time.time() - t0

    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}", response=r)

    data = r.json()
    return {
        'content':     data['choices'][0]['message']['content'],
        'elapsed':     elapsed,
        'tokens_used': data.get('usage', {}),
    }


def parse_llm_output(content: str) -> dict:
    result = {'scene_name': '', 'teaser': '', 'chronicle': '', 'query_prompt': ''}
    lines = content.splitlines()
    current, buffer = None, []

    def flush(field, buf):
        result[field] = '\n'.join(buf).strip()

    for line in lines:
        s = line.strip()
        matched = False
        for label, field in [('SCENE_NAME:', 'scene_name'), ('TEASER:', 'teaser'),
                              ('CHRONICLE:', 'chronicle'), ('QUERY_PROMPT:', 'query_prompt')]:
            if s.startswith(label):
                if current and buffer:
                    flush(current, buffer)
                current = field
                buffer = [s[len(label):].strip()]
                matched = True
                break
        if not matched and current:
            buffer.append(line)

    if current and buffer:
        flush(current, buffer)
    return result


# ── DB helpers ────────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.commit()


def get_done_scene_keys(conn: sqlite3.Connection) -> set:
    """
    Returns the set of scene_keys already written to the DB.
    scene_key = "{source_raw}::rank{rank}" — unique per scene across the corpus.
    This is the ONLY thing checked on restart — if the key exists, skip it.
    No dependence on build_progress or per-book state.
    """
    cur = conn.execute("SELECT scene_key FROM scenes WHERE scene_key IS NOT NULL")
    return {row[0] for row in cur.fetchall()}


def make_scene_key(source_raw: str, rank: int) -> str:
    return f"{source_raw}::rank{rank}"


def write_scene(conn: sqlite3.Connection, book_id: str, book_title: str,
                source_raw: str, chunk_count: int, scene: dict,
                parsed: dict, stitched_text: str):
    """Writes book (if missing) + scene + chunks in one transaction."""
    scene_key = make_scene_key(source_raw, scene['rank'])

    conn.execute("""
        INSERT OR IGNORE INTO books (book_id, title, source_raw, chunk_count)
        VALUES (?, ?, ?, ?)
    """, (book_id, book_title, source_raw, chunk_count))

    cur = conn.execute("""
        INSERT OR REPLACE INTO scenes
        (book_id, scene_key, scene_name, teaser, chronicle, query_prompt,
         stitched_text, scene_type, score, rank, woven_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        book_id, scene_key,
        parsed['scene_name'], parsed['teaser'],
        parsed['chronicle'],  parsed['query_prompt'],
        stitched_text,
        scene.get('scene_type', 'battle'),
        scene['cluster_score'],
        scene['rank'],
        datetime.utcnow().isoformat(),
    ))
    scene_id = cur.lastrowid

    if scene.get('chunk_ids'):
        conn.execute("DELETE FROM scene_chunks WHERE scene_id = ?", (scene_id,))
        conn.executemany("""
            INSERT INTO scene_chunks (scene_id, chunk_id, chunk_rank)
            VALUES (?, ?, ?)
        """, [(scene_id, cid, rank)
              for rank, cid in enumerate(scene['chunk_ids'])])

    conn.commit()


def update_book_phase2(conn: sqlite3.Connection, source_raw: str,
                        done_keys: set, all_scene_keys: list):
    """Mark a book as phase2_done only when every one of its scenes is written."""
    if all(k in done_keys for k in all_scene_keys):
        conn.execute("""
            UPDATE build_progress SET phase2_done = 1, processed_at = ?
            WHERE source_raw = ?
        """, (datetime.utcnow().isoformat(), source_raw))
        conn.commit()


def print_db_summary(conn: sqlite3.Connection):
    row = conn.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN phase2_done=1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN phase1_done=1 AND phase2_done=0
                         AND skipped=0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN skipped=1 THEN 1 ELSE 0 END)
        FROM build_progress
    """).fetchone()
    scene_count = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
    if row and row[0]:
        print(f"\n  ── DB Snapshot ──────────────────────────────────")
        print(f"     Total tracked    : {row[0]:,}")
        print(f"     Phase 2 complete : {row[1]:,}")
        print(f"     Awaiting Phase 2 : {row[2]:,}")
        print(f"     Skipped (gate)   : {row[3]:,}")
        print(f"     Scenes in DB     : {scene_count:,}")
        print(f"  ────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  OmnissiahCore  —  Phase 2: LLM Weave & DB Write")
    print("  Battle Scene Index Builder")
    print("=" * 70)

    # ── LM Studio check ───────────────────────────────────────────────────────
    print(f"\n[1/4]  Connecting to LM Studio at {LM_STUDIO_URL} ...")
    model = get_active_model()
    if not model:
        print("ERROR: No model found. Make sure LM Studio is running.")
        sys.exit(1)
    print(f"       Model   : {model}")
    if model != DEFAULT_MODEL:
        print(f"       [WARN] Expected '{DEFAULT_MODEL}' — make sure the right model is loaded")
    print(f"       Timeout : {TIMEOUT}s per call")
    print(f"       Max input: {MAX_INPUT_CHARS:,} chars (~{MAX_INPUT_CHARS//3:,} tokens)")

    # ── Load clusters ─────────────────────────────────────────────────────────
    print(f"\n[2/4]  Loading {CLUSTERS_PATH.name} ...")
    if not CLUSTERS_PATH.exists():
        print(f"ERROR: {CLUSTERS_PATH} not found. Run battle.py first.")
        sys.exit(1)
    with open(CLUSTERS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"       {len(data):,} books loaded")

    # ── Init DB ───────────────────────────────────────────────────────────────
    print(f"\n[3/4]  Initialising battle_scenes.db ...")
    conn = sqlite3.connect(str(SQLITE_PATH))
    init_db(conn)

    if args.reweave:
        print("       --reweave: clearing all scenes ...")
        conn.execute("DELETE FROM scene_chunks")
        conn.execute("DELETE FROM scenes")
        conn.execute("UPDATE build_progress SET phase2_done = 0")
        conn.commit()

    # KEY CHANGE: track done state per scene_key, not per book
    done_keys = get_done_scene_keys(conn)
    print(f"       {len(done_keys):,} scenes already woven — skipping those specifically")

    # Build todo: all scenes whose key isn't in done_keys, best-score first
    todo: list[dict] = []
    for source_raw, book in data.items():
        for scene in book['scenes']:
            key = make_scene_key(source_raw, scene['rank'])
            if key in done_keys:
                continue
            stitched = scene.get('stitched_text', '').strip()
            if not stitched:
                continue
            todo.append({
                'source_raw':  source_raw,
                'book_id':     book['book_id'],
                'book_title':  book['title'],
                'chunk_count': book['chunk_count'],
                'scene':       scene,
                'stitched':    stitched,
                'scene_key':   key,
                'score':       scene['cluster_score'],
                # all keys for this book, for phase2_done marking
                'all_book_keys': [
                    make_scene_key(source_raw, s['rank'])
                    for s in book['scenes']
                ],
            })

    todo.sort(key=lambda x: x['score'], reverse=True)
    print(f"       {len(todo):,} scenes to weave this run")

    if not todo:
        print("\n  Nothing to do — all scenes already woven.")
        print_db_summary(conn)
        conn.close()
        return

    # ── Weave ─────────────────────────────────────────────────────────────────
    print(f"\n[4/4]  Weaving {len(todo):,} scenes ...")
    print(f"       Best-first — kill any time, restart resumes exactly here\n")

    stats = {'done': 0, 'errors': 0, 'total_time': 0.0}

    for i, item in enumerate(todo):
        scene      = item['scene']
        scene_type = scene.get('scene_type', 'battle')

        print(f"  [{i+1:>5}/{len(todo)}]  {item['book_title'][:50]}  "
              f"(rank {scene['rank']}, score={item['score']:.3f}, type={scene_type})")

        try:
            result = call_lm_studio(
                model=model,
                book_title=item['book_title'],
                scene_type=scene_type,
                stitched_text=item['stitched'],
            )
            parsed  = parse_llm_output(result['content'])
            elapsed = result['elapsed']
            stats['total_time'] += elapsed

            missing = [f for f in ['scene_name', 'teaser', 'chronicle', 'query_prompt']
                       if not parsed[f]]
            if missing:
                print(f"          [WARN] Missing fields: {missing}")

            # Write to DB immediately after each call
            write_scene(conn, item['book_id'], item['book_title'],
                        item['source_raw'], item['chunk_count'],
                        scene, parsed, item['stitched'])

            # Update done_keys so the book-completion check is accurate
            done_keys.add(item['scene_key'])

            # Mark book done only when every one of its scenes is in done_keys
            update_book_phase2(conn, item['source_raw'],
                               done_keys, item['all_book_keys'])

            print(f"          → \"{parsed['scene_name']}\"  [{elapsed:.1f}s]")
            stats['done'] += 1

        except requests.exceptions.Timeout:
            print(f"          [TIMEOUT] Skipping after {TIMEOUT}s")
            stats['errors'] += 1

        except requests.HTTPError as e:
            err_str = str(e)
            print(f"          [HTTP ERROR] {err_str[:120]}")
            # If model was unloaded, stop immediately — all subsequent calls
            # will fail too. User needs to reload model in LM Studio.
            if 'unloaded' in err_str.lower() or 'no model' in err_str.lower():
                print("\n  [FATAL] Model unloaded in LM Studio.")
                print("  Reload the model and restart this script.")
                print("  All progress saved — will resume from next unwoven scene.")
                break
            stats['errors'] += 1

        except requests.exceptions.ConnectionError:
            print(f"\n  [CONNECTION LOST] LM Studio unreachable.")
            print(f"  Progress saved — restart to continue from next unwoven scene.")
            break

        except Exception as e:
            print(f"          [ERROR] {e}")
            stats['errors'] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print_db_summary(conn)

    avg = stats['total_time'] / stats['done'] if stats['done'] else 0
    remaining = len(todo) - stats['done'] - stats['errors']

    print(f"\n{'='*70}")
    print(f"  Phase 2 — Run Complete")
    print(f"{'='*70}")
    print(f"  Scenes woven this run  : {stats['done']:,}")
    print(f"  Errors / skipped       : {stats['errors']:,}")
    print(f"  Avg time per scene     : {avg:.1f}s")
    if remaining > 0:
        est = (remaining * avg) / 3600
        print(f"  Remaining scenes       : {remaining:,}  (~{est:.1f}h at current speed)")
        print(f"\n  Run phase2_build.py again to continue.")
    else:
        print(f"\n  All scenes woven. battle_scenes.db is ready for TheForge.")
    print(f"{'='*70}")
    conn.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\n  Wall time: {elapsed / 60:.1f} minutes")