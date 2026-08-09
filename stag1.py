"""
horus_stage2_scene_builder.py — Stage 2: Score, Cluster, Tag
Memoria (TheForge) / Cogitator (OmnissiahCore) — Horus Heresy corpus

Reads heresy_faiss.index + heresy_chapter_text.db (chapter-level corpus,
NOT paragraph-level like the old WH40K pipeline), scores every chapter
against THREE independent anchor pools (mass-battle, confrontation,
primarch-speech), clusters adjacent high-scorers per book, tags each
scene with primarch/legion names found in its text, and writes results
into battle_scenes.db — using the SAME table names/columns as the
existing production schema, with only ADDITIVE new columns.

NEW ADDITIVE COLUMNS on `scenes` (created if missing, never renamed):
    characters          TEXT   — JSON array of primarch names found, e.g. '["Horus","Erebus"]'
    legions             TEXT   — JSON array of legion names found, e.g. '["Sons of Horus"]'
    has_primarch_speech INTEGER DEFAULT 0  — 1 if a primarch speaks in this scene,
                                              regardless of scene_type

scene_type values: 'battle', 'confrontation', 'speech' — a chapter can
land in more than one pool's clusters (e.g. a battle scene that ALSO
clears the speech pool becomes two related scene rows, both flagged
has_primarch_speech=1). Character/legion search for anyone NOT a
primarch/legion is intentionally NOT done here — that's handled by
building an FTS5 virtual table over scene_name/teaser/chronicle
separately (see setup_fts.py), so minor characters are still findable
via free-text search without a maintained manifest.

NO LLM calls in this phase — scene_name/teaser/chronicle/query_prompt
are left NULL here, same as the old pipeline's Phase 1/Phase 2 split.
Stage 3 (LLM weaving) fills those in afterward.

Run:
    python horus_stage2_scene_builder.py --db-dir . --scenes-db battle_scenes.db
"""

import json
import re
import sqlite3
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"  # MUST match corpus embedding model


# ─────────────────────────────────────────────────────────────────────────────
#  SCORING CONFIG — rescaled for CHAPTER-level granularity
#  (old pipeline scored paragraph-level chunks; this corpus is chapters,
#   ~10-60 per book instead of hundreds-thousands, so gap/separation/
#   scene-count constants are proportionally much smaller here)
# ─────────────────────────────────────────────────────────────────────────────
FAISS_WEIGHT = 0.60
BM25_WEIGHT  = 0.40

SCORE_GATE_MASS          = 0.30
SCORE_GATE_CONFRONTATION = 0.22
SCORE_GATE_SPEECH        = 0.20   # looser — the hard name+verb match below carries precision

BOOK_GATE       = 1     # min candidate chapters in book before skipping (chapters are coarse; even 1 strong chapter matters)
GAP_THRESHOLD   = 1     # chapter-id gap to still merge into one cluster (adjacent chapters only)
MIN_CLUSTER_LEN = 1     # a single strong chapter can BE a scene at this granularity
EMBED_BATCH     = 32

FAISS_TOP_K_MASS          = 3000
FAISS_TOP_K_CONFRONTATION = 3000
FAISS_TOP_K_SPEECH        = 3000

SCENE_SEPARATION_FRACTION = 0.10
SCENE_SEPARATION_MIN      = 2     # never require less than 2 chapters between scenes
SCENE_SEPARATION_MAX      = 15
SCENE_SEPARATION_BOOK_FRACTION_CAP = 0.20

MIN_CONFRONTATION_SLOTS = 1
MIN_SPEECH_SLOTS        = 1

# Word-distance window for the speech "hard gate": a primarch name/epithet
# must appear within this many words of a speaking verb for a chapter to
# qualify as a speech candidate, regardless of its anchor/BM25 score.
SPEECH_NAME_VERB_WINDOW = 12


# ─────────────────────────────────────────────────────────────────────────────
#  ANCHOR PHRASES
# ─────────────────────────────────────────────────────────────────────────────
MASS_BATTLE_ANCHORS = [
    "the lines broke and warriors charged into the fray",
    "bolter fire tore through the advancing ranks without mercy",
    "they crashed into the enemy formation with brutal unstoppable force",
    "the charge broke against the line of desperate defenders",
    "the battle raged across the ruins of the burning city",
    "the regiment held its ground against the overwhelming tide",
    "his chainsword screamed as he carved through the foe",
    "las-fire split the darkness as the assault began in earnest",
    "blood spilled across the ground as the fighting reached its peak",
    "void shields collapsed under sustained weapons fire from the fleet",
    "the fleet engaged and void war consumed the heavens above",
    "warriors teleported into the heart of the enemy position",
    "orbital bombardment had scarred the surface before the landing",
    "the boarding action was savage and close-quarters fighting filled every corridor",
    "the daemon prince descended and warriors scattered before its power",
    "the primarch himself led the assault against the enemy line",
    "the siege walls were failing under the relentless assault",
    "the gates fell and the defenders fell back fighting",
    "the ambush was sprung and warriors fell screaming into the kill zone",
    "the warband descended without warning on their prey",
    "the swarm surged forward and the firing lines opened up",
    "bioplasma bolts rained down from the sky above the compound",
    "the hive mind drove its creatures forward without pause or mercy",
]

CONFRONTATION_ANCHORS = [
    "he raised his blade and struck with all his strength",
    "the blow landed and staggered the enemy warrior backwards",
    "she drew her weapon and faced what stood before her",
    "they faced each other in silence before the first blow fell",
    "the confrontation had been inevitable since the moment he entered",
    "something ancient and terrible barred his path forward",
    "the guardian of the vault turned slowly to face the intruder",
    "he fought his way through the chamber toward his prize",
    "the creature moved faster than thought and struck him hard",
    "warp energy crackled as the entity manifested in the chamber",
]

PRIMARCH_SPEECH_ANCHORS = [
    "the primarch stood before his legion and spoke with absolute authority",
    "his words carried the weight of destiny as warriors listened in silence",
    "the declaration echoed through the assembled ranks of soldiers",
    "he raised his voice and commanded absolute attention from every warrior present",
    "the warmaster addressed his commanders with cold and measured words",
    "silence fell across the chamber as he began to speak",
    "his oration stirred something ancient in every warrior who heard it",
    "he spoke of betrayal and destiny in the same breath",
    "the primarch's voice was low but every word carried across the hall",
    "he turned to face his sons and began to speak",
]


# ─────────────────────────────────────────────────────────────────────────────
#  BM25 VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
_SHARED_VOCAB = [
    "fought", "fight", "attack", "assault", "charge", "strike", "struck",
    "charged", "fighting", "screaming", "fell", "slain", "kill", "killed",
    "retreat", "advance", "clash", "engage", "flanked", "broke",
    "roared", "bellowed", "sprinted", "leapt", "crashed", "tore", "ripped",
    "parried", "thrust", "blocked", "confronted", "drew",
    "blood", "wound", "death", "enemy", "foe",
]

MASS_BATTLE_VOCAB = _SHARED_VOCAB + [
    "battle", "combat", "siege", "war", "raid", "assault", "ambush",
    "volley", "barrage", "breakthrough", "engagement",
    "blade", "sword", "chainsword", "bolter", "lasgun", "plasma", "melta",
    "cannon", "artillery", "powerfist", "claws", "fangs", "talons",
    "warriors", "marines", "legion", "squad", "troops", "warband", "horde",
    "guardsman", "primarch", "captain", "sergeant", "commander", "champion",
    "dreadnought", "titan",
    "slaughter", "carnage", "massacre",
    "fire", "shot", "blast", "explode", "detonated", "torpedo",
    "void", "fleet", "broadside", "boarding", "teleport", "orbital",
    "bombardment", "spore", "swarm", "hive", "bioplasma",
]

CONFRONTATION_VOCAB = _SHARED_VOCAB + [
    "confrontation", "faced", "silence", "stood", "alone",
    "guardian", "ancient", "vault", "descended", "barred", "path",
    "creature", "entity", "daemon",
]

SPEECH_VOCAB = [
    "said", "spoke", "speak", "speaking", "declared", "proclaimed",
    "announced", "addressed", "command", "commanded", "voice", "words",
    "oration", "speech", "silence", "listened", "assembled", "gathered",
    "primarch", "warmaster", "legion", "brothers", "sons",
]


# ─────────────────────────────────────────────────────────────────────────────
#  PRIMARCH + LEGION NAME MATCHING
# ─────────────────────────────────────────────────────────────────────────────
PRIMARCH_NAMES: dict[str, list[str]] = {
    "Horus":         ["the Warmaster", "Lupercal"],
    "Lorgar":        ["the Urizen", "Aurelian"],
    "Guilliman":     ["Roboute Guilliman", "Lord of Ultramar"],
    "Perturabo":     ["the Lord of Iron"],
    "Mortarion":     ["the Death Lord", "the Pale King"],
    "Angron":        ["the Red Angel"],
    "Magnus":        ["the Crimson King", "the Red Cyclops"],
    "Fulgrim":       ["the Phoenician"],
    "Alpharius":     ["the Hydra", "Omegon"],
    "Curze":         ["Konrad Curze", "the Night Haunter"],
    "Dorn":          ["Rogal Dorn", "the Praetorian of Terra"],
    "Leman Russ":    ["the Wolf King", "the Great Wolf"],
    "Jaghatai Khan": ["the Warhawk", "the Khan"],
    "Sanguinius":    ["the Great Angel"],
    "Vulkan":        ["the Fire Lord"],
    "Ferrus Manus":  ["the Gorgon"],
    "Corax":         ["Corvus Corax", "the Raven Lord"],
    "Lion El'Jonson": ["the Lion"],
}

LEGION_NAMES = [
    "Luna Wolves", "Sons of Horus", "Emperor's Children", "Death Guard",
    "World Eaters", "Thousand Sons", "Space Wolves", "Dark Angels",
    "White Scars", "Imperial Fists", "Blood Angels", "Iron Hands",
    "Ultramarines", "Salamanders", "Raven Guard", "Alpha Legion",
    "Word Bearers", "Iron Warriors", "Night Lords",
]

# Compile all primarch aliases -> canonical name, for fast tagging
_PRIMARCH_ALIAS_TO_CANON: dict[str, str] = {}
for canon, aliases in PRIMARCH_NAMES.items():
    _PRIMARCH_ALIAS_TO_CANON[canon.lower()] = canon
    for alias in aliases:
        _PRIMARCH_ALIAS_TO_CANON[alias.lower()] = canon

_PRIMARCH_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(a) for a in _PRIMARCH_ALIAS_TO_CANON.keys()) + r')\b',
    re.IGNORECASE
)
_LEGION_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(l) for l in LEGION_NAMES) + r')\b',
    re.IGNORECASE
)
_SPEECH_VERB_PATTERN = re.compile(
    r'\b(said|spoke|speaking|declared|proclaimed|announced|addressed|wrote)\b',
    re.IGNORECASE
)


def tag_characters_and_legions(text: str) -> tuple[list[str], list[str]]:
    """Returns (primarch_names_found, legion_names_found) as sorted unique lists."""
    if not text:
        return [], []

    found_primarchs = set()
    for match in _PRIMARCH_PATTERN.finditer(text):
        canon = _PRIMARCH_ALIAS_TO_CANON.get(match.group(1).lower())
        if canon:
            found_primarchs.add(canon)

    found_legions = set()
    for match in _LEGION_PATTERN.finditer(text):
        found_legions.add(match.group(1))

    return sorted(found_primarchs), sorted(found_legions)


def detect_primarch_speech(text: str, window: int = SPEECH_NAME_VERB_WINDOW) -> bool:
    """
    Hard gate: True if a primarch name/epithet appears within `window` words
    of a speaking verb anywhere in the text. This is the precision anchor for
    the speech pool — semantic/BM25 score alone is not enough to qualify.
    """
    if not text:
        return False

    words = text.split()
    word_lower = [w.lower().strip('.,!?"\'') for w in words]

    primarch_positions = [
        i for i, w in enumerate(word_lower)
        if w in _PRIMARCH_ALIAS_TO_CANON
        or any(w == a.split()[0].lower() for a in _PRIMARCH_ALIAS_TO_CANON if ' ' in a)
    ]
    if not primarch_positions:
        return False

    verb_positions = [i for i, w in enumerate(word_lower) if _SPEECH_VERB_PATTERN.fullmatch(w)]
    if not verb_positions:
        return False

    for p_pos in primarch_positions:
        for v_pos in verb_positions:
            if abs(p_pos - v_pos) <= window:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  TITLE / BOOK ID HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def make_book_id(book_title: str) -> str:
    slug = book_title.lower()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    slug = slug.strip('_')
    return slug[:80]


# ─────────────────────────────────────────────────────────────────────────────
#  DB SETUP — additive columns only, reuses EXISTING production table names
# ─────────────────────────────────────────────────────────────────────────────
BASE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS books (
        book_id     TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        source_raw  TEXT NOT NULL UNIQUE,
        chunk_count INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS scenes (
        scene_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id         TEXT NOT NULL REFERENCES books(book_id),
        scene_name      TEXT,
        teaser          TEXT,
        chronicle       TEXT,
        query_prompt    TEXT,
        stitched_text   TEXT,
        scene_type      TEXT DEFAULT 'battle',
        score           REAL NOT NULL,
        rank            INTEGER NOT NULL,
        scene_key       TEXT,
        woven_at        TEXT
    );

    CREATE TABLE IF NOT EXISTS scene_chunks (
        scene_id    INTEGER REFERENCES scenes(scene_id),
        chunk_id    INTEGER NOT NULL,
        chunk_rank  INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS build_progress (
        source_raw    TEXT PRIMARY KEY,
        book_id       TEXT,
        chunk_count   INTEGER,
        scenes_found  INTEGER DEFAULT 0,
        top_score     REAL    DEFAULT 0.0,
        phase1_done   INTEGER DEFAULT 0,
        phase2_done   INTEGER DEFAULT 0,
        skipped       INTEGER DEFAULT 0,
        skip_reason   TEXT,
        processed_at  TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_scenes_key  ON scenes(scene_key);
    CREATE INDEX IF NOT EXISTS idx_scenes_type ON scenes(scene_type);
    CREATE INDEX IF NOT EXISTS idx_scenes_rank ON scenes(book_id, rank);
    CREATE INDEX IF NOT EXISTS idx_scenes_book ON scenes(book_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_scene ON scene_chunks(scene_id);
"""

ADDITIVE_COLUMNS = [
    ("characters", "TEXT"),
    ("legions", "TEXT"),
    ("has_primarch_speech", "INTEGER DEFAULT 0"),
]


def init_db(conn: sqlite3.Connection):
    conn.executescript(BASE_SCHEMA)
    for col_name, col_type in ADDITIVE_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE scenes ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists — safe no-op
    conn.commit()


def mark_skipped(conn, source_raw, book_id, chunk_count, reason, top_score=0.0):
    conn.execute("""
        INSERT OR REPLACE INTO build_progress
        (source_raw, book_id, chunk_count, top_score,
         phase1_done, skipped, skip_reason, processed_at)
        VALUES (?, ?, ?, ?, 1, 1, ?, ?)
    """, (source_raw, book_id, chunk_count, top_score, reason, datetime.utcnow().isoformat()))
    conn.commit()


def mark_phase1_done(conn, source_raw, book_id, chunk_count, scenes_found, top_score):
    conn.execute("""
        INSERT OR REPLACE INTO build_progress
        (source_raw, book_id, chunk_count, scenes_found,
         top_score, phase1_done, phase2_done, skipped, processed_at)
        VALUES (?, ?, ?, ?, ?, 1, 0, 0, ?)
    """, (source_raw, book_id, chunk_count, scenes_found, top_score, datetime.utcnow().isoformat()))
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  SCENE COUNT SCALING — rescaled for chapter-level books (10-60 chapters)
# ─────────────────────────────────────────────────────────────────────────────
def max_scenes_for_book(chapter_count: int) -> int:
    if chapter_count < 10:  return 2
    if chapter_count < 20:  return 3
    if chapter_count < 40:  return 4
    if chapter_count < 80:  return 5
    return 6


# ─────────────────────────────────────────────────────────────────────────────
#  CLUSTERING (same algorithm as old pipeline, chunk_id == embedding_index here)
# ─────────────────────────────────────────────────────────────────────────────
def cluster_candidates(candidates: list[dict], gap: int, min_len: int) -> list[list[dict]]:
    if not candidates:
        return []
    sorted_c = sorted(candidates, key=lambda x: x['chunk_id'])
    clusters, current = [], [sorted_c[0]]
    for chunk in sorted_c[1:]:
        if chunk['chunk_id'] - current[-1]['chunk_id'] <= gap:
            current.append(chunk)
        else:
            if len(current) >= min_len:
                clusters.append(current)
            current = [chunk]
    if len(current) >= min_len:
        clusters.append(current)
    return clusters


def _score_clusters(clusters: list[list[dict]]) -> list[dict]:
    scored = []
    for cluster in clusters:
        scores = [c['combined_score'] for c in cluster]
        avg_score = sum(scores) / len(scores)
        peak_score = max(scores)
        cluster_score = 0.4 * avg_score + 0.6 * peak_score
        scored.append({
            'chunks': cluster,
            'cluster_score': cluster_score,
            'chunk_count': len(cluster),
            'chunk_id_start': cluster[0]['chunk_id'],
            'chunk_id_end': cluster[-1]['chunk_id'],
            'scene_type': cluster[0].get('scene_type', 'battle'),
        })
    scored.sort(key=lambda x: x['cluster_score'], reverse=True)
    return scored


def compute_adaptive_separation(clusters: list[dict], book_chunk_count: int | None = None) -> int:
    if not clusters:
        return SCENE_SEPARATION_MIN
    span_start = min(c['chunk_id_start'] for c in clusters)
    span_end = max(c['chunk_id_end'] for c in clusters)
    total_span = max(1, span_end - span_start)
    separation = int(total_span * SCENE_SEPARATION_FRACTION)
    separation = max(SCENE_SEPARATION_MIN, min(SCENE_SEPARATION_MAX, separation))
    if book_chunk_count:
        book_cap = max(1, int(book_chunk_count * SCENE_SEPARATION_BOOK_FRACTION_CAP))
        separation = min(separation, book_cap)
    return separation


def select_diverse_clusters(clusters: list[list[dict]], max_count: int,
                             min_separation: int | None = None,
                             min_confrontation_slots: int = MIN_CONFRONTATION_SLOTS,
                             min_speech_slots: int = MIN_SPEECH_SLOTS,
                             book_chunk_count: int | None = None) -> list[dict]:
    if not clusters or max_count <= 0:
        return []

    scored = _score_clusters(clusters)
    if min_separation is None:
        min_separation = compute_adaptive_separation(scored, book_chunk_count)

    def overlaps_or_too_close(candidate, picked):
        for p in picked:
            if candidate['chunk_id_start'] <= p['chunk_id_end'] and candidate['chunk_id_end'] >= p['chunk_id_start']:
                return True
            gap = max(candidate['chunk_id_start'] - p['chunk_id_end'], p['chunk_id_start'] - candidate['chunk_id_end'])
            if gap < min_separation:
                return True
        return False

    selected: list[dict] = []

    # Reserved slots: confrontation, then speech
    for scene_type, min_slots in (('confrontation', min_confrontation_slots), ('speech', min_speech_slots)):
        if min_slots <= 0:
            continue
        ranked = [c for c in scored if c['scene_type'] == scene_type]
        for cand in ranked:
            if len([s for s in selected if s['scene_type'] == scene_type]) >= min_slots:
                break
            if len(selected) >= max_count:
                break
            if not overlaps_or_too_close(cand, selected):
                selected.append(cand)

    for cand in scored:
        if len(selected) >= max_count:
            break
        if cand in selected:
            continue
        if not overlaps_or_too_close(cand, selected):
            selected.append(cand)

    selected.sort(key=lambda x: x['chunk_id_start'])
    return selected


# ─────────────────────────────────────────────────────────────────────────────
#  STITCHING — simplified for chapter granularity (chapters are already
#  clean prose units, so gap-fill/dedup-overlap matter far less than at
#  paragraph granularity, but kept for correctness when clusters span
#  multiple chapters with a skipped one in between).
# ─────────────────────────────────────────────────────────────────────────────
def gap_fill_cluster(cluster_chunks: list[dict], chunk_lookup: dict[int, dict]) -> list[dict]:
    if not cluster_chunks:
        return cluster_chunks
    start_id = cluster_chunks[0]['chunk_id']
    end_id = cluster_chunks[-1]['chunk_id']
    existing_ids = {c['chunk_id'] for c in cluster_chunks}
    filled = []
    for cid in range(start_id, end_id + 1):
        if cid in existing_ids:
            match = next(c for c in cluster_chunks if c['chunk_id'] == cid)
            filled.append(match)
        elif cid in chunk_lookup:
            src = chunk_lookup[cid]
            filled.append({
                'chunk_id': cid, 'text': src['text'],
                'combined_score': 0.0, 'gap_filled': True,
            })
    return filled


def stitch_cluster(cluster_chunks: list[dict], chunk_lookup: dict[int, dict]) -> dict:
    filled = gap_fill_cluster(cluster_chunks, chunk_lookup)
    if not filled:
        return {'stitched_text': '', 'chunks': []}
    stitched_text = '\n\n'.join(c['text'].strip() for c in filled if c['text'].strip())
    return {'stitched_text': stitched_text, 'chunks': filled}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 2: Score, cluster, and tag Horus Heresy scenes")
    parser.add_argument('--db-dir', type=Path, default=Path('.'))
    parser.add_argument('--scenes-db', type=Path, default=Path('battle_scenes.db'))
    args = parser.parse_args()

    faiss_path = args.db_dir / "heresy_faiss.index"
    text_db_path = args.db_dir / "heresy_chapter_text.db"
    output_json = args.db_dir / "horus_clusters_raw.json"

    print("=" * 70)
    print("  Stage 2 — Score, Cluster, Tag  (Horus Heresy / Memoria)")
    print("=" * 70)

    # ── 1. Load FAISS index ─────────────────────────────────────────────────
    print(f"\n[1/6]  Loading FAISS index ...")
    index = faiss.read_index(str(faiss_path))
    print(f"       {index.ntotal:,} vectors, dim={index.d}")

    # ── 2. Load chapter text from SQLite ────────────────────────────────────
    print(f"\n[2/6]  Loading chapter text ...")
    text_conn = sqlite3.connect(str(text_db_path))
    text_conn.row_factory = sqlite3.Row
    rows = text_conn.execute("SELECT * FROM chapter_text").fetchall()
    metadata = [dict(r) for r in rows]
    text_conn.close()
    print(f"       {len(metadata):,} chapters loaded")

    # ── 3. Group by book (source_file) ──────────────────────────────────────
    print(f"\n[3/6]  Grouping by book ...")
    books: dict[str, list[dict]] = {}
    for chapter in metadata:
        src = chapter.get('source_file', 'unknown')
        books.setdefault(src, []).append(chapter)
    for src in books:
        books[src].sort(key=lambda c: c['chapter_number'])
    print(f"       {len(books):,} unique books found")

    # ── 4. Init DB ───────────────────────────────────────────────────────────
    conn = sqlite3.connect(str(args.scenes_db))
    init_db(conn)
    cur = conn.cursor()
    cur.execute("SELECT source_raw FROM build_progress WHERE phase1_done = 1")
    already_done = {row[0] for row in cur.fetchall()}
    books_todo = {src: chs for src, chs in books.items() if src not in already_done}
    print(f"       {len(already_done):,} already complete — skipping")
    print(f"       {len(books_todo):,} books to process this run")

    if not books_todo:
        print("\n  Nothing to do.")
        conn.close()
        return

    # ── 5. Embed anchors (SAME model as corpus!) ────────────────────────────
    print(f"\n[4/6]  Loading {EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    def embed_anchors(phrases):
        return embedder.encode(phrases, batch_size=EMBED_BATCH,
                                normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    print(f"       Embedding anchor pools ...")
    mass_vecs = embed_anchors(MASS_BATTLE_ANCHORS)
    conf_vecs = embed_anchors(CONFRONTATION_ANCHORS)
    speech_vecs = embed_anchors(PRIMARCH_SPEECH_ANCHORS)

    def faiss_score_pool(anchor_vecs, label, top_k):
        scores: dict[int, float] = {}
        for anchor_vec in tqdm(anchor_vecs, desc=f"  {label} search", unit="anchor"):
            vec = np.expand_dims(anchor_vec, axis=0)
            distances, indices = index.search(vec, top_k)
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                similarity = float(dist)  # IndexFlatIP -> inner product == cosine (normalized)
                if idx not in scores or scores[idx] < similarity:
                    scores[idx] = similarity
        if scores:
            vals = list(scores.values())
            v_min, v_max = min(vals), max(vals)
            v_range = v_max - v_min if v_max > v_min else 1.0
            scores = {k: (v - v_min) / v_range for k, v in scores.items()}
        return scores

    print(f"\n[5/6]  Searching FAISS across corpus ...")
    faiss_mass = faiss_score_pool(mass_vecs, "mass-battle", FAISS_TOP_K_MASS)
    faiss_conf = faiss_score_pool(conf_vecs, "confrontation", FAISS_TOP_K_CONFRONTATION)
    faiss_speech = faiss_score_pool(speech_vecs, "primarch-speech", FAISS_TOP_K_SPEECH)

    # ── 6. Score + cluster + tag each book ──────────────────────────────────
    print(f"\n[6/6]  Scoring, clustering, and tagging {len(books_todo):,} books ...")
    all_clusters: dict[str, dict] = {}
    stats = {'processed': 0, 'skip_gate': 0, 'total_scenes': 0}

    for source_raw, chapters in tqdm(books_todo.items(), desc="  Books", unit="book"):
        book_title = chapters[0]['book_title']
        book_id = make_book_id(book_title)
        chapter_count = len(chapters)

        chunk_lookup = {c['embedding_index']: {'chunk_id': c['embedding_index'], 'text': c['text']} for c in chapters}

        tokenized = [c['text'].lower().split() for c in chapters]
        bm25 = BM25Okapi(tokenized)

        bm25_mass = bm25.get_scores(MASS_BATTLE_VOCAB)
        bm25_conf = bm25.get_scores(CONFRONTATION_VOCAB)
        bm25_speech = bm25.get_scores(SPEECH_VOCAB)

        def norm(arr):
            m = float(arr.max())
            return (arr / m) if m > 0 else arr

        bm25_mass_n = norm(bm25_mass)
        bm25_conf_n = norm(bm25_conf)
        bm25_speech_n = norm(bm25_speech)

        candidates_mass, candidates_conf, candidates_speech = [], [], []

        for i, chapter in enumerate(chapters):
            cid = chapter['embedding_index']
            text = chapter['text']

            mass_score = BM25_WEIGHT * float(bm25_mass_n[i]) + FAISS_WEIGHT * faiss_mass.get(cid, 0.0)
            conf_score = BM25_WEIGHT * float(bm25_conf_n[i]) + FAISS_WEIGHT * faiss_conf.get(cid, 0.0)
            speech_score = BM25_WEIGHT * float(bm25_speech_n[i]) + FAISS_WEIGHT * faiss_speech.get(cid, 0.0)

            has_speech_hard_match = detect_primarch_speech(text)

            if mass_score >= SCORE_GATE_MASS:
                candidates_mass.append({'chunk_id': cid, 'text': text, 'combined_score': mass_score, 'scene_type': 'battle'})
            if conf_score >= SCORE_GATE_CONFRONTATION:
                candidates_conf.append({'chunk_id': cid, 'text': text, 'combined_score': conf_score, 'scene_type': 'confrontation'})
            if speech_score >= SCORE_GATE_SPEECH and has_speech_hard_match:
                candidates_speech.append({'chunk_id': cid, 'text': text, 'combined_score': speech_score, 'scene_type': 'speech'})

        all_candidates = candidates_mass + candidates_conf + candidates_speech
        if len(all_candidates) < BOOK_GATE:
            mark_skipped(conn, source_raw, book_id, chapter_count, 'score_gate')
            stats['skip_gate'] += 1
            continue

        max_s = max_scenes_for_book(chapter_count)

        raw_mass = cluster_candidates(candidates_mass, GAP_THRESHOLD, MIN_CLUSTER_LEN)
        raw_conf = cluster_candidates(candidates_conf, GAP_THRESHOLD, MIN_CLUSTER_LEN)
        raw_speech = cluster_candidates(candidates_speech, GAP_THRESHOLD, MIN_CLUSTER_LEN)

        all_raw_clusters = raw_mass + raw_conf + raw_speech
        top_clusters = select_diverse_clusters(
            all_raw_clusters, max_count=max_s,
            min_separation=None,
            min_confrontation_slots=MIN_CONFRONTATION_SLOTS,
            min_speech_slots=MIN_SPEECH_SLOTS,
            book_chunk_count=chapter_count,
        )

        if not top_clusters and all_candidates:
            top_candidates = sorted(all_candidates, key=lambda x: x['combined_score'], reverse=True)[:max_s]
            top_clusters = [{
                'chunks': [c], 'cluster_score': c['combined_score'], 'chunk_count': 1,
                'chunk_id_start': c['chunk_id'], 'chunk_id_end': c['chunk_id'],
                'scene_type': c.get('scene_type', 'battle'),
            } for c in top_candidates]

        if not top_clusters:
            mark_skipped(conn, source_raw, book_id, chapter_count, 'no_clusters_formed')
            stats['skip_gate'] += 1
            continue

        # Stitch + tag each cluster
        for cl in top_clusters:
            stitched = stitch_cluster(cl['chunks'], chunk_lookup)
            cl['chunks'] = stitched['chunks']
            cl['stitched_text'] = stitched['stitched_text']
            if cl['chunks']:
                cl['chunk_count'] = len(cl['chunks'])
                cl['chunk_id_start'] = cl['chunks'][0]['chunk_id']
                cl['chunk_id_end'] = cl['chunks'][-1]['chunk_id']

            characters, legions = tag_characters_and_legions(cl['stitched_text'])
            cl['characters'] = characters
            cl['legions'] = legions
            cl['has_primarch_speech'] = 1 if detect_primarch_speech(cl['stitched_text']) else 0

        # Upsert book
        conn.execute("""
            INSERT OR IGNORE INTO books (book_id, title, source_raw, chunk_count)
            VALUES (?, ?, ?, ?)
        """, (book_id, book_title, source_raw, chapter_count))

        scenes_found = len(top_clusters)
        top_score = max(cl['cluster_score'] for cl in top_clusters)
        mark_phase1_done(conn, source_raw, book_id, chapter_count, scenes_found, top_score)

        # Write scenes (scene_name/teaser/chronicle/query_prompt left NULL for Stage 3)
        for rank, cl in enumerate(top_clusters, start=1):
            scene_key = f"{source_raw}::rank{rank}::{cl['scene_type']}"
            cur.execute("""
                INSERT INTO scenes
                (book_id, scene_type, score, rank, scene_key, stitched_text,
                 characters, legions, has_primarch_speech, woven_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                book_id, cl['scene_type'], cl['cluster_score'], rank, scene_key,
                cl['stitched_text'], json.dumps(cl['characters']), json.dumps(cl['legions']),
                cl['has_primarch_speech'], datetime.utcnow().isoformat()
            ))
            scene_id = cur.lastrowid
            for chunk_rank, chunk in enumerate(cl['chunks']):
                cur.execute("""
                    INSERT INTO scene_chunks (scene_id, chunk_id, chunk_rank)
                    VALUES (?, ?, ?)
                """, (scene_id, chunk['chunk_id'], chunk_rank))
        conn.commit()

        all_clusters[source_raw] = {
            'book_id': book_id, 'title': book_title, 'source_raw': source_raw,
            'chunk_count': chapter_count,
            'scenes': [
                {
                    'rank': rank, 'cluster_score': cl['cluster_score'], 'scene_type': cl['scene_type'],
                    'chunk_id_start': cl['chunk_id_start'], 'chunk_id_end': cl['chunk_id_end'],
                    'chunk_count': cl['chunk_count'], 'stitched_text': cl['stitched_text'],
                    'characters': cl['characters'], 'legions': cl['legions'],
                    'has_primarch_speech': cl['has_primarch_speech'],
                    'chunk_ids': [c['chunk_id'] for c in cl['chunks']],
                }
                for rank, cl in enumerate(top_clusters, start=1)
            ],
        }
        stats['processed'] += 1
        stats['total_scenes'] += scenes_found

    print(f"\n  Writing {output_json.name} ...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_clusters, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("  Stage 2 Complete")
    print("=" * 70)
    print(f"  Books processed        : {stats['processed']:,}")
    print(f"  Books skipped          : {stats['skip_gate']:,}")
    print(f"  Total scenes clustered : {stats['total_scenes']:,}")
    print(f"\n  horus_clusters_raw.json  →  {output_json}")
    print(f"  scenes written to        →  {args.scenes_db}")
    print("=" * 70)
    print("\n  Next: Stage 3 — LLM weaving (scene_name/teaser/chronicle generation)")

    conn.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n  Wall time: {(time.time() - t0) / 60:.1f} minutes")