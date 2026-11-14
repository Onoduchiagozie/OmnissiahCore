"""
scene_cluster_builder.py  —  Phase 1: Score & Cluster
OmnissiahCore Battle Scene Index Builder

Reads metadata.json + faiss.index, scores every chunk against combat
signals (BM25 + FAISS cosine), clusters adjacent high-scorers by book,
writes clusters_raw.json and populates build_progress in battle_scenes.db.

NO LLM calls in this phase. FAISS search runs on CPU (faiss-cpu is
installed; GPU FAISS on Windows requires a conda-based faiss-gpu setup
that hasn't been done here, and the FAISS search step is already fast
enough — single-digit seconds — that it isn't the bottleneck anyway).
The book-by-book BM25 scoring/clustering loop is the bulk of runtime and
is CPU-only regardless, since rank_bm25 has no GPU path. Crash-safe
checkpointing — safe to kill and restart at any point.

Assumptions:
  - FAISS index was built sequentially: index position i == chunk_id i
  - metadata.json is a flat list of {chunk_id, source, text}
  - BAAI/bge-m3 is already cached locally (offline mode safe)

Run:
    python scene_cluster_builder.py

Place this file alongside your Db/ folder, or adjust BASE_DIR below.
"""

import json
import re
import sqlite3
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS  —  adjust BASE_DIR if script is not next to Db/
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DB_DIR        = BASE_DIR / "Db"
METADATA_PATH = DB_DIR / "metadata.json"
FAISS_PATH    = DB_DIR / "faiss.index"
OUTPUT_JSON   = BASE_DIR / "clusters_raw.json"
SQLITE_PATH   = BASE_DIR / "battle_scenes.db"


# ─────────────────────────────────────────────────────────────────────────────
#  SCORING CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FAISS_WEIGHT    = 0.60   # weight for semantic (FAISS) score
BM25_WEIGHT     = 0.40   # weight for keyword (BM25) score

# SCORE_GATE: min combined score for a chunk to be a candidate.
# Confrontation chunks get their own (lower) gate because intimate,
# one-on-one scenes are inherently sparser in this corpus than mass
# combat — gating them at the same threshold as mass-battle would
# systematically exclude genuine confrontation scenes that simply never
# reach mass-battle-level keyword/semantic density. This is intentional,
# not a relaxed standard — it's why the confrontation reservation slot
# exists at all.
SCORE_GATE_MASS          = 0.30
SCORE_GATE_CONFRONTATION = 0.22
# Back-compat alias — some helper functions/tests reference SCORE_GATE
# directly; keep it pointing at the mass threshold (the original value).
SCORE_GATE = SCORE_GATE_MASS

BOOK_GATE       = 3      # min candidate chunks in book before skipping entirely
GAP_THRESHOLD   = 10     # max chunk_id gap to still be considered same cluster
MIN_CLUSTER_LEN = 2      # discard clusters shorter than this
EMBED_BATCH     = 32     # batch size for anchor embedding

# FAISS_TOP_K: top-K chunks retrieved from FAISS PER ANCHOR. This is the
# main lever controlling what fraction of the corpus gets ANY semantic
# score at all — a chunk that doesn't land in any anchor's top-K silently
# defaults to faiss_score=0.0 via .get(cid, 0.0), regardless of how good a
# match it might actually be. Splitting one 33-anchor pool into smaller
# mass/confrontation pools shrinks total coverage unless top_k is raised
# to compensate — confrontation in particular needs a higher per-anchor
# top_k since it only has 10 anchors carrying the whole pool's coverage.
# FAISS_TOP_K: top-K chunks retrieved from FAISS PER ANCHOR. This is the
# main lever controlling what fraction of the corpus gets ANY semantic
# score at all — a chunk that doesn't land in any anchor's top-K silently
# defaults to faiss_score=0.0 via .get(cid, 0.0), regardless of how good a
# match it might actually be.
#
# The original single 33-anchor pool at top_k=6000 covered ~110,386 chunks
# (22.1% of the 498,850-chunk corpus) — confirmed from an actual run.
# Splitting into 19 mass + 10 confrontation anchors shrinks total
# anchor-slot budget unless top_k is raised to compensate, and
# confrontation in particular needs a much higher per-anchor top_k since
# it carries the whole pool's coverage on only 10 anchors. These values
# are set so each pool's total anchor-slot budget (anchors x top_k) is at
# or above the original pool's 198,000, rather than a guessed partial fix.
FAISS_TOP_K_MASS          = 11000   # 19 anchors x 11000 = 209,000 slots
FAISS_TOP_K_CONFRONTATION = 20000   # 10 anchors x 20000 = 200,000 slots
# Back-compat alias for any code still referencing the old single constant.
FAISS_TOP_K = FAISS_TOP_K_MASS

# Spread enforcement is now ADAPTIVE per book (see compute_adaptive_separation
# below) rather than a single fixed chunk-id distance. A fixed number can't
# work across this corpus: short stories may have all their action packed
# into a few hundred chunks, while omnibuses can spread genuinely distinct
# battles tens of thousands of chunks apart. A separation tuned for one
# shape either collapses every scene into 1 (too large for a tight book)
# or lets near-duplicate fragments of the same battle pass as "distinct"
# scenes (too small for a sprawling one). These constants define how the
# adaptive value is derived, not a separation distance itself.
SCENE_SEPARATION_FRACTION = 0.06   # target: ~6% of the book's candidate span
SCENE_SEPARATION_MIN      = 15     # never require less than this many chunks
SCENE_SEPARATION_MAX      = 800    # never require more than this many chunks

# Separate cap based on the book's TOTAL length (not just candidate span).
# Without this, SCENE_SEPARATION_MIN can swallow a disproportionate chunk
# of a genuinely short book (e.g. 15 chunks is ~32% of a 47-chunk book),
# silently preventing max_scenes_for_book from ever being reached even
# when the book has a legitimate second scene. This only ever lowers the
# computed separation for short books — it never raises it for long ones.
SCENE_SEPARATION_BOOK_FRACTION_CAP = 0.10

# Within the scenes ultimately chosen for a book, try to reserve at least
# this many slots for confrontation-type clusters (intimate, one-on-one,
# guardian/duel scenes) even if their raw score loses to mass-battle
# clusters elsewhere in the book. Set to 0 to disable the reservation and
# fall back to pure score ranking across both categories.
MIN_CONFRONTATION_SLOTS = 1


# ─────────────────────────────────────────────────────────────────────────────
#  BATTLE ANCHOR PHRASES  —  split into two scored-separately categories
#
#  Mass-battle and intimate-confrontation language compete for the same
#  FAISS similarity budget if pooled together, and mass-battle language is
#  both more common and denser in this corpus, so it was winning every
#  scene slot even in books (like Vengeful Spirit) that contain a genuine
#  quiet confrontation scene. Scoring them as two separate pools, each
#  max-pooled independently, lets a confrontation scene compete on its own
#  terms instead of against fleet battles and siege walls.
# ─────────────────────────────────────────────────────────────────────────────
MASS_BATTLE_ANCHORS = [
    # Mass engagement
    "the lines broke and warriors charged into the fray",
    "bolter fire tore through the advancing ranks without mercy",
    "they crashed into the enemy formation with brutal unstoppable force",
    "the charge broke against the line of desperate defenders",
    "the battle raged across the ruins of the burning city",
    "the regiment held its ground against the overwhelming tide",

    # Weapons and violence (battlefield-scale)
    "his chainsword screamed as he carved through the foe",
    "las-fire split the darkness as the assault began in earnest",
    "blood spilled across the ground as the fighting reached its peak",

    # Space and void war
    "void shields collapsed under sustained weapons fire from the fleet",
    "the fleet engaged and void war consumed the heavens above",
    "warriors teleported into the heart of the enemy position",
    "orbital bombardment had scarred the surface before the landing",
    "the boarding action was savage and close-quarters fighting filled every corridor",

    # Daemons and supernaturals (large-scale manifestation)
    "the daemon prince descended and warriors scattered before its power",
    "the primarch himself led the assault against the enemy line",

    # Siege and fortification
    "the siege walls were failing under the relentless assault",
    "the gates fell and the defenders fell back fighting",
    "the ambush was sprung and warriors fell screaming into the kill zone",
    "the warband descended without warning on their prey",

    # Tyranid and alien (swarm-scale)
    "the swarm surged forward and the firing lines opened up",
    "bioplasma bolts rained down from the sky above the compound",
    "the hive mind drove its creatures forward without pause or mercy",
]

CONFRONTATION_ANCHORS = [
    # One-on-one duels and stand-offs
    "he raised his blade and struck with all his strength",
    "the blow landed and staggered the enemy warrior backwards",
    "she drew her weapon and faced what stood before her",
    "they faced each other in silence before the first blow fell",
    "the confrontation had been inevitable since the moment he entered",

    # Intimate guardian / vault encounters (catches Molech-type scenes)
    "something ancient and terrible barred his path forward",
    "the guardian of the vault turned slowly to face the intruder",
    "he fought his way through the chamber toward his prize",
    "the creature moved faster than thought and struck him hard",
    "warp energy crackled as the entity manifested in the chamber",
]

# Kept for any code path that still wants the full combined list
# (e.g. quick diagnostics) — not used for scoring directly anymore.
BATTLE_ANCHORS = MASS_BATTLE_ANCHORS + CONFRONTATION_ANCHORS


# ─────────────────────────────────────────────────────────────────────────────
#  BM25 COMBAT VOCABULARY  —  split to match the anchor split above.
#  Generic violence verbs apply to either scene type and are shared; nouns
#  that signal SCALE (fleet, regiment, horde, barrage) go to mass-battle
#  only, and nouns that signal an INTIMATE one-on-one beat (guardian,
#  vault, confrontation) go to confrontation only. Without this split,
#  BM25's half of the combined score still drowned confrontation chunks
#  in mass-battle vocabulary density even after the FAISS anchors split.
# ─────────────────────────────────────────────────────────────────────────────
_SHARED_VOCAB = [
    # Core verbs — apply equally to a duel or a war
    "fought", "fight", "attack", "assault", "charge", "strike", "struck",
    "charged", "fighting", "screaming", "fell", "slain", "kill", "killed",
    "retreat", "advance", "clash", "engage", "flanked", "broke",
    "roared", "bellowed", "sprinted", "leapt", "crashed", "tore", "ripped",
    "parried", "thrust", "blocked", "confronted", "drew",

    # Casualties and violence — scale-agnostic
    "blood", "wound", "death", "enemy", "foe",
]

MASS_BATTLE_VOCAB = _SHARED_VOCAB + [
    # Battle nouns (scale)
    "battle", "combat", "siege", "war", "raid", "assault", "ambush",
    "volley", "barrage", "breakthrough", "engagement",

    # Weapons (battlefield/army issue)
    "blade", "sword", "chainsword", "bolter", "lasgun", "plasma", "melta",
    "cannon", "artillery", "powerfist", "claws", "fangs", "talons",

    # Units and people in battle (army-scale)
    "warriors", "marines", "legion", "squad", "troops", "warband", "horde",
    "guardsman", "primarch", "captain", "sergeant", "commander", "champion",
    "dreadnought", "titan",

    # Mass casualties
    "slaughter", "carnage", "massacre",

    # Ranged / ordnance
    "fire", "shot", "blast", "explode", "detonated", "torpedo",

    # Space battle specific
    "void", "fleet", "broadside", "boarding", "teleport", "orbital",
    "bombardment", "spore", "swarm", "hive", "bioplasma",
]

CONFRONTATION_VOCAB = _SHARED_VOCAB + [
    # Duel / stand-off framing
    "confrontation", "faced", "silence", "stood", "alone",

    # Guardian / vault / intimate encounter (catches Molech-type scenes)
    "guardian", "ancient", "vault", "descended", "barred", "path",
    "creature", "entity", "daemon",
]

# Kept for any code path that still wants the full combined vocabulary.
COMBAT_VOCAB = list(dict.fromkeys(MASS_BATTLE_VOCAB + CONFRONTATION_VOCAB))


# ─────────────────────────────────────────────────────────────────────────────
#  TITLE CLEANING
# ─────────────────────────────────────────────────────────────────────────────
# Pattern order matters — strip outer noise first, then find the real title.
_STRIP_PATTERNS = [
    (r'\.(pdf|txt|epub|mobi|doc)$',            ''),   # extension
    (r'^[\d]+\.?[\d]*\s+',                     ''),   # leading seq nums "0.5 "
    (r'^(WH40K|Warhammer\s*40[,\s]?0{3}?'
     r'|WarHammer)\s*[-–]\s*',                 ''),   # "WH40K - "
    (r'[\[\(][^\]\)]{1,40}[\]\)]',             ''),   # "[extra]" "(Short Story)"
    (r'^\s*(short|graphic.novel|novel|omnibus'
     r'|anthology|codex|supplement)\s*[-–]\s*',''),   # type tags
    # Author name "Firstname Lastname - " pattern (2-3 word name)
    (r'^[A-Z][a-záéíóú\-]+ '
     r'(?:[A-Z][a-záéíóú\-]+ )?'
     r'[A-Z][a-záéíóú\-]+\s*[-–]\s*',         ''),
    (r'[-–_]+$',                               ''),   # trailing dashes
    (r'\s+',                                   ' '),  # multiple spaces
]


def clean_title(source_raw: str) -> str:
    title = source_raw
    for pattern, replacement in _STRIP_PATTERNS:
        title = re.sub(pattern, replacement, title, flags=re.IGNORECASE)
    title = title.strip(' -–_')
    return title if title else source_raw


def make_book_id(source_raw: str) -> str:
    slug = clean_title(source_raw).lower()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    slug = slug.strip('_')
    return slug[:80]


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE SETUP
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA = """
    CREATE TABLE IF NOT EXISTS books (
        book_id     TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        source_raw  TEXT NOT NULL UNIQUE,
        chunk_count INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS scenes (
        scene_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id         TEXT NOT NULL REFERENCES books(book_id),
        title           TEXT,
        teaser          TEXT,
        query_prompt    TEXT,
        full_narration  TEXT,
        scene_type      TEXT DEFAULT 'battle',
        score           REAL NOT NULL,
        rank            INTEGER NOT NULL
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

    CREATE INDEX IF NOT EXISTS idx_scenes_book ON scenes(book_id);
    CREATE INDEX IF NOT EXISTS idx_scenes_rank ON scenes(book_id, rank);
    CREATE INDEX IF NOT EXISTS idx_chunks_scene ON scene_chunks(scene_id);
    CREATE INDEX IF NOT EXISTS idx_progress_phase ON build_progress(phase1_done, phase2_done);
"""


def init_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.commit()


def mark_skipped(conn, source_raw, book_id, chunk_count, reason, top_score=0.0):
    conn.execute("""
        INSERT OR REPLACE INTO build_progress
        (source_raw, book_id, chunk_count, top_score,
         phase1_done, skipped, skip_reason, processed_at)
        VALUES (?, ?, ?, ?, 1, 1, ?, ?)
    """, (source_raw, book_id, chunk_count, top_score,
          reason, datetime.utcnow().isoformat()))
    conn.commit()


def mark_phase1_done(conn, source_raw, book_id, chunk_count,
                     scenes_found, top_score):
    conn.execute("""
        INSERT OR REPLACE INTO build_progress
        (source_raw, book_id, chunk_count, scenes_found,
         top_score, phase1_done, phase2_done, skipped, processed_at)
        VALUES (?, ?, ?, ?, ?, 1, 0, 0, ?)
    """, (source_raw, book_id, chunk_count, scenes_found,
          top_score, datetime.utcnow().isoformat()))
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMIC SCENE COUNT
# ─────────────────────────────────────────────────────────────────────────────
def max_scenes_for_book(chunk_count: int) -> int:
    """
    Scales scene ceiling with book length.
    Shorter stories: 2-3 scenes.
    Full novels: 4-5.
    Omnibuses: 6-8.
    """
    if chunk_count < 50:    return 2
    if chunk_count < 200:   return 3
    if chunk_count < 800:   return 4
    if chunk_count < 3000:  return 5
    if chunk_count < 10000: return 6
    if chunk_count < 25000: return 7
    return 8


# ─────────────────────────────────────────────────────────────────────────────
#  CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
def cluster_candidates(candidates: list[dict],
                        gap: int,
                        min_len: int) -> list[list[dict]]:
    """
    Groups high-scoring chunks into contiguous scene clusters.
    candidates: [{chunk_id, text, combined_score, ...}]
    Chunks within `gap` IDs of each other are merged into one cluster.
    Clusters shorter than min_len are discarded.
    """
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


def rank_and_trim_clusters(clusters: list[list[dict]],
                            max_count: int) -> list[dict]:
    """
    Scores each cluster by (0.4 × avg_score + 0.6 × peak_score),
    returns top max_count as structured dicts.
    """
    scored = []
    for cluster in clusters:
        scores       = [c['combined_score'] for c in cluster]
        avg_score    = sum(scores) / len(scores)
        peak_score   = max(scores)
        cluster_score = 0.4 * avg_score + 0.6 * peak_score

        scored.append({
            'chunks':          cluster,
            'cluster_score':   cluster_score,
            'chunk_count':     len(cluster),
            'chunk_id_start':  cluster[0]['chunk_id'],
            'chunk_id_end':    cluster[-1]['chunk_id'],
        })

    scored.sort(key=lambda x: x['cluster_score'], reverse=True)
    return scored[:max_count]


def _score_clusters(clusters: list[list[dict]]) -> list[dict]:
    """
    Shared scoring step used by select_diverse_clusters: turns raw chunk-id
    clusters into structured dicts with cluster_score, span, and scene_type
    (carried from the chunks themselves — all chunks in one cluster share
    the same type, since mass/confrontation are clustered separately).
    """
    scored = []
    for cluster in clusters:
        scores        = [c['combined_score'] for c in cluster]
        avg_score     = sum(scores) / len(scores)
        peak_score    = max(scores)
        cluster_score = 0.4 * avg_score + 0.6 * peak_score

        scored.append({
            'chunks':          cluster,
            'cluster_score':   cluster_score,
            'chunk_count':     len(cluster),
            'chunk_id_start':  cluster[0]['chunk_id'],
            'chunk_id_end':    cluster[-1]['chunk_id'],
            'scene_type':      cluster[0].get('scene_type', 'battle'),
        })

    scored.sort(key=lambda x: x['cluster_score'], reverse=True)
    return scored


def compute_adaptive_separation(clusters: list[dict],
                                 book_chunk_count: int | None = None) -> int:
    """
    Derives a per-book minimum scene separation from the actual chunk_id
    span covered by this book's candidate clusters, instead of using one
    fixed distance across every book in the corpus.

    Books vary enormously in how their action is distributed: a short
    story's only fight might live in a 200-chunk pocket, while an omnibus
    can have genuinely separate battles tens of thousands of chunks apart.
    A fixed separation tuned for one shape breaks the other — too large
    and it collapses every scene to 1 pick; too small and it lets
    near-duplicate fragments of the same battle count as distinct scenes.

    clusters: the SCORED cluster dicts (post _score_clusters), i.e. each
    has chunk_id_start / chunk_id_end already.

    book_chunk_count: the book's TOTAL chunk count (not just the candidate
    span). Without this, SCENE_SEPARATION_MIN (15) can disproportionately
    swallow a short book — e.g. in a 47-chunk book, a 15-chunk floor
    consumes nearly a third of the entire book just to seat a second
    scene, even when max_scenes_for_book says the book should be allowed
    one. When provided, separation is additionally capped at a fraction
    of the whole book so the floor can never dominate a short book this
    way; the proportional/ceiling behavior for longer books is unaffected
    since this only ever LOWERS the result, never raises it.
    """
    if not clusters:
        return SCENE_SEPARATION_MIN

    span_start = min(c['chunk_id_start'] for c in clusters)
    span_end   = max(c['chunk_id_end']   for c in clusters)
    total_span = max(1, span_end - span_start)

    separation = int(total_span * SCENE_SEPARATION_FRACTION)
    separation = max(SCENE_SEPARATION_MIN, min(SCENE_SEPARATION_MAX, separation))

    if book_chunk_count:
        # Never let the separation requirement alone consume more than
        # this fraction of the entire book — protects short books from a
        # floor that's reasonable in absolute terms but oversized relative
        # to their length.
        book_cap = max(3, int(book_chunk_count * SCENE_SEPARATION_BOOK_FRACTION_CAP))
        separation = min(separation, book_cap)

    return separation


def select_diverse_clusters(clusters: list[list[dict]],
                             max_count: int,
                             min_separation: int | None = None,
                             min_confrontation_slots: int = MIN_CONFRONTATION_SLOTS,
                             book_chunk_count: int | None = None
                             ) -> list[dict]:
    """
    Replaces plain top-N selection with two corrections:

    1. SPREAD ENFORCEMENT — a candidate cluster is rejected if its span
       sits within the book's ADAPTIVE separation distance (see
       compute_adaptive_separation) of an already-selected cluster.
       Without this, one sprawling high-scoring set-piece can fill every
       scene slot for a book with near-duplicate fragments of the same
       sequence, crowding out a distinct, lower-but-still-strong scene
       elsewhere in the book. The separation scales with how spread out
       this particular book's content actually is, rather than using one
       fixed distance that only suits some books.

    2. CONFRONTATION RESERVATION — up to `min_confrontation_slots` of the
       final picks are reserved for the best confrontation-type clusters
       even if their raw score loses to several mass-battle clusters.
       Mass-battle language is denser in this corpus, so without a floor,
       intimate one-on-one scenes (the Molech cave guardian, a duel, a
       quiet stand-off) can lose every slot to louder fleet/siege action
       even when they're clearly a distinct, narratively important scene.

    min_separation: pass an explicit value to override the adaptive
    calculation (mainly for testing). Leave as None in normal use so it's
    computed per-book from the actual cluster spread.

    book_chunk_count: the book's total chunk count, passed through to
    compute_adaptive_separation so the separation can't disproportionately
    swallow a short book. Strongly recommended whenever min_separation is
    left as None; only relevant when min_separation is None (an explicit
    override skips the adaptive calculation entirely).

    Selection order: confrontation reservation fills first (best-scoring
    confrontation clusters, respecting spread against each other), then
    remaining slots fill from the full combined ranked list (respecting
    spread against everything already picked, regardless of type).
    """
    if not clusters or max_count <= 0:
        return []

    scored = _score_clusters(clusters)

    if min_separation is None:
        min_separation = compute_adaptive_separation(scored, book_chunk_count)

    def overlaps_or_too_close(candidate: dict, picked: list[dict]) -> bool:
        for p in picked:
            # direct overlap is always rejected regardless of separation
            if candidate['chunk_id_start'] <= p['chunk_id_end'] and \
               candidate['chunk_id_end'] >= p['chunk_id_start']:
                return True
            gap = max(
                candidate['chunk_id_start'] - p['chunk_id_end'],
                p['chunk_id_start'] - candidate['chunk_id_end'],
            )
            if gap < min_separation:
                return True
        return False

    selected: list[dict] = []

    # ── Pass 1: confrontation reservation ──────────────────────────────
    if min_confrontation_slots > 0:
        confrontation_ranked = [c for c in scored if c['scene_type'] == 'confrontation']
        for cand in confrontation_ranked:
            if len([s for s in selected if s['scene_type'] == 'confrontation']) >= min_confrontation_slots:
                break
            if len(selected) >= max_count:
                break
            if not overlaps_or_too_close(cand, selected):
                selected.append(cand)

    # ── Pass 2: fill remaining slots from the full ranked list ──────────
    for cand in scored:
        if len(selected) >= max_count:
            break
        if cand in selected:
            continue
        if not overlaps_or_too_close(cand, selected):
            selected.append(cand)

    # Preserve book-order (by chunk_id_start) for the final output rather
    # than score-order, so scenes read top-to-bottom as they occur in the
    # book — feels more like a table of contents than a leaderboard.
    selected.sort(key=lambda x: x['chunk_id_start'])
    return selected


# ─────────────────────────────────────────────────────────────────────────────
#  STITCHING  —  gap-fill, edge-trim, overlap-dedup
#  Turns a cluster of (possibly gappy, possibly overlapping) chunks into one
#  clean contiguous block of text ready to hand to the Phase 2 LLM weaver.
# ─────────────────────────────────────────────────────────────────────────────

# Matches the start of a new sentence: a capital letter (or opening quote
# then capital) immediately preceded by sentence-ending punctuation + space.
_SENTENCE_START_RE = re.compile(r'(?<=[.!?]\s)["\u201c]?[A-Z]')

# Matches the end of a sentence: ., !, or ? optionally followed by a
# closing quote, at the END of a candidate trim region.
_SENTENCE_END_RE = re.compile(r'[.!?]["\u201d]?')


def gap_fill_cluster(cluster_chunks: list[dict],
                      chunk_lookup: dict[int, dict]) -> list[dict]:
    """
    Given a cluster's candidate chunks (already sorted by chunk_id, but
    possibly missing chunks between start and end because those chunks
    never scored high enough to be a 'candidate'), pull in every chunk_id
    in [start, end] from the book's full chunk_lookup so the cluster
    becomes one contiguous span with no holes.

    Gap-filled chunks are marked with gap_filled=True so downstream code
    (and Phase 2 prompts, if useful) can tell which chunks were below the
    score gate but included purely for narrative continuity.
    """
    if not cluster_chunks:
        return cluster_chunks

    start_id = cluster_chunks[0]['chunk_id']
    end_id   = cluster_chunks[-1]['chunk_id']

    existing_ids = {c['chunk_id'] for c in cluster_chunks}
    filled: list[dict] = []

    for cid in range(start_id, end_id + 1):
        if cid in existing_ids:
            # keep the original candidate dict (has bm25/faiss/combined scores)
            match = next(c for c in cluster_chunks if c['chunk_id'] == cid)
            filled.append(match)
        elif cid in chunk_lookup:
            src_chunk = chunk_lookup[cid]
            filled.append({
                'chunk_id':       cid,
                'text':           src_chunk['text'],
                'bm25_score':     0.0,
                'faiss_score':    0.0,
                'combined_score': 0.0,
                'gap_filled':     True,
            })
        # if cid isn't in chunk_lookup at all (shouldn't normally happen),
        # we just skip it — a missing chunk_id is better than a crash.

    return filled


def trim_edges(text: str) -> str:
    """
    Strips a partial sentence from the very start and very end of a block
    of stitched text, so the cluster doesn't begin or end mid-sentence.

    Conservative by design: if no clean cut point is found within a
    reasonable lookahead/lookback window, the text is left untouched
    rather than risking mangled output.
    """
    if not text:
        return text

    SEARCH_WINDOW = 400  # chars to search from each edge before giving up

    # ── Trim the start: find the first sentence-start within the window ──
    head = text[:SEARCH_WINDOW]
    match = _SENTENCE_START_RE.search(head)
    if match:
        text = text[match.start():]

    # ── Trim the end: find the last sentence-end within the window ──
    tail_start = max(0, len(text) - SEARCH_WINDOW)
    tail = text[tail_start:]
    matches = list(_SENTENCE_END_RE.finditer(tail))
    if matches:
        last = matches[-1]
        text = text[:tail_start + last.end()]

    return text.strip()


def dedup_overlap(text_a: str, text_b: str) -> str:
    """
    Detects when text_b begins by repeating the last sentence (or close to
    it) of text_a — the overlap pattern visible in the real chunk data,
    e.g. chunk N ends '...she leapt about-face.' and chunk N+1 starts
    'She leapt about-face. The wheel was already...'

    Returns text_b with the repeated leading sentence stripped, or text_b
    unchanged if no overlap is detected.
    """
    if not text_a or not text_b:
        return text_b

    LOOKBACK = 200  # chars to check at the tail of text_a

    tail = text_a[-LOOKBACK:].strip()
    # last sentence of text_a: text after the second-to-last sentence end
    tail_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', tail) if s.strip()]
    if not tail_sentences:
        return text_b
    last_sentence = tail_sentences[-1]

    if len(last_sentence) < 8:
        # too short to reliably match — avoid false positives
        return text_b

    head = text_b[:LOOKBACK + len(last_sentence)].strip()

    if head.startswith(last_sentence):
        return text_b[len(last_sentence):].lstrip()

    # fuzzy fallback: normalize whitespace/case for comparison only
    norm_last = re.sub(r'\s+', ' ', last_sentence).strip().lower()
    norm_head = re.sub(r'\s+', ' ', head).strip().lower()
    if norm_head.startswith(norm_last):
        return text_b[len(last_sentence):].lstrip()

    return text_b


def stitch_cluster(cluster_chunks: list[dict],
                    chunk_lookup: dict[int, dict]) -> dict:
    """
    Full pipeline for one cluster: gap-fill -> dedup overlaps between
    consecutive chunks -> join -> trim edges. Returns a dict with the
    final stitched_text plus the (gap-filled) chunk list, so callers get
    both the clean prose and the underlying chunk_ids/scores.
    """
    filled = gap_fill_cluster(cluster_chunks, chunk_lookup)

    if not filled:
        return {'stitched_text': '', 'chunks': []}

    pieces = [filled[0]['text']]
    for prev, curr in zip(filled, filled[1:]):
        deduped = dedup_overlap(pieces[-1], curr['text'])
        pieces.append(deduped)

    joined = ' '.join(p.strip() for p in pieces if p.strip())
    stitched_text = trim_edges(joined)

    return {'stitched_text': stitched_text, 'chunks': filled}


# ─────────────────────────────────────────────────────────────────────────────
#  FAISS METRIC DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def is_inner_product_index(index: faiss.Index) -> bool:
    """
    Returns True if the index uses inner product (cosine when normalized),
    False if it uses L2. Determines how to interpret distances.
    """
    return isinstance(
        index,
        (faiss.IndexFlatIP, faiss.IndexIVFFlat)
    ) or 'IP' in type(index).__name__


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  OmnissiahCore  —  Phase 1: Score & Cluster")
    print("  Battle Scene Index Builder")
    print("=" * 70)

    # ── 1. Load metadata ─────────────────────────────────────────────────────
    print(f"\n[1/5]  Loading metadata ...")
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata: list[dict] = json.load(f)
    print(f"       {len(metadata):,} chunks loaded")

    # Build chunk_id → list-index map (FAISS position safety net)
    # Assumption: FAISS was built in chunk_id order, i.e. list position i
    # corresponds to chunk_id i. This holds for every chunk in your
    # metadata.json EXCEPT a known batch of 6,026 chunks (from a build_db.py
    # run that only wrote 'text' + 'source') which are missing 'chunk_id'
    # entirely. Those are backfilled here using their list position, which
    # is safe because it matches the existing chunk_id == list-position
    # pattern everywhere else in the file (verified via check_metadata.py).
    chunk_id_to_pos: dict[int, int] = {}
    missing_chunk_id = 0

    for i, chunk in enumerate(metadata):
        if 'chunk_id' not in chunk:
            missing_chunk_id += 1
            chunk['chunk_id'] = i
        chunk_id_to_pos[chunk['chunk_id']] = i

    if missing_chunk_id:
        print(f"       WARNING: {missing_chunk_id:,} chunk(s) were missing "
              f"'chunk_id' — backfilled from list position.")

    # ── 2. Group by source ───────────────────────────────────────────────────
    print("\n[2/5]  Grouping by source ...")
    books: dict[str, list[dict]] = {}
    for chunk in metadata:
        src = chunk.get('source', 'unknown')
        books.setdefault(src, []).append(chunk)
    print(f"       {len(books):,} unique books found")

    # ── 3. Init DB + checkpoint ──────────────────────────────────────────────
    conn = sqlite3.connect(str(SQLITE_PATH))
    init_db(conn)

    cur = conn.cursor()
    cur.execute(
        "SELECT source_raw FROM build_progress WHERE phase1_done = 1"
    )
    already_done = {row[0] for row in cur.fetchall()}

    books_todo = {
        src: chunks
        for src, chunks in books.items()
        if src not in already_done
    }
    print(f"       {len(already_done):,} already complete — skipping")
    print(f"       {len(books_todo):,} books to process this run")

    if not books_todo:
        print("\n  Nothing to do — all books already at Phase 1.")
        _print_db_summary(conn)
        conn.close()
        return

    # ── 4. Load FAISS + embed anchors ─────────────────────────────────────────
    print(f"\n[3/5]  Loading FAISS index ...")
    index = faiss.read_index(str(FAISS_PATH))
    print(f"       {index.ntotal:,} vectors, dim={index.d}")

    use_ip = is_inner_product_index(index)
    print(f"       Metric: {'Inner Product (cosine)' if use_ip else 'L2'}")

    print(f"\n       Loading BAAI/bge-m3 embedder ...")
    # HF_HUB_OFFLINE must be unset if model not cached, set if it is.
    # Script does not touch env vars — manage externally via lenovo_build profile.
    embedder = SentenceTransformer('BAAI/bge-m3')

    def embed_anchors(phrases: list[str]) -> np.ndarray:
        return embedder.encode(
            phrases,
            batch_size=EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    print(f"       Embedding {len(MASS_BATTLE_ANCHORS)} mass-battle anchors "
          f"+ {len(CONFRONTATION_ANCHORS)} confrontation anchors ...")
    mass_anchor_vecs = embed_anchors(MASS_BATTLE_ANCHORS)
    confrontation_anchor_vecs = embed_anchors(CONFRONTATION_ANCHORS)
    print(f"       Anchors ready — mass={mass_anchor_vecs.shape}  "
          f"confrontation={confrontation_anchor_vecs.shape}")

    # ── 5. Pre-compute FAISS scores across entire corpus, PER CATEGORY ────────
    # Scored as two independent pools so mass-battle language (which is both
    # more common and denser in this corpus) can't drown out confrontation-
    # type chunks just by sharing the same max-pool. Each chunk ends up with
    # two scores: how well it matches mass battle, and how well it matches
    # an intimate confrontation. The better category wins for that chunk.
    def faiss_score_pool(anchor_vecs: np.ndarray, label: str, top_k: int) -> dict[int, float]:
        print(f"\n[4/5]  Searching FAISS — {label} "
              f"({len(anchor_vecs)} anchors, top {top_k:,} each) ...")

        scores: dict[int, float] = {}  # chunk_id → raw score

        for anchor_vec in tqdm(anchor_vecs, desc=f"  {label} search", unit="anchor"):
            vec = np.expand_dims(anchor_vec, axis=0)
            distances, indices = index.search(vec, top_k)

            for dist, faiss_idx in zip(distances[0], indices[0]):
                if faiss_idx < 0:
                    continue

                # Convert distance to similarity
                if use_ip:
                    similarity = float(dist)           # IP: higher = more similar
                else:
                    similarity = 1.0 / (1.0 + float(dist))  # L2: invert distance

                # Map FAISS position back to chunk_id
                # Assumes FAISS position == chunk_id; adjust if build order differs
                chunk_id = faiss_idx
                if chunk_id not in scores or scores[chunk_id] < similarity:
                    scores[chunk_id] = similarity

        print(f"       {label}: scored {len(scores):,} unique chunks "
              f"({len(scores) / index.ntotal:.1%} of corpus)")

        # Normalise to [0, 1]
        if scores:
            vals = list(scores.values())
            v_min, v_max = min(vals), max(vals)
            v_range = v_max - v_min if v_max > v_min else 1.0
            scores = {k: (v - v_min) / v_range for k, v in scores.items()}

        return scores

    faiss_scores_mass = faiss_score_pool(
        mass_anchor_vecs, "mass-battle", FAISS_TOP_K_MASS
    )
    faiss_scores_confrontation = faiss_score_pool(
        confrontation_anchor_vecs, "confrontation", FAISS_TOP_K_CONFRONTATION
    )

    # ── 6. Score + cluster each book ─────────────────────────────────────────
    print(f"\n[5/5]  Scoring and clustering {len(books_todo):,} books ...")

    all_clusters: dict[str, dict] = {}
    stats = {
        'processed':     0,
        'skip_gate':     0,
        'skip_tiny':     0,
        'total_scenes':  0,
    }

    for source_raw, chunks in tqdm(books_todo.items(),
                                    desc="  Books",
                                    unit="book"):
        book_id     = make_book_id(source_raw)
        title       = clean_title(source_raw)
        chunk_count = len(chunks)

        # ── Tiny book gate ────────────────────────────────────────────────
        if chunk_count < 5:
            mark_skipped(conn, source_raw, book_id, chunk_count, 'too_few_chunks')
            stats['skip_tiny'] += 1
            continue

        # chunk_id -> chunk dict, for this book only. Needed by gap-fill
        # to pull in chunks that never scored above SCORE_GATE but sit
        # inside a cluster's span (e.g. a quiet beat between two strikes).
        chunk_lookup: dict[int, dict] = {c['chunk_id']: c for c in chunks}

        # ── BM25 scoring within this book — TWO vocabularies ────────────────
        tokenized_corpus = [c['text'].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        bm25_raw_mass = bm25.get_scores(MASS_BATTLE_VOCAB)
        bm25_raw_conf = bm25.get_scores(CONFRONTATION_VOCAB)

        bm25_max_mass = float(bm25_raw_mass.max())
        bm25_max_conf = float(bm25_raw_conf.max())
        bm25_norm_mass = (bm25_raw_mass / bm25_max_mass) if bm25_max_mass > 0 else bm25_raw_mass
        bm25_norm_conf = (bm25_raw_conf / bm25_max_conf) if bm25_max_conf > 0 else bm25_raw_conf

        # ── Combine BM25 + FAISS into two parallel scores per chunk ─────────
        # Each chunk gets a mass-battle score and a confrontation score,
        # computed independently. Whichever is higher determines the
        # chunk's scene_type and which score it's gated/clustered on — a
        # chunk doesn't have to win on raw combat density to register as a
        # strong confrontation moment if that's the pool it actually fits.
        candidates_mass: list[dict] = []
        candidates_conf: list[dict] = []

        for i, chunk in enumerate(chunks):
            cid = chunk['chunk_id']

            mass_score = (BM25_WEIGHT * float(bm25_norm_mass[i])
                          + FAISS_WEIGHT * faiss_scores_mass.get(cid, 0.0))
            conf_score = (BM25_WEIGHT * float(bm25_norm_conf[i])
                          + FAISS_WEIGHT * faiss_scores_confrontation.get(cid, 0.0))

            if mass_score >= SCORE_GATE_MASS:
                candidates_mass.append({
                    'chunk_id':       cid,
                    'text':           chunk['text'],
                    'bm25_score':     float(bm25_norm_mass[i]),
                    'faiss_score':    faiss_scores_mass.get(cid, 0.0),
                    'combined_score': mass_score,
                    'scene_type':     'battle',
                })

            if conf_score >= SCORE_GATE_CONFRONTATION:
                candidates_conf.append({
                    'chunk_id':       cid,
                    'text':           chunk['text'],
                    'bm25_score':     float(bm25_norm_conf[i]),
                    'faiss_score':    faiss_scores_confrontation.get(cid, 0.0),
                    'combined_score': conf_score,
                    'scene_type':     'confrontation',
                })

        candidates = candidates_mass + candidates_conf

        # ── Book-level score gate ─────────────────────────────────────────
        # Filters stat blocks, codex tables, lore encyclopedias.
        if len(candidates) < BOOK_GATE:
            top = max(
                (c['combined_score'] for c in candidates), default=0.0
            )
            mark_skipped(conn, source_raw, book_id, chunk_count,
                         'score_gate', top)
            stats['skip_gate'] += 1
            continue

        # ── Cluster each category separately ────────────────────────────────
        # Clustering mass and confrontation candidates together would let a
        # confrontation cluster get absorbed into an adjacent mass-battle
        # cluster mid-merge (or vice versa) purely because their chunk_ids
        # are close. Clustering separately keeps a quiet one-on-one scene
        # distinct even if it happens to sit near a larger battle.
        max_s = max_scenes_for_book(chunk_count)

        raw_clusters_mass = cluster_candidates(candidates_mass, GAP_THRESHOLD, MIN_CLUSTER_LEN)
        raw_clusters_conf = cluster_candidates(candidates_conf, GAP_THRESHOLD, MIN_CLUSTER_LEN)

        for cl in raw_clusters_mass:
            for c in cl:
                c.setdefault('scene_type', 'battle')
        for cl in raw_clusters_conf:
            for c in cl:
                c.setdefault('scene_type', 'confrontation')

        all_raw_clusters = raw_clusters_mass + raw_clusters_conf

        top_clusters = select_diverse_clusters(
            all_raw_clusters,
            max_count=max_s,
            min_separation=None,  # adaptive — derived per-book from actual cluster spread
            min_confrontation_slots=MIN_CONFRONTATION_SLOTS,
            book_chunk_count=chunk_count,
        )

        # Edge case: candidates exist but clustering failed (all singletons)
        # → treat top max_s individual candidates as micro-scenes
        if not top_clusters and candidates:
            top_candidates = sorted(
                candidates,
                key=lambda x: x['combined_score'],
                reverse=True
            )[:max_s]
            top_clusters = [
                {
                    'chunks':          [c],
                    'cluster_score':   c['combined_score'],
                    'chunk_count':     1,
                    'chunk_id_start':  c['chunk_id'],
                    'chunk_id_end':    c['chunk_id'],
                    'scene_type':      c.get('scene_type', 'battle'),
                }
                for c in top_candidates
            ]

        if not top_clusters:
            mark_skipped(conn, source_raw, book_id, chunk_count,
                         'no_clusters_formed')
            stats['skip_gate'] += 1
            continue

        # ── Stitch each cluster: gap-fill -> dedup overlap -> trim edges ───
        # Replaces each cluster's raw chunk list with the gap-filled version
        # and computes the final clean prose block Phase 2 will weave.
        for cl in top_clusters:
            stitched = stitch_cluster(cl['chunks'], chunk_lookup)
            cl['chunks']        = stitched['chunks']
            cl['stitched_text'] = stitched['stitched_text']
            # gap-fill may have changed the chunk_count and exact span
            if cl['chunks']:
                cl['chunk_count']    = len(cl['chunks'])
                cl['chunk_id_start'] = cl['chunks'][0]['chunk_id']
                cl['chunk_id_end']   = cl['chunks'][-1]['chunk_id']

        # ── Upsert book into DB ───────────────────────────────────────────
        conn.execute("""
            INSERT OR IGNORE INTO books
            (book_id, title, source_raw, chunk_count)
            VALUES (?, ?, ?, ?)
        """, (book_id, title, source_raw, chunk_count))

        scenes_found = len(top_clusters)
        # top_clusters is sorted by book-order (chunk_id_start), not score,
        # so take the max explicitly for the progress-tracking column.
        top_score = max(cl['cluster_score'] for cl in top_clusters)

        mark_phase1_done(conn, source_raw, book_id,
                         chunk_count, scenes_found, top_score)

        # ── Build output record ───────────────────────────────────────────
        all_clusters[source_raw] = {
            'book_id':     book_id,
            'title':       title,
            'source_raw':  source_raw,
            'chunk_count': chunk_count,
            'scenes': [
                {
                    'rank':           rank + 1,
                    'cluster_score':  cl['cluster_score'],
                    'scene_type':     cl.get('scene_type', 'battle'),
                    'chunk_id_start': cl['chunk_id_start'],
                    'chunk_id_end':   cl['chunk_id_end'],
                    'chunk_count':    cl['chunk_count'],
                    # clean, gap-filled, overlap-deduped, edge-trimmed prose
                    # ready to hand straight to the Phase 2 LLM weaver
                    'stitched_text':  cl['stitched_text'],
                    # chunk_ids for reference / future re-stitching
                    'chunk_ids': [c['chunk_id'] for c in cl['chunks']],
                    # raw per-chunk texts, kept for debugging / inspection
                    'texts':     [c['text']     for c in cl['chunks']],
                    # individual chunk scores for Phase 2 quality reference
                    # (gap-filled chunks show 0.0 — they were below the gate)
                    'scores':    [round(c['combined_score'], 4)
                                  for c in cl['chunks']],
                }
                for rank, cl in enumerate(top_clusters)
            ],
        }

        stats['processed']    += 1
        stats['total_scenes'] += scenes_found

    # ── Write clusters_raw.json ───────────────────────────────────────────────
    print(f"\n  Writing {OUTPUT_JSON.name} ...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_clusters, f, ensure_ascii=False, indent=2)
    size_mb = OUTPUT_JSON.stat().st_size / 1_048_576
    print(f"  Written — {len(all_clusters):,} books, {size_mb:.1f} MB")

    # ── Final report ──────────────────────────────────────────────────────────
    _print_db_summary(conn)

    print("\n" + "=" * 70)
    print("  Phase 1 Complete")
    print("=" * 70)
    print(f"  Books processed         : {stats['processed']:,}")
    print(f"  Books skipped (gate)    : {stats['skip_gate']:,}")
    print(f"  Books skipped (tiny)    : {stats['skip_tiny']:,}")
    print(f"  Total scenes clustered  : {stats['total_scenes']:,}")
    print(f"\n  clusters_raw.json  →  {OUTPUT_JSON}")
    print(f"  battle_scenes.db   →  {SQLITE_PATH}")
    print("=" * 70)
    print("\n  ✓ Run scene_cluster_builder_phase2.py to weave with the LLM")

    conn.close()


def _print_db_summary(conn: sqlite3.Connection):
    """Prints a live progress snapshot from build_progress."""
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*)                              AS total,
            SUM(CASE WHEN phase2_done = 1 THEN 1 ELSE 0 END) AS p2_done,
            SUM(CASE WHEN phase1_done = 1
                      AND phase2_done = 0
                      AND skipped = 0 THEN 1 ELSE 0 END)     AS p1_only,
            SUM(CASE WHEN skipped = 1 THEN 1 ELSE 0 END)     AS skipped
        FROM build_progress
    """)
    row = cur.fetchone()
    if row and row[0]:
        total, p2, p1, skipped = row
        print(f"\n  ── DB Snapshot ──────────────────────────────────")
        print(f"     Total tracked      : {total:,}")
        print(f"     Phase 2 complete   : {p2:,}")
        print(f"     Awaiting Phase 2   : {p1:,}")
        print(f"     Skipped (gate)     : {skipped:,}")
        print(f"  ────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\n  Wall time: {elapsed / 60:.1f} minutes")