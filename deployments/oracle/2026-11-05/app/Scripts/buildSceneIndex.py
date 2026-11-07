"""
OmnissiahCore - Scripts/build_scene_index.py

Builds a small FAISS + BM25 index over battle_scenes.db so scenes can be
searched live at query time, the same way the main corpus is searched.
Reads real model/path config from Core/config_loader.py instead of
hardcoded guesses.

Run from anywhere inside the project (needs to be able to `import Core`,
so either run from project root, or leave this script in Scripts/ and
run it as `python Scripts/build_scene_index.py` from root):

    python Scripts/build_scene_index.py                       # build
    python Scripts/build_scene_index.py --test "your query"   # search

Outputs (per config_loader paths, inside Db/):
    Db/scene_faiss.index
    Db/scene_metadata.json
    Db/scene_bm25_cache/
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Make sure `Core` is importable regardless of where this script is run from.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name.lower() == "scripts" else _THIS_DIR
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import faiss

try:
    import bm25s
    _BM25_INSTALLED = True
except ImportError:
    _BM25_INSTALLED = False
    print("Warning: bm25s not installed. Scene BM25 index will be skipped.")

from sentence_transformers import SentenceTransformer

from Core.config_loader import embedding_cfg, paths

DB_PATH = Path(paths["battle_scenes_db"])
INDEX_PATH = Path(paths["scene_faiss"])
METADATA_PATH = Path(paths["scene_metadata"])
BM25_CACHE_DIR = Path(paths["scene_bm25_cache"])

CHRONICLE_CHAR_LIMIT = 3000  # how much of the chronicle to embed
CHRONICLE_PREVIEW_LEN = 300  # how much chronicle text to keep in metadata.json for eyeballing


def load_scenes(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT s.scene_id, s.book_id, b.title AS book_title, s.scene_name,
               s.teaser, s.chronicle, s.scene_type, s.score, s.rank
        FROM scenes s
        JOIN books b ON b.book_id = s.book_id
        WHERE s.scene_name IS NOT NULL AND s.chronicle IS NOT NULL
    """).fetchall()
    cols = ["scene_id", "book_id", "book_title", "scene_name", "teaser",
            "chronicle", "scene_type", "score", "rank"]
    return [dict(zip(cols, row)) for row in rows]


def build_embed_text(scene: dict) -> str:
    """
    scene_name and teaser carry the most direct signal for what a user
    would actually type, so they're never truncated. Chronicle is capped
    since it's long prose and the first ~3000 chars usually establish
    setting/combatants/stakes.
    """
    chronicle_snippet = (scene["chronicle"] or "")[:CHRONICLE_CHAR_LIMIT]
    return (
        f"{scene['scene_name']}. {scene['teaser']} "
        f"Book: {scene['book_title']}. {chronicle_snippet}"
    )


def build():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        sys.exit(1)

    os.makedirs(INDEX_PATH.parent, exist_ok=True)

    print(f"Loading scenes from {DB_PATH} ...")
    conn = sqlite3.connect(str(DB_PATH))
    scenes = load_scenes(conn)
    conn.close()
    print(f"  {len(scenes):,} woven scenes found")

    if not scenes:
        print("ERROR: no scenes with scene_name + chronicle populated. Has phase2_build.py run?")
        sys.exit(1)

    model_name = embedding_cfg["model"]
    max_seq_length = embedding_cfg["max_seq_length"]
    print(f"Loading embedding model: {model_name}  (max_seq_length={max_seq_length}) ...")
    embedder = SentenceTransformer(model_name)
    embedder.max_seq_length = max_seq_length

    texts = [build_embed_text(s) for s in scenes]
    print(f"Embedding {len(texts):,} scenes ...")
    vectors = embedder.encode(
        texts,
        normalize_embeddings=embedding_cfg["normalize"],
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine similarity
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"FAISS scene index saved: {INDEX_PATH}  ({index.ntotal:,} vectors, dim={dim})")

    metadata = []
    for s in scenes:
        m = {k: v for k, v in s.items() if k != "chronicle"}
        m["chronicle_preview"] = (s["chronicle"] or "")[:CHRONICLE_PREVIEW_LEN]
        metadata.append(m)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Scene metadata saved: {METADATA_PATH}")
    print("  (Note: full chronicle text is NOT duplicated here — retriever.py")
    print("   fetches it live from battle_scenes.db by scene_id at query time.)")

    if _BM25_INSTALLED:
        print("Building BM25 index over scene text ...")
        corpus_tokens = bm25s.tokenize(texts, stopwords=None, show_progress=False)
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens, show_progress=False)
        bm25.save(str(BM25_CACHE_DIR))
        print(f"BM25 scene index saved: {BM25_CACHE_DIR}")

    print("\nBuild complete. Run with --test \"your query\" to try it.")


def test_query(query: str, max_sources: int = 2):
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        print("ERROR: index not built yet. Run without --test first.")
        sys.exit(1)

    model_name = embedding_cfg["model"]
    max_seq_length = embedding_cfg["max_seq_length"]
    embedder = SentenceTransformer(model_name)
    embedder.max_seq_length = max_seq_length

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    k = min(10, index.ntotal)
    scores, indices = index.search(vec, k)

    print(f"\nQuery: \"{query}\"")
    print(f"{'='*70}")
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        m = metadata[idx]
        print(f"\n  score={score:.4f}  book={m['book_title']}")
        print(f"  scene: \"{m['scene_name']}\"")
        print(f"  teaser: {m['teaser']}")

    print(f"\n{'='*70}")
    top_score = float(scores[0][0]) if len(scores[0]) else 0.0
    print(f"Top score: {top_score:.4f}  (retriever.py's FOCUS_CONFIDENCE_THRESHOLD gates on this)")
    print(f"(real /query/focus applies a confidence threshold and caps at max_sources={max_sources})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default=None, help="Search the index with this query instead of building")
    args = parser.parse_args()

    if args.test:
        test_query(args.test)
    else:
        build()