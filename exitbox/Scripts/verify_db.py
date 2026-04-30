"""
OmnissiahCore — Scripts/verify_db.py

Run this on EITHER machine to verify the FAISS index and metadata are healthy.
Safe to run on Dell — does NOT re-embed anything.

Run from project root:
    python Scripts/verify_db.py

Checks:
  1. Files exist (faiss.index, metadata.json, manifest.json)
  2. Vector count matches metadata count
  3. Embedding model dimension matches FAISS index dimension
  4. Metadata chunk_id continuity
  5. Source distribution (books and file types)
  6. Sample chunk readability
  7. Failed files log
"""

import os
import sys
import json
import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Core.config_loader import paths, embedding_cfg

SEP = "─" * 70
PASS = "✅"
WARN = "⚠️ "
FAIL = "❌"


def check(label: str, condition: bool, warn_only: bool = False) -> bool:
    icon = PASS if condition else (WARN if warn_only else FAIL)
    print(f"  {icon}  {label}")
    return condition


print("\n" + "═" * 70)
print("  OmnissiahCore — Index Verification")
print("═" * 70)

# ── 1. File existence ─────────────────────────────────────────────────────
print("\n[1] File Existence")
ok_faiss  = check("faiss.index   exists", os.path.exists(paths["faiss"]))
ok_meta   = check("metadata.json exists", os.path.exists(paths["metadata"]))
ok_manif  = check("manifest.json exists", os.path.exists(paths["manifest"]), warn_only=True)
ok_proc   = check("processed_files.json exists", os.path.exists(paths["processed"]), warn_only=True)
ok_failed = check("failed_files.json exists", os.path.exists(paths["failed_log"]), warn_only=True)

if not ok_faiss or not ok_meta:
    print(f"\n{FAIL} Critical files missing. Cannot continue.")
    sys.exit(1)

# ── 2. Load index ─────────────────────────────────────────────────────────
print(f"\n[2] FAISS Index")
index    = faiss.read_index(paths["faiss"])
n_vecs   = index.ntotal
idx_dim  = index.d
print(f"  {PASS}  Loaded: {n_vecs:,} vectors  |  dim={idx_dim}")

# ── 3. Load metadata ──────────────────────────────────────────────────────
print(f"\n[3] Metadata")
with open(paths["metadata"], "r", encoding="utf-8", errors="replace") as f:
    metadata = json.load(f)
n_chunks = len(metadata)
print(f"  {PASS}  Loaded: {n_chunks:,} chunks")

match = n_vecs == n_chunks
check(f"Vector count matches chunk count ({n_vecs:,} == {n_chunks:,})", match)
if not match:
    print(f"       Gap: {abs(n_vecs - n_chunks)} — index may be partially built")

# ── 4. Dimension check ────────────────────────────────────────────────────
print(f"\n[4] Embedding Dimension Check")
print(f"  Loading {embedding_cfg['model']} to verify output dimension...")
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(embedding_cfg["model"])
    m.max_seq_length = embedding_cfg["max_seq_length"]
    test = m.encode(["dimension probe"], normalize_embeddings=True, convert_to_numpy=True)
    model_dim = test.shape[1]
    check(
        f"Model dim ({model_dim}) matches index dim ({idx_dim})",
        model_dim == idx_dim
    )
except Exception as e:
    print(f"  {WARN}  Could not load model to verify dim: {e}")

# ── 5. chunk_id continuity ────────────────────────────────────────────────
print(f"\n[5] chunk_id Continuity")
ids = [m.get("chunk_id", i) for i, m in enumerate(metadata)]
gaps = sum(1 for a, b in zip(ids, ids[1:]) if b - a != 1)
check(f"chunk_id is sequential (gaps found: {gaps})", gaps == 0, warn_only=True)

# ── 6. Source distribution ────────────────────────────────────────────────
print(f"\n[6] Source Distribution")
from collections import Counter
sources    = Counter(m.get("source", "unknown") for m in metadata)
file_types = Counter(m.get("file_type", "?") for m in metadata)

print(f"  {PASS}  Unique sources  : {len(sources):,}")
print(f"  {PASS}  File types      : {dict(file_types)}")
print(f"\n  Top 15 sources by chunk count:")
for src, cnt in sources.most_common(15):
    bar = "█" * min(cnt // 200, 40)
    print(f"    {cnt:>6,}  {bar}  {src[:60]}")

# ── 7. Sample chunks ──────────────────────────────────────────────────────
print(f"\n[7] Sample Chunks (first 3)")
for i, m in enumerate(metadata[:3]):
    print(f"  [{i}] source={m.get('source','?')}  chapter={m.get('chapter','?')}")
    preview = m.get("text", "")[:120].replace("\n", " ")
    print(f"       {preview}...")
    print()

# ── 8. Failed files ───────────────────────────────────────────────────────
print(f"\n[8] Failed Files")
if os.path.exists(paths["failed_log"]):
    with open(paths["failed_log"]) as f:
        failed = json.load(f)
    print(f"  {WARN if failed else PASS}  {len(failed)} files in failed log")
    if failed:
        reasons = Counter(v for v in failed.values())
        for reason, cnt in reasons.most_common():
            print(f"    {cnt:>4}  {reason}")
        print(f"\n  Run:  python Scripts/build_db.py --retry-failed")
else:
    print(f"  {PASS}  No failed files log.")

# ── 9. Manifest ───────────────────────────────────────────────────────────
print(f"\n[9] Manifest")
if os.path.exists(paths["manifest"]):
    with open(paths["manifest"]) as f:
        manifest = json.load(f)
    for k, v in manifest.items():
        print(f"  {k:<24}: {v}")

print(f"\n{'═'*70}")
print(f"  Verification complete.  {n_chunks:,} chunks ready for query.")
print(f"{'═'*70}\n")
