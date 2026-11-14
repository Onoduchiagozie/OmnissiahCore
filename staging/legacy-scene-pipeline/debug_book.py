"""
debug_book.py — runs Phase 1's scoring/clustering pipeline for ONE book
only, printing every intermediate count, so we can see exactly where
scenes are being lost: BOOK_GATE, per-category candidate counts, raw
cluster counts, or the final select_diverse_clusters step.

Run from the same folder as battle.py and Db/:
    python debug_book.py "29. Graham McNeill - Vengeful Spirit (The Horus Heresy, Book 29).pdf"

If no argument given, defaults to the Vengeful Spirit filename above.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import battle  # noqa: E402

TARGET_SOURCE = (
    sys.argv[1] if len(sys.argv) > 1
    else "29. Graham McNeill - Vengeful Spirit (The Horus Heresy, Book 29).pdf"
)


def main():
    print(f"Loading metadata for: {TARGET_SOURCE}")
    with open(battle.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for i, chunk in enumerate(metadata):
        if "chunk_id" not in chunk:
            chunk["chunk_id"] = i

    chunks = [c for c in metadata if c.get("source") == TARGET_SOURCE]
    print(f"chunk_count = {len(chunks)}")

    if not chunks:
        print("No chunks found for that source string. Check exact filename via:")
        print("  python -c \"import json; d=json.load(open('Db/metadata.json',encoding='utf-8')); "
              "print(set(c['source'] for c in d if 'molech' in c.get('source','').lower() "
              "or 'vengeful' in c.get('source','').lower()))\"")
        return

    print("\nLoading FAISS index ...")
    index = battle.faiss.read_index(str(battle.FAISS_PATH))
    use_ip = battle.is_inner_product_index(index)

    print("Loading embedder ...")
    embedder = battle.SentenceTransformer('BAAI/bge-m3')

    def embed(phrases):
        return embedder.encode(
            phrases, batch_size=battle.EMBED_BATCH,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(battle.np.float32)

    mass_vecs = embed(battle.MASS_BATTLE_ANCHORS)
    conf_vecs = embed(battle.CONFRONTATION_ANCHORS)

    def faiss_pool(anchor_vecs, top_k):
        scores = {}
        for av in anchor_vecs:
            vec = battle.np.expand_dims(av, axis=0)
            distances, indices = index.search(vec, top_k)
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                sim = float(dist) if use_ip else 1.0 / (1.0 + float(dist))
                if idx not in scores or scores[idx] < sim:
                    scores[idx] = sim
        if scores:
            vals = list(scores.values())
            v_min, v_max = min(vals), max(vals)
            v_range = v_max - v_min if v_max > v_min else 1.0
            scores = {k: (v - v_min) / v_range for k, v in scores.items()}
        return scores

    print("Scoring FAISS (full corpus, same as real run) ...")
    faiss_mass = faiss_pool(mass_vecs, battle.FAISS_TOP_K_MASS)
    faiss_conf = faiss_pool(conf_vecs, battle.FAISS_TOP_K_CONFRONTATION)

    chunk_lookup = {c['chunk_id']: c for c in chunks}

    tokenized = [c['text'].lower().split() for c in chunks]
    bm25 = battle.BM25Okapi(tokenized)
    bm25_raw_mass = bm25.get_scores(battle.MASS_BATTLE_VOCAB)
    bm25_raw_conf = bm25.get_scores(battle.CONFRONTATION_VOCAB)
    bm25_max_mass = float(bm25_raw_mass.max())
    bm25_max_conf = float(bm25_raw_conf.max())
    bm25_norm_mass = (bm25_raw_mass / bm25_max_mass) if bm25_max_mass > 0 else bm25_raw_mass
    bm25_norm_conf = (bm25_raw_conf / bm25_max_conf) if bm25_max_conf > 0 else bm25_raw_conf

    candidates_mass, candidates_conf = [], []
    for i, chunk in enumerate(chunks):
        cid = chunk['chunk_id']
        mass_score = (battle.BM25_WEIGHT * float(bm25_norm_mass[i])
                      + battle.FAISS_WEIGHT * faiss_mass.get(cid, 0.0))
        conf_score = (battle.BM25_WEIGHT * float(bm25_norm_conf[i])
                      + battle.FAISS_WEIGHT * faiss_conf.get(cid, 0.0))
        if mass_score >= battle.SCORE_GATE_MASS:
            candidates_mass.append({'chunk_id': cid, 'text': chunk['text'],
                                     'combined_score': mass_score, 'scene_type': 'battle'})
        if conf_score >= battle.SCORE_GATE_CONFRONTATION:
            candidates_conf.append({'chunk_id': cid, 'text': chunk['text'],
                                     'combined_score': conf_score, 'scene_type': 'confrontation'})

    print(f"\ncandidates_mass = {len(candidates_mass)}")
    print(f"candidates_conf = {len(candidates_conf)}")

    raw_clusters_mass = battle.cluster_candidates(candidates_mass, battle.GAP_THRESHOLD, battle.MIN_CLUSTER_LEN)
    raw_clusters_conf = battle.cluster_candidates(candidates_conf, battle.GAP_THRESHOLD, battle.MIN_CLUSTER_LEN)

    print(f"\nraw_clusters_mass count = {len(raw_clusters_mass)}")
    for cl in raw_clusters_mass:
        scores = [c['combined_score'] for c in cl]
        print(f"  span={cl[0]['chunk_id']}-{cl[-1]['chunk_id']}  "
              f"len={len(cl)}  peak={max(scores):.3f}")

    print(f"\nraw_clusters_conf count = {len(raw_clusters_conf)}")
    for cl in raw_clusters_conf:
        scores = [c['combined_score'] for c in cl]
        print(f"  span={cl[0]['chunk_id']}-{cl[-1]['chunk_id']}  "
              f"len={len(cl)}  peak={max(scores):.3f}")

    max_s = battle.max_scenes_for_book(len(chunks))
    print(f"\nmax_scenes_for_book({len(chunks)}) = {max_s}")

    all_raw = raw_clusters_mass + raw_clusters_conf
    final = battle.select_diverse_clusters(
        all_raw, max_count=max_s,
        min_separation=battle.MIN_SCENE_SEPARATION,
        min_confrontation_slots=battle.MIN_CONFRONTATION_SLOTS,
    )
    print(f"\nFINAL selected scenes = {len(final)}")
    for f in final:
        print(f"  type={f['scene_type']:14s} span={f['chunk_id_start']}-{f['chunk_id_end']}  "
              f"score={f['cluster_score']:.3f}")


if __name__ == "__main__":
    main()