"""
memory_profile.py

Standalone diagnostic — mirrors the startup sequence of OmnissiahRetriever
(post bm25s migration), printing RSS memory after each stage so you can see
precisely which step causes the RAM jump.

Run this FROM THE PROJECT ROOT (same place you'd run `python main.py api`):
    python memory_profile.py

Requires: pip install psutil bm25s --break-system-packages
"""

import os
import sys

import psutil

# Make Core/ and Api/ importable, same as server.py does
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_proc = psutil.Process(os.getpid())


def rss_mb() -> float:
    return _proc.memory_info().rss / (1024 * 1024)


def checkpoint(label: str, baseline: float):
    current = rss_mb()
    delta = current - baseline
    print(f"[{label:35s}] RSS = {current:8.1f} MB   (+{delta:8.1f} MB this stage)")
    return current


def main():
    print("=" * 80)
    print("MEMORY PROFILE — OmnissiahCore startup sequence")
    print("=" * 80)

    baseline = rss_mb()
    print(f"[{'baseline (python + imports)':35s}] RSS = {baseline:8.1f} MB")

    # --- Stage 1: config_loader (cheap, just JSON) ---
    from Core.config_loader import embedding_cfg, machine_role, paths, retrieval_cfg
    baseline = checkpoint("after config_loader import", baseline)

    # --- Stage 2: heavy library imports (torch, sentence_transformers, faiss) ---
    import faiss
    baseline = checkpoint("after `import faiss`", baseline)

    import torch
    baseline = checkpoint("after `import torch`", baseline)

    from sentence_transformers import SentenceTransformer, CrossEncoder
    baseline = checkpoint("after `import sentence_transformers`", baseline)

    try:
        import bm25s
        bm25_available = True
    except ImportError:
        bm25_available = False
    baseline = checkpoint("after `import bm25s`", baseline)

    # --- Stage 3: FAISS index load ---
    index = faiss.read_index(paths["faiss"])
    print(f"   -> FAISS index: {index.ntotal:,} vectors, dim={index.d}")
    baseline = checkpoint("after faiss.read_index()", baseline)

    # --- Stage 4: metadata.json load (ONCE — this is the fixed version) ---
    import json
    with open(paths["metadata"], "r", encoding="utf-8", errors="replace") as f:
        metadata = json.load(f)
    print(f"   -> metadata: {len(metadata):,} chunks")
    baseline = checkpoint("after metadata.json load", baseline)

    # --- Stage 5: embedder load (bge-m3) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   -> resolved device: {device}")
    embedder = SentenceTransformer(embedding_cfg["model"], device=device)
    embedder.max_seq_length = embedding_cfg["max_seq_length"]
    baseline = checkpoint(f"after SentenceTransformer({embedding_cfg['model']})", baseline)

    # Dimension check, same as retriever.py does
    test_vec = embedder.encode(["dimension check"], normalize_embeddings=True, convert_to_numpy=True)
    baseline = checkpoint("after one embedder.encode() call", baseline)

    # --- Stage 6: BM25 corpus build ---
    if retrieval_cfg["use_bm25"] and bm25_available:
        corpus_texts = [m.get("text", "") for m in metadata]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords=None, show_progress=False)
        baseline = checkpoint("after bm25s.tokenize(corpus)", baseline)

        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens, show_progress=False)
        baseline = checkpoint("after bm25.index() build", baseline)
    else:
        print("   -> BM25 disabled, skipping")

    # --- Stage 7: CrossEncoder reranker load ---
    if retrieval_cfg["use_reranker"] and retrieval_cfg.get("rerank_model"):
        reranker = CrossEncoder(retrieval_cfg["rerank_model"], device=device)
        baseline = checkpoint(f"after CrossEncoder({retrieval_cfg['rerank_model']})", baseline)
    else:
        print("   -> Reranker disabled, skipping")

    print("=" * 80)
    print(f"FINAL RSS: {rss_mb():.1f} MB  ({rss_mb() / 1024:.2f} GB)")
    print("=" * 80)
    print("\nNOTE: This does NOT load any LLM (Gemma/LM Studio model).")
    print("That happens separately, inside LM Studio's own process, not here.")


if __name__ == "__main__":
    main()