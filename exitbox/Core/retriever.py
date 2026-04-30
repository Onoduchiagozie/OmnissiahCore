"""
OmnissiahCore - Core/retriever.py

Hybrid RAG retriever:
  1. FAISS dense vector search
  2. BM25 sparse keyword search
  3. Reciprocal Rank Fusion
  4. Query-overlap grounding boost
  5. Chunk stitching
  6. Optional cross-encoder rerank
"""

import json
import os
import re
from hashlib import md5

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from Core.config_loader import embedding_cfg, machine_role, paths, retrieval_cfg

try:
    from rank_bm25 import BM25Okapi

    _BM25_INSTALLED = True
except ImportError:
    _BM25_INSTALLED = False
    print("Warning: rank_bm25 not installed. BM25 disabled.")

try:
    from sentence_transformers import CrossEncoder

    _CROSSENCODER_AVAILABLE = True
except ImportError:
    _CROSSENCODER_AVAILABLE = False


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "from", "by", "about", "what", "who", "when", "where", "how", "why", "is",
    "was", "were", "be", "been", "being", "first", "scene", "description", "made",
    "attack", "battle", "story", "tell", "me", "this", "that", "these", "those",
}

TERM_VARIANTS = {
    "five": {"v", "5", "five"},
    "third": {"iii", "3rd", "third", "3"},
    "fourth": {"iv", "4th", "fourth", "4"},
    "sixth": {"vi", "6th", "sixth", "6"},
}


class OmnissiahRetriever:
    def __init__(self):
        self._device = self._resolve_device()
        print(f"Initialising retriever  [machine_role={machine_role}  device={self._device}]")

        self._check_file(paths["faiss"], "FAISS index")
        self._check_file(paths["metadata"], "metadata.json")

        self.index = faiss.read_index(paths["faiss"])
        print(f"   FAISS index: {self.index.ntotal:,} vectors  dim={self.index.d}")

        with open(paths["metadata"], "r", encoding="utf-8", errors="replace") as f:
            self.metadata: list[dict] = json.load(f)
        print(f"   Metadata: {len(self.metadata):,} chunks")

        self._id_to_idx: dict[int, int] = {}
        for idx, chunk in enumerate(self.metadata):
            cid = chunk.get("chunk_id", idx)
            self._id_to_idx[cid] = idx

        print(f"   Loading embedding model: {embedding_cfg['model']}")
        self.embedder = SentenceTransformer(embedding_cfg["model"], device=self._device)
        self.embedder.max_seq_length = embedding_cfg["max_seq_length"]

        test_vec = self.embedder.encode(
            ["dimension check"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if test_vec.shape[1] != self.index.d:
            raise RuntimeError(
                f"DIMENSION MISMATCH: embedding model outputs {test_vec.shape[1]}d "
                f"but FAISS index expects {self.index.d}d."
            )
        print(f"   Dimension check passed ({test_vec.shape[1]}d)")

        self.bm25 = None
        if retrieval_cfg["use_bm25"] and _BM25_INSTALLED:
            print("   Building BM25 index...")
            corpus = [m["text"].lower().split() for m in self.metadata]
            self.bm25 = BM25Okapi(corpus)
            print("   BM25 ready")
        else:
            print("   BM25 disabled (config or missing package)")

        self.reranker = None
        if retrieval_cfg["use_reranker"] and retrieval_cfg.get("rerank_model"):
            if _CROSSENCODER_AVAILABLE:
                print(f"   Loading reranker: {retrieval_cfg['rerank_model']}")
                try:
                    self.reranker = CrossEncoder(
                        retrieval_cfg["rerank_model"],
                        device=self._device,
                    )
                    print("   Cross-encoder reranker ready")
                except Exception as e:
                    print(f"   Warning: reranker failed: {e} - continuing without it")
            else:
                print("   Warning: CrossEncoder not available")

        print("Retriever ready\n")

    def search(
        self,
        query: str,
        top_k: int = None,
        candidate_pool: int = None,
        stitching_window: int = None,
        book_filter: str = None,
        source_filter: list[str] = None,
    ) -> list[dict]:
        top_k = top_k or retrieval_cfg["top_k"]
        candidate_pool = candidate_pool or retrieval_cfg["candidate_pool"]
        stitching_window = stitching_window or retrieval_cfg["stitching_window"]

        enriched_query = self._enrich_query(query)
        faiss_hits = self._faiss_search(enriched_query, k=candidate_pool)
        bm25_hits = self._bm25_search(query, k=candidate_pool) if self.bm25 else []
        merged = self._rrf_merge(faiss_hits, bm25_hits, k=candidate_pool)

        if book_filter:
            merged = [c for c in merged if book_filter.lower() in c["source"].lower()]
        if source_filter:
            merged = [c for c in merged if c["source"] in source_filter]

        merged = self._apply_query_grounding(query, merged)
        stitched = self._stitch_chunks(merged[:top_k], window=stitching_window)

        if self.reranker and len(stitched) > 1:
            stitched = self._rerank(query, stitched, top_k=top_k)
        else:
            stitched = stitched[:top_k]

        return stitched

    def inspect(
        self,
        query: str,
        top_k: int = None,
        candidate_pool: int = None,
        stitching_window: int = None,
        book_filter: str = None,
        source_filter: list[str] = None,
    ) -> dict:
        top_k = top_k or retrieval_cfg["top_k"]
        candidate_pool = candidate_pool or retrieval_cfg["candidate_pool"]
        stitching_window = stitching_window or retrieval_cfg["stitching_window"]

        enriched_query = self._enrich_query(query)
        faiss_hits = self._faiss_search(enriched_query, k=candidate_pool)
        bm25_hits = self._bm25_search(query, k=candidate_pool) if self.bm25 else []
        merged = self._rrf_merge(faiss_hits, bm25_hits, k=candidate_pool)

        if book_filter:
            merged = [c for c in merged if book_filter.lower() in c["source"].lower()]
        if source_filter:
            merged = [c for c in merged if c["source"] in source_filter]

        grounded = self._apply_query_grounding(query, merged)
        stitched = self._stitch_chunks(grounded[:top_k], window=stitching_window)
        return {
            "query": query,
            "query_terms": self._query_terms(query),
            "faiss_hits": faiss_hits[: min(5, len(faiss_hits))],
            "bm25_hits": bm25_hits[: min(5, len(bm25_hits))],
            "grounded_hits": grounded[: min(top_k, len(grounded))],
            "stitched_hits": stitched,
        }

    def _faiss_search(self, query: str, k: int) -> list[dict]:
        vec = self.embedder.encode(
            [query],
            normalize_embeddings=embedding_cfg["normalize"],
            convert_to_numpy=True,
        ).astype("float32")
        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(vec, actual_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            results.append({
                "chunk_id": m.get("chunk_id", idx),
                "text": m.get("text", ""),
                "source": m.get("source", "unknown"),
                "chapter": m.get("chapter", "unknown"),
                "file_type": m.get("file_type", "pdf"),
                "faiss_rank": rank,
                "faiss_score": float(score),
            })
        return results

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_idx):
            if idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            results.append({
                "chunk_id": m.get("chunk_id", int(idx)),
                "text": m.get("text", ""),
                "source": m.get("source", "unknown"),
                "chapter": m.get("chapter", "unknown"),
                "file_type": m.get("file_type", "pdf"),
                "bm25_rank": rank,
                "bm25_score": float(scores[idx]),
            })
        return results

    def _rrf_merge(
        self,
        faiss_hits: list[dict],
        bm25_hits: list[dict],
        k: int,
        rrf_k: int = None,
    ) -> list[dict]:
        rrf_k = rrf_k or retrieval_cfg.get("rrf_k", 60)
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, doc in enumerate(faiss_hits):
            key = _md5(doc["text"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            docs[key] = doc

        for rank, doc in enumerate(bm25_hits):
            key = _md5(doc["text"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in docs:
                docs[key] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        merged = []
        for key, score in ranked[:k]:
            d = dict(docs[key])
            d["rrf_score"] = score
            merged.append(d)
        return merged

    def _apply_query_grounding(self, query: str, docs: list[dict]) -> list[dict]:
        query_terms = self._query_terms(query)
        for doc in docs:
            haystack = f"{doc.get('source', '')} {doc.get('chapter', '')} {doc.get('text', '')}".lower()
            matched_terms = sorted(term for term in query_terms if self._term_matches(term, haystack))
            overlap_score = len(matched_terms)
            doc["query_overlap_terms"] = matched_terms
            doc["query_overlap_score"] = float(overlap_score)

        strong = [d for d in docs if d["query_overlap_score"] > 0]
        weak = [d for d in docs if d["query_overlap_score"] == 0]

        strong.sort(
            key=lambda d: (
                d.get("query_overlap_score", 0.0),
                d.get("rerank_score", d.get("rrf_score", d.get("faiss_score", 0.0))),
            ),
            reverse=True,
        )
        weak.sort(
            key=lambda d: d.get("rrf_score", d.get("faiss_score", 0.0)),
            reverse=True,
        )
        return strong + weak

    def _stitch_chunks(self, hits: list[dict], window: int) -> list[dict]:
        if window == 0:
            return hits

        seen_ids: set[int] = set()
        stitched: list[dict] = []
        for hit in hits:
            center_id = hit.get("chunk_id")
            if center_id is None:
                stitched.append(hit)
                continue

            neighbour_ids = list(range(max(0, center_id - window), center_id + window + 1))
            parts = []
            valid_ids = []
            for nid in neighbour_ids:
                if nid in seen_ids:
                    continue
                if nid in self._id_to_idx:
                    chunk = self.metadata[self._id_to_idx[nid]]
                    parts.append(chunk.get("text", "").strip())
                    valid_ids.append(nid)

            if not parts:
                stitched.append(hit)
                continue

            seen_ids.update(valid_ids)
            stitched_doc = dict(hit)
            stitched_doc["text"] = "\n\n".join(parts)
            stitched_doc["stitch_range"] = f"chunk_id {valid_ids[0]}-{valid_ids[-1]}"
            stitched.append(stitched_doc)

        return stitched

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(
            key=lambda x: (
                x.get("rerank_score", 0.0),
                x.get("query_overlap_score", 0.0),
            ),
            reverse=True,
        )
        return candidates[:top_k]

    def _query_terms(self, query: str) -> set[str]:
        raw_terms = re.findall(r"[a-z0-9']+", query.lower())
        terms = {term for term in raw_terms if len(term) > 2 and term not in STOPWORDS}
        expanded = set()
        for term in terms:
            expanded.update(TERM_VARIANTS.get(term, {term}))
        return expanded or terms

    def _term_matches(self, term: str, haystack: str) -> bool:
        variants = TERM_VARIANTS.get(term, {term})
        return any(variant in haystack for variant in variants)

    def _enrich_query(self, query: str) -> str:
        return f"Represent this sentence for searching relevant passages: {query}"

    def _resolve_device(self) -> str:
        setting = embedding_cfg.get("device", "auto")
        if setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return setting

    @staticmethod
    def _check_file(filepath: str, label: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"[FATAL] {label} not found at: {filepath}\n"
                f"  Lenovo: run Scripts/build_db.py first.\n"
                f"  Dell:   copy Db/faiss.index and Db/metadata.json from Lenovo."
            )


def _md5(text: str) -> str:
    return md5(text.encode("utf-8", errors="ignore")).hexdigest()
