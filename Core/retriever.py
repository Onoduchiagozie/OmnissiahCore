"""
OmnissiahCore - Core/retriever.py

Hybrid RAG retriever:
  1. FAISS dense vector search
  2. BM25 sparse keyword search
  3. Reciprocal Rank Fusion
  4. Query-overlap grounding boost
  5. Chunk stitching with global text deduplication
  6. Optional cross-encoder rerank

  7. Focus mode (search_focus): narrows results to at most `max_sources`
     stitched groups. Tries the pre-woven scene library
     (battle_scenes.db, via a small side FAISS+BM25 index built by
     Scripts/build_scene_index.py) first; falls back to a narrowed pass
     over the full corpus if no scene matches confidently.
"""

import json
import os
import re
import sqlite3
from hashlib import md5

import faiss
import torch
from sentence_transformers import SentenceTransformer

from Core.config_loader import embedding_cfg, machine_role, paths, retrieval_cfg

try:
    import bm25s

    _BM25_INSTALLED = True
except ImportError:
    _BM25_INSTALLED = False
    print("Warning: bm25s not installed. BM25 disabled.")

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

# Cosine-similarity floor (IndexFlatIP over normalized vectors) that a scene
# result must clear before search_focus() trusts the pre-woven scene library
# over falling back to a live narrowed corpus search.
#
# Raised from 0.42 -> 0.55 after real testing: query "string" (meaningless)
# still scored 0.4523 on its top scene match, dangerously close to a real
# query like "the destruction of Prospero" (0.6207). 0.42 wasn't leaving
# enough daylight between signal and noise. Still not in config.json —
# keep tuning against real queries before promoting it to retrieval_cfg.
FOCUS_CONFIDENCE_THRESHOLD = 0.55

# The MS MARCO cross-encoder emits a relevance logit.  Zero is its natural
# relevance boundary: negative candidates are less likely than not to answer
# the query and must not be passed to generation as evidence.
RERANK_RELEVANCE_FLOOR = 0.0

# Absolute ceiling on focus-mode source count regardless of what's requested.
FOCUS_MAX_SOURCES_CEILING = 3


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
            self.bm25 = self._load_or_build_bm25()
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

        # --- Scene Archive index (optional — focus mode only) ---
        self.scene_index = None
        self.scene_metadata: list[dict] = []
        self.scene_bm25 = None
        self._load_scene_index()

        print("Retriever ready\n")

    # ------------------------------------------------------------------
    # Main corpus search (unchanged from before)
    # ------------------------------------------------------------------

    def search(
            self,
            query: str,
            top_k: int = None,
            candidate_pool: int = None,
            stitching_window: int = None,
            book_filter: str = None,
            source_filter: list[str] = None,
            on_stage=None,  # optional callable(str) -> None, fired at each real boundary
    ) -> list[dict]:
        top_k = top_k or retrieval_cfg["top_k"]
        candidate_pool = candidate_pool or retrieval_cfg["candidate_pool"]
        stitching_window = retrieval_cfg["stitching_window"] if stitching_window is None else stitching_window

        if on_stage:
            on_stage("embedding")
        enriched_query = self._enrich_query(query)
        faiss_hits = self._faiss_search(enriched_query, k=candidate_pool)

        if on_stage:
            on_stage("retrieving")
        bm25_hits = self._bm25_search(query, k=candidate_pool) if self.bm25 else []
        merged = self._rrf_merge(faiss_hits, bm25_hits, k=candidate_pool)

        if book_filter:
            merged = [c for c in merged if book_filter.lower() in c["source"].lower()]
        if source_filter:
            merged = [c for c in merged if c["source"] in source_filter]

        merged = self._apply_query_grounding(query, merged)
        # A cross-encoder must see the full retrieval candidate set.
        # Reranking after truncating to top_k only reshuffles results the
        # lexical-overlap stage already selected.
        stitch_limit = candidate_pool if self.reranker else top_k
        stitched = self._stitch_chunks(merged[:stitch_limit], window=stitching_window)

        if self.reranker and len(stitched) > 1:
            if on_stage:
                on_stage("reranking")
            reranked = self._rerank(query, stitched, top_k=top_k)
            return [
                chunk for chunk in reranked
                if chunk.get("rerank_score", float("-inf")) >= RERANK_RELEVANCE_FLOOR
            ]

        return stitched[:top_k]

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
        stitching_window = retrieval_cfg["stitching_window"] if stitching_window is None else stitching_window

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
            # Full real selection (cross-encoder and score floor included),
            # unlike stitched_hits, which only shows the raw diagnostic stage.
            "selected_hits": self.search(
                query=query, top_k=top_k, candidate_pool=candidate_pool,
                stitching_window=stitching_window,
                book_filter=book_filter, source_filter=source_filter,
            ),
        }

    # ------------------------------------------------------------------
    # Focus mode — new
    # ------------------------------------------------------------------

    # CHANGED: added book_filter/source_filter params — previously this
    # method had no filtering ability at all, so agent.py had nothing to
    # forward them into even though it accepted them.
    def search_focus(
            self,
            query: str,
            max_sources: int = 2,
            book_filter: str = None,
            source_filter: list[str] = None,
            on_stage=None,
    ) -> list[dict]:
        """
        Narrows to at most `max_sources` stitched groups (books don't have
        to differ — two groups from the same book are fine).

        Path A (fast): search the pre-woven scene library
        (battle_scenes.db, via self.scene_index). If the top hit clears
        FOCUS_CONFIDENCE_THRESHOLD, return up to max_sources scenes as
        chunk-shaped dicts, no CrossEncoder pass needed — they're already
        curated prose.

        Path B (fallback): if the scene library isn't loaded, or nothing
        in it scores confidently (the query is about something outside
        the pre-captured battle/confrontation scenes), fall back to the
        normal corpus pipeline with top_k pinned to max_sources and the
        stitching window widened, so what comes back is fewer, fuller
        groups instead of scattered fragments.

        CHANGED: book_filter/source_filter now apply in both paths.
        Scene hits are filtered against their book_title BEFORE the
        confidence check, so a scene that doesn't match the filter can't
        gate whether we trust the scene library at all — a wrong-book top
        hit now correctly falls through to the live fallback instead of
        silently blocking it. The live fallback forwards both filters into
        search() exactly as narrator/wide-net mode already did.
        """
        max_sources = max(1, min(max_sources or 2, FOCUS_MAX_SOURCES_CEILING))

        if on_stage:
            on_stage("embedding")

        scene_hits = self._search_scene_index(query, k=10) if self.scene_index is not None else []

        # CHANGED: filter scene hits by book_filter/source_filter before the
        # confidence gate runs, so a confidently-scored wrong-book scene
        # can't block the live fallback from being tried instead.
        if scene_hits and book_filter:
            scene_hits = [
                s for s in scene_hits
                if book_filter.lower() in (s.get("book_title") or "").lower()
            ]
        if scene_hits and source_filter:
            # NOTE: scene_metadata's book_title is a display title (e.g.
            # "Betrayer"), not the raw corpus filename source_filter is
            # built to match exactly (e.g. "12. Aaron Dembski-Bowden -
            # Betrayer.pdf"). This will likely miss real matches on the
            # scene path — book_filter is the reliable one here. Left in
            # for parity with search(), but don't trust source_filter to
            # narrow scene results correctly until book_title vs source
            # naming gets reconciled.
            scene_hits = [s for s in scene_hits if s.get("book_title") in source_filter]

        top_confident = (
            scene_hits
            and scene_hits[0]["scene_score"] >= FOCUS_CONFIDENCE_THRESHOLD
            and self._scene_has_query_grounding(query, scene_hits[0])
        )

        if top_confident:
            if on_stage:
                on_stage("retrieving")
            # Grounding was already confirmed for scene_hits[0] above (that's
            # what made top_confident True). But taking the next max_sources-1
            # entries by RRF rank alone let weakly-related runners-up ride
            # along on the strength of the #1 result — e.g. a real Isstvan V
            # match paired with an unrelated Codex entry that merely ranked
            # #2. Every selected scene now has to individually pass the same
            # grounding check, not just the top one.
            grounded_scenes = [
                s for s in scene_hits
                if self._scene_has_query_grounding(query, s)
            ]
            top_scenes = grounded_scenes[:max_sources]
            return self._scenes_to_chunks(top_scenes)

        # Fallback: narrowed live corpus search
        if on_stage:
            on_stage("retrieving")
        base_window = retrieval_cfg["stitching_window"]
        widened_window = min(6, max(base_window * 2, base_window + 2))

        # CHANGED: book_filter/source_filter now forwarded here — previously
        # this call to search() left both out entirely.
        stitched = self.search(
            query=query,
            top_k=max_sources,
            candidate_pool=retrieval_cfg["candidate_pool"],
            stitching_window=widened_window,
            book_filter=book_filter,
            source_filter=source_filter,
            on_stage=None,  # already fired embedding/retrieving above
        )
        return stitched

    def search_ghost(
            self,
            query: str,
            max_sources: int = 2,
            book_filter: str = None,
            source_filter: list[str] = None,
    ) -> list[dict]:
        """
        Pure scene-library retrieval. Zero LLM. Zero thread-pool usage.
        Zero network calls.

        Same confidence gate and lexical grounding check as search_focus()'s
        fast path — but with NO fallback whatsoever. If the scene library
        doesn't have a confident, grounded match, this returns [] immediately.
        The /query/ghost route turns that into a clean "matched: false" response.

        This is not "lower accuracy" — for high-confidence scene matches,
        returning the pre-woven chronicle verbatim IS the most accurate thing
        possible. The LLM can only add hallucination risk on top of already-
        correct prose. Ghost mode is strictly better than focus mode for
        queries the scene library already covers.

        Uses data already in RAM (self.scene_index, self.scene_metadata) —
        the only I/O is one SQLite lookup per matched scene to fetch the
        full chronicle text from battle_scenes.db.
        """
        if self.scene_index is None:
            return []

        max_sources = max(1, min(max_sources or 2, FOCUS_MAX_SOURCES_CEILING))

        scene_hits = self._search_scene_index(query, k=10)

        if book_filter or source_filter:
            scene_hits = [
                s for s in scene_hits
                if self._scene_matches_filters(s, book_filter, source_filter)
            ]

        if not scene_hits:
            return []

        # Confidence gate — same threshold as focus mode's fast path.
        top_confident = (
                scene_hits[0]["scene_score"] >= FOCUS_CONFIDENCE_THRESHOLD
                and self._scene_has_query_grounding(query, scene_hits[0])
        )
        if not top_confident:
            return []

        # Every returned scene must individually pass grounding,
        # not just ride along on the top result's score.
        grounded = [
            s for s in scene_hits
            if self._scene_has_query_grounding(query, s)
        ]
        return self._scenes_to_chunks(grounded[:max_sources])
    
    def _scene_has_query_grounding(self, query: str, scene_meta: dict) -> bool:
        """
        Semantic similarity alone isn't proof of topical relevance — two
        unrelated battle scenes can score a high cosine similarity just
        because they're both full of explosions and desperate escapes.
        This requires at least one real, specific query term (stopwords
        like 'attack'/'battle' already excluded by _query_terms) to
        actually appear somewhere in the matched scene's title, teaser,
        book title, or chronicle preview. No overlap at all is a strong
        signal the FAISS score is a vocabulary coincidence, not a real
        match — in that case search_focus() falls back to live search
        instead of trusting it.
        """
        terms = self._query_terms(query)
        if not terms:
            return True  # nothing specific to check against — don't block on this alone

        haystack = " ".join([
            scene_meta.get("scene_name") or "",
            scene_meta.get("teaser") or "",
            scene_meta.get("book_title") or "",
            scene_meta.get("chronicle_preview") or "",
        ]).lower()

        return any(self._term_matches(term, haystack) for term in terms)

    def _load_scene_index(self):
        scene_faiss_path = paths.get("scene_faiss")
        scene_metadata_path = paths.get("scene_metadata")

        if not scene_faiss_path or not os.path.exists(scene_faiss_path):
            print("   Scene index not found — focus mode will use live fallback only.")
            print("   (Run Scripts/build_scene_index.py to enable the fast scene-library path.)")
            return

        try:
            self.scene_index = faiss.read_index(scene_faiss_path)
            with open(scene_metadata_path, "r", encoding="utf-8") as f:
                self.scene_metadata = json.load(f)
            print(f"   Scene index loaded: {self.scene_index.ntotal:,} scenes")
        except Exception as e:
            print(f"   Warning: failed to load scene index: {e} — focus mode will use live fallback only.")
            self.scene_index = None
            return

        scene_bm25_cache = paths.get("scene_bm25_cache")
        if scene_bm25_cache and os.path.isdir(scene_bm25_cache) and _BM25_INSTALLED:
            try:
                self.scene_bm25 = bm25s.BM25.load(scene_bm25_cache, load_corpus=False)
                print("   Scene BM25 index loaded")
            except Exception as e:
                print(f"   Warning: scene BM25 load failed: {e} — scene search will be FAISS-only")

    def _search_scene_index(self, query: str, k: int = 10) -> list[dict]:
        """
        Returns scene metadata dicts ranked by RRF-merged FAISS+BM25 score.
        Each item is annotated with its OWN real FAISS cosine similarity as
        'scene_score' (0.0 if it only surfaced via BM25). search_focus()
        uses the top-ranked item's scene_score as the confidence gate —
        RRF scores aren't on a comparable 0-1 scale, so the gate checks the
        FAISS score directly, not the fused rank.
        """
        vec = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        actual_k = min(k, self.scene_index.ntotal)
        faiss_scores, faiss_idx = self.scene_index.search(vec, actual_k)

        faiss_hits = [
            {"scene_idx": int(idx), "rank": rank, "score": float(score)}
            for rank, (idx, score) in enumerate(zip(faiss_idx[0], faiss_scores[0]))
            if idx >= 0
        ]

        bm25_hits = []
        if self.scene_bm25 is not None:
            query_tokens = bm25s.tokenize([query], stopwords=None, show_progress=False)
            bm25_k = min(k, len(self.scene_metadata))
            top_idx_arr, scores_arr = self.scene_bm25.retrieve(
                query_tokens, k=bm25_k, show_progress=False
            )
            bm25_hits = [
                {"scene_idx": int(idx), "rank": rank, "score": float(score)}
                for rank, (idx, score) in enumerate(zip(top_idx_arr[0], scores_arr[0]))
            ]

        rrf_k = retrieval_cfg.get("rrf_k", 60)
        scores: dict[int, float] = {}
        for h in faiss_hits:
            scores[h["scene_idx"]] = scores.get(h["scene_idx"], 0.0) + 1.0 / (rrf_k + h["rank"])
        for h in bm25_hits:
            scores[h["scene_idx"]] = scores.get(h["scene_idx"], 0.0) + 1.0 / (rrf_k + h["rank"])

        # Per-item real FAISS cosine score, so every returned scene carries
        # its OWN genuine similarity value — not just whichever item happened
        # to rank #1 overall. A scene that only surfaced via BM25 (not in
        # faiss_hits at all) gets 0.0 here, which is honest: no semantic
        # score exists for it.
        faiss_score_by_idx = {h["scene_idx"]: h["score"] for h in faiss_hits}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for scene_idx, rrf_score in ranked:
            if scene_idx >= len(self.scene_metadata):
                continue
            meta = dict(self.scene_metadata[scene_idx])
            meta["rrf_score"] = rrf_score
            meta["scene_score"] = faiss_score_by_idx.get(scene_idx, 0.0)
            results.append(meta)

        return results

    def _scenes_to_chunks(self, scenes: list[dict]) -> list[dict]:
        """
        Reshapes scene records into the same chunk-dict shape the rest of
        the pipeline (prompt builders, SourceInfo, stitching) already
        expects. Full chronicle text is fetched live from battle_scenes.db
        by scene_id — scene_metadata.json only holds a short preview to
        keep that file small.
        """
        scene_ids = [s.get("scene_id") for s in scenes if s.get("scene_id") is not None]
        chronicles = self._fetch_scene_chronicles(scene_ids)

        chunks = []
        for s in scenes:
            scene_id = s.get("scene_id")
            chronicle = chronicles.get(scene_id) or s.get("chronicle_preview", "")
            chunks.append({
                "chunk_id": None,
                "text": chronicle,
                "source": s.get("book_title", "unknown"),
                "chapter": s.get("scene_name", "unknown"),
                "file_type": "scene",
                "stitch_range": f"pre-woven scene, rank {s.get('rank', '?')}",
                "rrf_score": s.get("rrf_score", 0.0),
                "scene_score": s.get("scene_score", 0.0),
                "scene_id": scene_id,
                "query_overlap_score": 0.0,
            })
        return chunks

    def _fetch_scene_chronicles(self, scene_ids: list[int]) -> dict[int, str]:
        if not scene_ids:
            return {}
        battle_db_path = paths.get("battle_scenes_db")
        if not battle_db_path or not os.path.exists(battle_db_path):
            return {}
        try:
            conn = sqlite3.connect(battle_db_path)
            placeholders = ",".join("?" for _ in scene_ids)
            rows = conn.execute(
                f"SELECT scene_id, chronicle FROM scenes WHERE scene_id IN ({placeholders})",
                scene_ids,
            ).fetchall()
            conn.close()
            return {row[0]: row[1] for row in rows}
        except Exception as e:
            print(f"   Warning: failed to fetch chronicle text from battle_scenes.db: {e}")
            return {}

    # ------------------------------------------------------------------
    # Internals shared by both search paths (unchanged from before)
    # ------------------------------------------------------------------

    def _load_or_build_bm25(self):
        """
        Loads a saved bm25s index from disk if one exists and is newer than
        metadata.json. Otherwise builds from scratch and saves for next startup.

        bm25s stores term frequencies as compact sparse arrays instead of
        rank_bm25's Python list-of-list-of-strings, which is what was costing
        roughly 12-13GB of peak RAM on first build. Loading a saved bm25s
        index is also far cheaper than rebuilding from raw text every time.
        """
        cache_dir = paths.get("bm25_cache")
        metadata_path = paths["metadata"]

        if cache_dir and os.path.isdir(cache_dir):
            cache_mtime = os.path.getmtime(cache_dir)
            metadata_mtime = os.path.getmtime(metadata_path)
            if cache_mtime >= metadata_mtime:
                print("   Loading cached BM25 index (bm25s)...")
                try:
                    bm25 = bm25s.BM25.load(cache_dir, load_corpus=False)
                    print("   BM25 ready (from cache)")
                    return bm25
                except Exception as e:
                    print(f"   Warning: BM25 cache load failed ({e}) - rebuilding")
            else:
                print("   BM25 cache is stale (metadata.json changed) - rebuilding")

        print("   Building BM25 index (this is the expensive one-time cost)...")
        corpus_texts = [m.get("text", "") for m in self.metadata]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords=None, show_progress=False)
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens, show_progress=False)
        print("   BM25 ready")

        if cache_dir:
            try:
                bm25.save(cache_dir)
                print(f"   BM25 index cached to: {cache_dir}")
            except Exception as e:
                print(f"   Warning: failed to write BM25 cache: {e}")

        return bm25

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
                "chunk_id": int(m.get("chunk_id", idx)),
                "text": m.get("text", ""),
                "source": m.get("source", "unknown"),
                "chapter": m.get("chapter", "unknown"),
                "file_type": m.get("file_type", "pdf"),
                "faiss_rank": rank,
                "faiss_score": float(score),
            })
        return results

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        query_tokens = bm25s.tokenize([query], stopwords=None, show_progress=False)
        actual_k = min(k, len(self.metadata))
        top_idx_arr, scores_arr = self.bm25.retrieve(query_tokens, k=actual_k, show_progress=False)
        top_idx = top_idx_arr[0]
        scores = scores_arr[0]

        results = []
        for rank, (idx, score) in enumerate(zip(top_idx, scores)):
            idx = int(idx)
            if idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            results.append({
                "chunk_id": m.get("chunk_id", idx),
                "text": m.get("text", ""),
                "source": m.get("source", "unknown"),
                "chapter": m.get("chapter", "unknown"),
                "file_type": m.get("file_type", "pdf"),
                "bm25_rank": rank,
                "bm25_score": float(score),
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
        """
        Stitches surrounding neighbor chunks to establish context window tracking.
        Applies strict Global Text Deduplication to neutralize duplicate book entries
        (e.g., cross-format duplication between EPUB and PDF editions) before context injection.
        """
        if window == 0:
            return hits

        seen_ids: set[int] = set()
        seen_texts_normalized: set[str] = set()
        stitched: list[dict] = []

        for hit in hits:
            center_id = hit.get("chunk_id")
            if center_id is None:
                # Fallback hashing pattern for unbound chunk segments
                text_hash = _md5(hit.get("text", "").strip().lower())
                if text_hash in seen_texts_normalized:
                    continue
                seen_texts_normalized.add(text_hash)
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
                    # Chunk IDs are global, so never stitch across sources.
                    if chunk.get("source") != hit.get("source"):
                        continue
                    chunk_text = chunk.get("text", "").strip()

                    # Normalize white-spaces and look for string matches across overlapping chunks
                    normalized_chunk = re.sub(r"\s+", " ", chunk_text.lower())
                    if normalized_chunk in seen_texts_normalized:
                        continue

                    parts.append(chunk_text)
                    valid_ids.append(nid)
                    seen_texts_normalized.add(normalized_chunk)

            if not parts:
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
        # Controlled A/B experiment: BGE-M3 does not require BGE-v1.5 instructions.
        if (retrieval_cfg.get("bge_m3_raw_queries", False)
                and embedding_cfg.get("model", "").lower().endswith("bge-m3")):
            return query
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
