"""
OmnissiahCore - Api/services/runtime_service.py

Key changes vs previous version:
- Semaphore cap=1: only one LLM request runs at a time. Previously the
  ThreadPoolExecutor allowed up to 4 concurrent LLM calls, which caused
  LM Studio / Ollama to choke and return errors for ALL callers, not
  just the excess ones. Now the second caller waits cleanly instead of
  both dying.
- _query_semaphore + _active_queries counter exposed via get_llm_status()
  for the GET /llm/status traffic-light endpoint.
- metadata.json double-load fixed: OmnissiahRetriever already loads it
  into self.metadata; RuntimeService now just points at that same object
  instead of reading the file a second time.
"""

import json as _json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import faiss
import numpy as np

from Api.models import QueryRequest
from Core.agent import OmnissiahAgent
from Core.config_loader import active_profile, machine_role, llm_cfg, paths, retrieval_cfg
from Core.prompt import build_prompt
from Core.retriever import OmnissiahRetriever


# ---------------------------------------------------------------------------
# Concurrency control
# ---------------------------------------------------------------------------

# Cap at 1: only one LLM generation runs at a time.
# LM Studio / Ollama serialises anyway — allowing >1 concurrent just meant
# both callers got errors instead of the second one waiting cleanly.
_LLM_MAX_CONCURRENT = 1
_query_semaphore = threading.Semaphore(_LLM_MAX_CONCURRENT)
_active_queries = 0
_queued_queries = 0
_concurrency_lock = threading.Lock()


def _acquire_llm_slot():
    global _active_queries, _queued_queries
    with _concurrency_lock:
        _queued_queries += 1
    _query_semaphore.acquire()
    with _concurrency_lock:
        _queued_queries -= 1
        _active_queries += 1


def _release_llm_slot():
    global _active_queries
    with _concurrency_lock:
        _active_queries -= 1
    _query_semaphore.release()


def get_llm_status() -> dict:
    with _concurrency_lock:
        active = _active_queries
        queued = _queued_queries

    if active == 0:
        return {
            "status": "green",
            "active": active,
            "queued": queued,
            "message": "Ready",
            "max_concurrent": _LLM_MAX_CONCURRENT,
        }
    elif active >= _LLM_MAX_CONCURRENT and queued == 0:
        return {
            "status": "amber",
            "active": active,
            "queued": queued,
            "message": "LLM busy — your query will begin shortly",
            "max_concurrent": _LLM_MAX_CONCURRENT,
        }
    else:
        return {
            "status": "amber",
            "active": active,
            "queued": queued,
            "message": f"LLM busy — {queued} query waiting",
            "max_concurrent": _LLM_MAX_CONCURRENT,
        }


def _sanitize_numpy(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_numpy(i) for i in obj]
    if isinstance(obj, set):
        return [_sanitize_numpy(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def clean_response(text: str) -> str:
    if not text:
        return text
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')
    text = re.sub(r'\s*/\s*', ' ', text)
    text = re.sub(r'(?<!\w)/(?!\w)', ' ', text)
    text = re.sub(r'\\\s*', ' ', text)
    text = re.sub(r'\\/', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


class RuntimeService:
    def __init__(self):
        self._retriever: Optional[OmnissiahRetriever] = None
        self._session_memory: dict[str, list[dict]] = {}
        self._memory_lock = threading.Lock()
        # NOT loading metadata here — retriever loads it, we reuse that.
        self._metadata_cache: list[dict] = []

    @property
    def retriever(self) -> Optional[OmnissiahRetriever]:
        return self._retriever

    def startup(self):
        print("[Startup] Loading retriever...")
        self._retriever = OmnissiahRetriever()
        # Fix: point at retriever's already-loaded metadata instead of
        # reading metadata.json a second time from disk.
        self._metadata_cache = self._retriever.metadata
        print("[Startup] Runtime ready")

    def _get_session_memory(self, session_id: str) -> list[dict]:
        with self._memory_lock:
            return list(self._session_memory.get(session_id, []))

    def _set_session_memory(self, session_id: str, memory: list[dict]):
        with self._memory_lock:
            self._session_memory[session_id] = memory

    def _build_agent(self, session_id: str, mode: str) -> OmnissiahAgent:
        agent = OmnissiahAgent(verbose=False, mode=mode, retriever=self._retriever)
        agent._memory = self._get_session_memory(session_id)
        return agent

    def ensure_ready(self):
        if not self._retriever:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="Retriever not initialised. The service is still starting up."
            )

    def run_query(self, req: QueryRequest, mode: str, stream: bool) -> tuple[str, list[dict]]:
        session_id = req.session_id or "default"
        agent = self._build_agent(session_id, mode)

        # Ghost mode never acquires the LLM slot — it never calls the LLM.
        if mode == "ghost":
            chunks = self._retriever.search_ghost(
                query=req.query,
                max_sources=req.max_sources,
                book_filter=req.book_filter,
                source_filter=req.source_filter,
            )
            return "", chunks

        # All other modes acquire the semaphore — only 1 LLM call at a time.
        _acquire_llm_slot()
        try:
            response, chunks = agent.ask(
                query=req.query,
                book_filter=req.book_filter,
                source_filter=req.source_filter,
                top_k=req.top_k,
                candidate_pool=req.candidate_pool,
                stitching_window=req.stitching_window,
                max_sources=req.max_sources,
                stream=stream,
            )
        finally:
            _release_llm_slot()

        self._set_session_memory(session_id, agent.memory)
        response = clean_response(response)
        return response, chunks

    def stream_query_mode(self, req: QueryRequest, mode: str = "remembrancer"):
        session_id = req.session_id or "default"
        agent = self._build_agent(session_id, mode=mode)
        pending_stages: list[str] = []

        def _on_stage(stage_name: str):
            pending_stages.append(stage_name)

        # Acquire semaphore before streaming starts — holds it for the
        # entire stream duration so a second caller waits, not crashes.
        _acquire_llm_slot()
        try:
            stream = agent.ask_stream(
                query=req.query,
                book_filter=req.book_filter,
                source_filter=req.source_filter,
                top_k=req.top_k,
                candidate_pool=req.candidate_pool,
                stitching_window=req.stitching_window,
                max_sources=req.max_sources,
                on_stage=_on_stage,
            )

            stream_failed = False
            for token in stream:
                while pending_stages:
                    stage_name = pending_stages.pop(0)
                    yield f"data: __STAGE__:{stage_name}\n\n"

                if token.startswith("__SOURCES__:"):
                    yield f"data: {token}\n\n"
                    continue

                clean_token = token.replace(" / ", " ").replace("/", " ")
                if clean_token.startswith("[ERROR]"):
                    stream_failed = True
                # Every line of an SSE event needs its own data prefix.
                # Unprefixed paragraph lines are discarded by SSE clients.
                yield "\n".join("data: " + line for line in clean_token.split("\n")) + "\n\n"

            while pending_stages:
                stage_name = pending_stages.pop(0)
                yield f"data: __STAGE__:{stage_name}\n\n"

            final_stage = "failed" if stream_failed else "complete"
            yield f"data: __STAGE__:{final_stage}\n\n"
            self._set_session_memory(session_id, agent.memory)

        finally:
            _release_llm_slot()

    def inspect_query(self, req: QueryRequest) -> dict:
        inspection = self._retriever.inspect(
            query=req.query,
            book_filter=req.book_filter,
            source_filter=req.source_filter,
            top_k=req.top_k,
            candidate_pool=req.candidate_pool,
            stitching_window=req.stitching_window,
        )
        inspection = _sanitize_numpy(inspection)
        system_prompt, user_message = build_prompt(req.query, inspection["stitched_hits"])
        return {
            "inspection": inspection,
            "prompt_preview": {
                "system_prompt": system_prompt[:4000],
                "user_message": user_message,
            },
        }

    def health_payload(self) -> dict:
        return {
            "status": "online" if self._retriever else "starting",
            "active_profile": active_profile,
            "machine_role": machine_role,
            "llm_model": llm_cfg["model"],
            "llm_url": llm_cfg["url"],
            "metadata_loaded": len(self._metadata_cache),
            "llm_status": get_llm_status(),
        }

    def info_payload(self) -> dict:
        try:
            index = faiss.read_index(paths["faiss"])
            n_vectors = index.ntotal
            dim = index.d
        except Exception:
            n_vectors = -1
            dim = -1

        manifest = {}
        if os.path.exists(paths["manifest"]):
            with open(paths["manifest"], "r", encoding="utf-8") as f:
                manifest = _json.load(f)

        return {
            "index_vectors": n_vectors,
            "index_dim": dim,
            "machine_role": machine_role,
            "manifest": manifest,
            "retrieval": retrieval_cfg,
            "cached_sources": len(set(m.get("source", "unknown") for m in self._metadata_cache)),
        }

    def runtime_config_payload(self) -> dict:
        return {
            "active_profile": active_profile,
            "machine_role": machine_role,
            "llm": {
                "url": llm_cfg["url"],
                "model": llm_cfg["model"],
                "timeout": llm_cfg["timeout"],
                "temperature": llm_cfg["temperature"],
                "top_p": llm_cfg["top_p"],
                "max_tokens": llm_cfg.get("max_tokens"),
            },
            "retrieval": retrieval_cfg,
            "paths": {
                "db_dir": paths["db_dir"],
                "faiss": paths["faiss"],
                "metadata": paths["metadata"],
            },
        }

    def list_sources_payload(self) -> dict:
        sources = sorted(set(m.get("source", "unknown") for m in self._metadata_cache))
        return {"total": len(sources), "sources": sources}

    def source_chunks_payload(self, source_name: str, limit: int) -> dict:
        matched = [
            m for m in self._metadata_cache
            if source_name.lower() in m.get("source", "").lower()
        ][:limit]
        return {
            "source": source_name,
            "matched": len(matched),
            "chunks": matched,
        }

    def clear_memory(self, session_id: str) -> dict:
        with self._memory_lock:
            self._session_memory.pop(session_id, None)
        return {"status": f"Session memory cleared for '{session_id}'."}

    def get_memory(self, session_id: str) -> dict:
        return {"session_id": session_id, "memory": self._get_session_memory(session_id)}


runtime_service = RuntimeService()
