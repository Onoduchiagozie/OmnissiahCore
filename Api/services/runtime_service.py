import os
from threading import Lock
from typing import Optional

import faiss
import json as _json

from Api.models import QueryRequest
from Core.agent import OmnissiahAgent
from Core.config_loader import active_profile, machine_role, ollama_cfg, paths, retrieval_cfg
from Core.prompt import build_prompt
from Core.retriever import OmnissiahRetriever


class RuntimeService:
    def __init__(self):
        self._retriever: Optional[OmnissiahRetriever] = None
        self._session_memory: dict[str, list[dict]] = {}
        self._memory_lock = Lock()   # only protects session_memory dict, not Ollama
        self._metadata_cache: list[dict] = []

    @property
    def retriever(self) -> Optional[OmnissiahRetriever]:
        return self._retriever

    def startup(self):
        print("[Startup] Loading retriever...")
        self._retriever = OmnissiahRetriever()
        self._metadata_cache = self._load_metadata()
        print("[Startup] Runtime ready")

    def _load_metadata(self) -> list[dict]:
        if not os.path.exists(paths["metadata"]):
            return []
        with open(paths["metadata"], "r", encoding="utf-8", errors="replace") as f:
            return _json.load(f)

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
            raise RuntimeError("Retriever not initialised.")

    def run_query(self, req: QueryRequest, mode: str, stream: bool) -> tuple[str, list[dict]]:
        """
        Lock is NOT held during Ollama inference.
        Memory is read before and written after - the only critical sections.
        """
        session_id = req.session_id or "default"

        # Build agent with snapshotted memory (no lock held during inference)
        agent = self._build_agent(session_id, mode)

        response, chunks = agent.ask(
            query=req.query,
            book_filter=req.book_filter,
            source_filter=req.source_filter,
            top_k=req.top_k,
            candidate_pool=req.candidate_pool,
            stitching_window=req.stitching_window,
            stream=stream,
        )

        # Write back memory after inference completes
        self._set_session_memory(session_id, agent.memory)
        return response, chunks

    def stream_query_mode(self, req: QueryRequest, mode: str = "remembrancer"):
        """stream_query with selectable mode (remembrancer / narrator / explorer)."""
        session_id = req.session_id or "default"
        agent = self._build_agent(session_id, mode=mode)
        full_response = ""

        for token in agent.ask_stream(
            query=req.query,
            book_filter=req.book_filter,
            source_filter=req.source_filter,
            top_k=req.top_k,
            candidate_pool=req.candidate_pool,
            stitching_window=req.stitching_window,
        ):
            if token.startswith("__SOURCES__:"):
                yield f"data: {token}\n\n"
            else:
                full_response += token
                yield f"data: {token}\n\n"

        self._set_session_memory(session_id, agent.memory)
        yield "data: [DONE]\n\n"

    def stream_query(self, req: QueryRequest):
        session_id = req.session_id or "default"

        agent = self._build_agent(session_id, mode="remembrancer")
        full_response = ""

        for token in agent.ask_stream(
            query=req.query,
            book_filter=req.book_filter,
            source_filter=req.source_filter,
            top_k=req.top_k,
            candidate_pool=req.candidate_pool,
            stitching_window=req.stitching_window,
        ):
            if token.startswith("__SOURCES__:"):
                yield f"data: {token}\n\n"
            else:
                full_response += token
                yield f"data: {token}\n\n"

        self._set_session_memory(session_id, agent.memory)
        yield "data: [DONE]\n\n"

    def inspect_query(self, req: QueryRequest) -> dict:
        inspection = self._retriever.inspect(
            query=req.query,
            book_filter=req.book_filter,
            source_filter=req.source_filter,
            top_k=req.top_k,
            candidate_pool=req.candidate_pool,
            stitching_window=req.stitching_window,
        )
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
            "status": "online",
            "active_profile": active_profile,
            "machine_role": machine_role,
            "ollama_model": ollama_cfg["model"],
            "ollama_url": ollama_cfg["url"],
            "metadata_loaded": len(self._metadata_cache),
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
            "ollama": {
                "url": ollama_cfg["url"],
                "model": ollama_cfg["model"],
                "num_ctx": ollama_cfg["num_ctx"],
                "timeout": ollama_cfg["timeout"],
                "temperature": ollama_cfg["temperature"],
                "top_p": ollama_cfg["top_p"],
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
            m for m in self._metadata_cache if source_name.lower() in m.get("source", "").lower()
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