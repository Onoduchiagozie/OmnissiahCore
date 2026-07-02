import os
import re
from threading import Lock
from typing import Optional

import faiss
import json as _json
import numpy as np

from Api.models import QueryRequest
from Core.agent import OmnissiahAgent
from Core.config_loader import active_profile, machine_role, llm_cfg, paths, retrieval_cfg
from Core.prompt import build_prompt
from Core.retriever import OmnissiahRetriever


def _sanitize_numpy(obj):
    """Recursively convert numpy scalar/array types to native Python types for JSON serialisation."""
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
    """
    Aggressive LLM response text cleaning.
    Applied to full sync responses and accumulated stream responses.
    """
    if not text:
        return text

    # Remove all control characters except newline and tab
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')

    # Remove slash artifacts: " / ", "/", "\ ", "\/"
    text = re.sub(r'\s*/\s*', ' ', text)
    text = re.sub(r'(?<!\w)/(?!\w)', ' ', text)
    text = re.sub(r'\\\s*', ' ', text)
    text = re.sub(r'\\/', ' ', text)

    # Collapse 3+ newlines to double (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse single newlines within paragraph to space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Strip leading/trailing
    text = text.strip()

    return text


class RuntimeService:
    def __init__(self):
        self._retriever: Optional[OmnissiahRetriever] = None
        self._session_memory: dict[str, list[dict]] = {}
        self._memory_lock = Lock()
        self._metadata_cache: list[dict] = []

    @property
    def retriever(self) -> Optional[OmnissiahRetriever]:
        return self._retriever

    def startup(self):
        print("[Startup] Loading retriever...")
        self._retriever = OmnissiahRetriever()
        # Reuse the retriever's already-loaded metadata instead of re-reading
        # metadata.json from disk a second time. This used to load the same
        # file twice into memory (once here, once inside OmnissiahRetriever).
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
            raise RuntimeError("Retriever not initialised.")

    def run_query(self, req: QueryRequest, mode: str, stream: bool) -> tuple[str, list[dict]]:
        """
        Lock is NOT held during LLM inference.
        Memory is read before and written after — the only critical sections.
        clean_response is applied before returning to the route.
        """
        session_id = req.session_id or "default"

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

        self._set_session_memory(session_id, agent.memory)

        # Clean response text before returning
        response = clean_response(response)

        return response, chunks

    def stream_query_mode(self, req: QueryRequest, mode: str = "remembrancer"):
        """
        Stream with selectable mode: remembrancer / narrator / explorer.

        NOTE: This method does NOT yield [DONE]. The route that calls this
        method is responsible for yielding exactly one [DONE] frame.
        This prevents the duplicate [DONE] bug.

        Emits __STAGE__: frames at the REAL pipeline boundaries (embedding,
        retrieving, reranking, generating, complete) via an on_stage callback
        threaded through agent.ask_stream -> retriever.search. Stages reflect
        actual timing, not a flashed-through fake sequence.
        """
        session_id = req.session_id or "default"
        agent = self._build_agent(session_id, mode=mode)
        full_response = ""

        # agent.ask_stream() is a generator, but on_stage() fires synchronously
        # from inside retriever.search() and agent.ask_stream() BEFORE either
        # of those functions yields anything. A plain callback can't "yield"
        # across that boundary on its own, so we stash fired stage names into
        # a small queue-like list and drain it before/around each real token.
        pending_stages: list[str] = []

        def _on_stage(stage_name: str):
            pending_stages.append(stage_name)

        stream = agent.ask_stream(
            query=req.query,
            book_filter=req.book_filter,
            source_filter=req.source_filter,
            top_k=req.top_k,
            candidate_pool=req.candidate_pool,
            stitching_window=req.stitching_window,
            on_stage=_on_stage,
        )

        for token in stream:
            # Drain any stage events that fired since the last token (this is
            # how "embedding"/"retrieving"/"reranking" surface even though
            # they happen before the generator's first yield — the callback
            # appends to pending_stages synchronously during that blocking
            # setup, and the very first time control returns to us here, all
            # three are already queued up in order).
            while pending_stages:
                stage_name = pending_stages.pop(0)
                yield f"data: __STAGE__:{stage_name}\n\n"

            if token.startswith("__SOURCES__:"):
                yield f"data: {token}\n\n"
                continue

            full_response += token
            clean_token = token.replace(" / ", " ").replace("/", " ")
            yield f"data: {clean_token}\n\n"

        # Drain anything left over (shouldn't normally happen, but safe)
        while pending_stages:
            stage_name = pending_stages.pop(0)
            yield f"data: __STAGE__:{stage_name}\n\n"

        yield "data: __STAGE__:complete\n\n"

        # Write back memory after stream completes
        self._set_session_memory(session_id, agent.memory)

        # DO NOT yield [DONE] here. The route handles it.
    # def stream_query_mode(self, req: QueryRequest, mode: str = "remembrancer"):
    #     """
    #     Stream with selectable mode: remembrancer / narrator / explorer.
    #
    #     NOTE: This method does NOT yield [DONE]. The route that calls this
    #     method is responsible for yielding exactly one [DONE] frame.
    #     This prevents the duplicate [DONE] bug.
    #     """
    #     session_id = req.session_id or "default"
    #     agent = self._build_agent(session_id, mode=mode)
    #     full_response = ""
    #
    #     for token in agent.ask_stream(
    #         query=req.query,
    #         book_filter=req.book_filter,
    #         source_filter=req.source_filter,
    #         top_k=req.top_k,
    #         candidate_pool=req.candidate_pool,
    #         stitching_window=req.stitching_window,
    #     ):
    #         if token.startswith("__SOURCES__:"):
    #             # Sources frame — forward as-is, no cleaning needed
    #             yield f"data: {token}\n\n"
    #         else:
    #             full_response += token
    #             # Clean slash artifacts at token level before sending
    #             clean_token = token.replace(" / ", " ").replace("/", " ")
    #             yield f"data: {clean_token}\n\n"
    #
    #     # Write back memory after stream completes
    #     self._set_session_memory(session_id, agent.memory)
    #
    #     # DO NOT yield [DONE] here. The route handles it.

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
            "status": "online",
            "active_profile": active_profile,
            "machine_role": machine_role,
            "llm_model": llm_cfg["model"],
            "llm_url": llm_cfg["url"],
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

# import os
# import re
# from threading import Lock
# from typing import Optional
#
# import faiss
# import json as _json
# import numpy as np
#
# from Api.models import QueryRequest
# from Core.agent import OmnissiahAgent
# from Core.config_loader import active_profile, machine_role, llm_cfg, paths, retrieval_cfg
# from Core.prompt import build_prompt
# from Core.retriever import OmnissiahRetriever
#
#
# def _sanitize_numpy(obj):
#     """Recursively convert numpy scalar/array types to native Python types for JSON serialisation."""
#     if isinstance(obj, dict):
#         return {k: _sanitize_numpy(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [_sanitize_numpy(i) for i in obj]
#     if isinstance(obj, set):
#         return [_sanitize_numpy(i) for i in obj]
#     if isinstance(obj, np.integer):
#         return int(obj)
#     if isinstance(obj, np.floating):
#         return float(obj)
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     return obj
#
# def clean_response(text: str) -> str:
#     """
#     Aggressive LLM response text cleaning.
#     Applied to full sync responses and accumulated stream responses.
#     """
#     if not text:
#         return text
#
#     # Remove all control characters except newline and tab
#     text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')
#
#     # Remove slash artifacts: " / ", "/", "\ ", "\/"
#     text = re.sub(r'\s*/\s*', ' ', text)
#     text = re.sub(r'(?<!\w)/(?!\w)', ' ', text)
#     text = re.sub(r'\\\s*', ' ', text)
#     text = re.sub(r'\\/', ' ', text)
#
#     # Collapse 3+ newlines to double (preserve paragraph breaks)
#     text = re.sub(r'\n{3,}', '\n\n', text)
#
#     # Collapse single newlines within paragraph to space
#     text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
#
#     # Collapse multiple spaces
#     text = re.sub(r' {2,}', ' ', text)
#
#     # Strip leading/trailing
#     text = text.strip()
#
#     return text
#
#
# class RuntimeService:
#     def __init__(self):
#         self._retriever: Optional[OmnissiahRetriever] = None
#         self._session_memory: dict[str, list[dict]] = {}
#         self._memory_lock = Lock()
#         self._metadata_cache: list[dict] = []
#
#     @property
#     def retriever(self) -> Optional[OmnissiahRetriever]:
#         return self._retriever
#
#     def startup(self):
#         print("[Startup] Loading retriever...")
#         self._retriever = OmnissiahRetriever()
#         self._metadata_cache = self._load_metadata()
#         print("[Startup] Runtime ready")
#
#     def _load_metadata(self) -> list[dict]:
#         if not os.path.exists(paths["metadata"]):
#             return []
#         with open(paths["metadata"], "r", encoding="utf-8", errors="replace") as f:
#             return _json.load(f)
#
#     def _get_session_memory(self, session_id: str) -> list[dict]:
#         with self._memory_lock:
#             return list(self._session_memory.get(session_id, []))
#
#     def _set_session_memory(self, session_id: str, memory: list[dict]):
#         with self._memory_lock:
#             self._session_memory[session_id] = memory
#
#     def _build_agent(self, session_id: str, mode: str) -> OmnissiahAgent:
#         agent = OmnissiahAgent(verbose=False, mode=mode, retriever=self._retriever)
#         agent._memory = self._get_session_memory(session_id)
#         return agent
#
#     def ensure_ready(self):
#         if not self._retriever:
#             raise RuntimeError("Retriever not initialised.")
#
#     def run_query(self, req: QueryRequest, mode: str, stream: bool) -> tuple[str, list[dict]]:
#         """
#         Lock is NOT held during LLM inference.
#         Memory is read before and written after — the only critical sections.
#         clean_response is applied before returning to the route.
#         """
#         session_id = req.session_id or "default"
#
#         agent = self._build_agent(session_id, mode)
#
#         response, chunks = agent.ask(
#             query=req.query,
#             book_filter=req.book_filter,
#             source_filter=req.source_filter,
#             top_k=req.top_k,
#             candidate_pool=req.candidate_pool,
#             stitching_window=req.stitching_window,
#             stream=stream,
#         )
#
#         self._set_session_memory(session_id, agent.memory)
#
#         # Clean response text before returning
#         response = clean_response(response)
#
#         return response, chunks
#
#     def stream_query_mode(self, req: QueryRequest, mode: str = "remembrancer"):
#         """
#         Stream with selectable mode: remembrancer / narrator / explorer.
#
#         NOTE: This method does NOT yield [DONE]. The route that calls this
#         method is responsible for yielding exactly one [DONE] frame.
#         This prevents the duplicate [DONE] bug.
#         """
#         session_id = req.session_id or "default"
#         agent = self._build_agent(session_id, mode=mode)
#         full_response = ""
#
#         for token in agent.ask_stream(
#             query=req.query,
#             book_filter=req.book_filter,
#             source_filter=req.source_filter,
#             top_k=req.top_k,
#             candidate_pool=req.candidate_pool,
#             stitching_window=req.stitching_window,
#         ):
#             if token.startswith("__SOURCES__:"):
#                 # Sources frame — forward as-is, no cleaning needed
#                 yield f"data: {token}\n\n"
#             else:
#                 full_response += token
#                 # Clean slash artifacts at token level before sending
#                 clean_token = token.replace(" / ", " ").replace("/", " ")
#                 yield f"data: {clean_token}\n\n"
#
#         # Write back memory after stream completes
#         self._set_session_memory(session_id, agent.memory)
#
#         # DO NOT yield [DONE] here. The route handles it.
#
#     def inspect_query(self, req: QueryRequest) -> dict:
#         inspection = self._retriever.inspect(
#             query=req.query,
#             book_filter=req.book_filter,
#             source_filter=req.source_filter,
#             top_k=req.top_k,
#             candidate_pool=req.candidate_pool,
#             stitching_window=req.stitching_window,
#         )
#         inspection = _sanitize_numpy(inspection)
#         system_prompt, user_message = build_prompt(req.query, inspection["stitched_hits"])
#         return {
#             "inspection": inspection,
#             "prompt_preview": {
#                 "system_prompt": system_prompt[:4000],
#                 "user_message": user_message,
#             },
#         }
#
#     def health_payload(self) -> dict:
#         return {
#             "status": "online",
#             "active_profile": active_profile,
#             "machine_role": machine_role,
#             "llm_model": llm_cfg["model"],
#             "llm_url": llm_cfg["url"],
#             "metadata_loaded": len(self._metadata_cache),
#         }
#
#     def info_payload(self) -> dict:
#         try:
#             index = faiss.read_index(paths["faiss"])
#             n_vectors = index.ntotal
#             dim = index.d
#         except Exception:
#             n_vectors = -1
#             dim = -1
#
#         manifest = {}
#         if os.path.exists(paths["manifest"]):
#             with open(paths["manifest"], "r", encoding="utf-8") as f:
#                 manifest = _json.load(f)
#
#         return {
#             "index_vectors": n_vectors,
#             "index_dim": dim,
#             "machine_role": machine_role,
#             "manifest": manifest,
#             "retrieval": retrieval_cfg,
#             "cached_sources": len(set(m.get("source", "unknown") for m in self._metadata_cache)),
#         }
#
#     def runtime_config_payload(self) -> dict:
#         return {
#             "active_profile": active_profile,
#             "machine_role": machine_role,
#             "llm": {
#                 "url": llm_cfg["url"],
#                 "model": llm_cfg["model"],
#                 "timeout": llm_cfg["timeout"],
#                 "temperature": llm_cfg["temperature"],
#                 "top_p": llm_cfg["top_p"],
#                 "max_tokens": llm_cfg.get("max_tokens"),
#             },
#             "retrieval": retrieval_cfg,
#             "paths": {
#                 "db_dir": paths["db_dir"],
#                 "faiss": paths["faiss"],
#                 "metadata": paths["metadata"],
#             },
#         }
#
#     def list_sources_payload(self) -> dict:
#         sources = sorted(set(m.get("source", "unknown") for m in self._metadata_cache))
#         return {"total": len(sources), "sources": sources}
#
#     def source_chunks_payload(self, source_name: str, limit: int) -> dict:
#         matched = [
#             m for m in self._metadata_cache
#             if source_name.lower() in m.get("source", "").lower()
#         ][:limit]
#         return {
#             "source": source_name,
#             "matched": len(matched),
#             "chunks": matched,
#         }
#
#     def clear_memory(self, session_id: str) -> dict:
#         with self._memory_lock:
#             self._session_memory.pop(session_id, None)
#         return {"status": f"Session memory cleared for '{session_id}'."}
#
#     def get_memory(self, session_id: str) -> dict:
#         return {"session_id": session_id, "memory": self._get_session_memory(session_id)}
#
#
# runtime_service = RuntimeService()