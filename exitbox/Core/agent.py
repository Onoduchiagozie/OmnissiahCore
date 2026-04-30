"""
OmnissiahCore - Core/agent.py

Coordinates retrieval, prompt construction, Ollama calls, and session memory.

Modes:
  remembrancer  - grounded Q&A, witness voice
  narrator      - full scene reconstruction, perspective shifts signalled
  explorer      - object/artefact analysis
"""

import json
from typing import Generator, Optional

import requests

from Core.app_text import app_text
from Core.config_loader import ollama_cfg
from Core.prompt import (
    build_narrate_prompt,
    build_object_explorer_prompt,
    build_prompt,
    format_debug,
)
from Core.retriever import OmnissiahRetriever


class OmnissiahAgent:
    def __init__(
        self,
        verbose: bool = False,
        mode: str = "remembrancer",
        retriever: Optional[OmnissiahRetriever] = None,
    ):
        self.verbose = verbose
        self.mode = mode
        self.retriever = retriever or OmnissiahRetriever()
        self._memory: list[dict] = []
        self._MAX_MEM = 4

    def ask(
        self,
        query: str,
        book_filter: str = None,
        source_filter: list[str] = None,
        top_k: int = None,
        candidate_pool: int = None,
        stitching_window: int = None,
        stream: bool = None,
    ) -> tuple[str, list[dict]]:
        intent = self._classify_intent(query)
        if self.verbose:
            print(f"   [Agent] Intent: {intent}  Mode: {self.mode}")

        chunks = self.retriever.search(
            query=query,
            top_k=top_k,
            candidate_pool=candidate_pool,
            stitching_window=stitching_window,
            book_filter=book_filter,
            source_filter=source_filter,
        )

        if self.verbose:
            print(f"   [Agent] {len(chunks)} chunks after retrieval + stitching")

        if not chunks:
            return app_text["agent"]["no_chunks_message"], []

        system_prompt, user_msg = self._build_prompt(query, chunks)

        if self._memory:
            user_msg = self._format_memory() + "\n\n" + user_msg

        use_stream = stream if stream is not None else ollama_cfg.get("stream", True)
        if use_stream:
            response = self._call_ollama_stream(system_prompt, user_msg)
        else:
            response = self._call_ollama_sync(system_prompt, user_msg)

        self._update_memory(query, response)

        if self.verbose:
            print(format_debug(query, chunks, response))

        return response, chunks

    def ask_stream(
        self,
        query: str,
        book_filter: str = None,
        source_filter: list[str] = None,
        top_k: int = None,
        candidate_pool: int = None,
        stitching_window: int = None,
    ) -> Generator[str, None, None]:
        _ = self._classify_intent(query)

        chunks = self.retriever.search(
            query=query,
            top_k=top_k,
            candidate_pool=candidate_pool,
            stitching_window=stitching_window,
            book_filter=book_filter,
            source_filter=source_filter,
        )

        if not chunks:
            yield app_text["agent"]["stream_no_chunks_message"]
            return

        system_prompt, user_msg = self._build_prompt(query, chunks)

        if self._memory:
            user_msg = self._format_memory() + "\n\n" + user_msg

        full_response = ""
        for token in self._stream_ollama(system_prompt, user_msg):
            full_response += token
            yield token

        sources_data = [
            {
                "source": c.get("source", "?"),
                "chapter": c.get("chapter", "?"),
                "stitch_range": c.get("stitch_range", ""),
                "score": round(
                    c.get("rerank_score")
                    or c.get("query_overlap_score")
                    or c.get("rrf_score")
                    or c.get("faiss_score")
                    or 0.0,
                    4,
                ),
            }
            for c in chunks
        ]
        self._update_memory(query, full_response)
        yield f"__SOURCES__:{json.dumps(sources_data)}"

    def _build_prompt(self, query: str, chunks: list[dict]) -> tuple[str, str]:
        """Dispatch to the right prompt builder based on current mode."""
        if self.mode == "narrator":
            return build_narrate_prompt(query, chunks)
        elif self.mode == "explorer":
            return build_object_explorer_prompt(query, chunks)
        else:
            return build_prompt(query, chunks)

    def _classify_intent(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["who is", "who was", "what is", "what was", "tell me about", "describe"]):
            return "lore_lookup"
        if any(w in q for w in ["compare", "difference between", "vs", "versus", "how does", "how did"]):
            return "comparison"
        if any(w in q for w in ["summarize", "summary", "overview", "arc", "timeline", "entire story"]):
            return "summarization"
        if any(w in q for w in ["fight", "duel", "battle", "siege", "war", "assault", "narrate"]):
            return "battle_reconstruction"
        return "narrative_generation"

    def _call_ollama_sync(self, system_prompt: str, user_msg: str) -> str:
        payload = {
            "model": ollama_cfg["model"],
            "stream": False,
            "options": {
                "num_ctx": ollama_cfg["num_ctx"],
                "temperature": ollama_cfg["temperature"],
                "top_p": ollama_cfg["top_p"],
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }
        try:
            resp = requests.post(
                ollama_cfg["url"],
                json=payload,
                timeout=ollama_cfg.get("timeout", 300),
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            return "[ERROR] Ollama is not running. Start with: ollama serve"
        except requests.exceptions.Timeout:
            return "[ERROR] Ollama timed out. Try a smaller model or reduce num_ctx."
        except Exception as e:
            return f"[ERROR] Ollama call failed: {e}"

    def _call_ollama_stream(self, system_prompt: str, user_msg: str) -> str:
        return "".join(self._stream_ollama(system_prompt, user_msg))

    def _stream_ollama(self, system_prompt: str, user_msg: str) -> Generator[str, None, None]:
        payload = {
            "model": ollama_cfg["model"],
            "stream": True,
            "options": {
                "num_ctx": ollama_cfg["num_ctx"],
                "temperature": ollama_cfg["temperature"],
                "top_p": ollama_cfg["top_p"],
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }
        try:
            with requests.post(
                ollama_cfg["url"],
                json=payload,
                stream=True,
                timeout=ollama_cfg.get("timeout", 300),
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
        except requests.exceptions.ConnectionError:
            yield "[ERROR] Ollama is not running. Start with: ollama serve"
        except requests.exceptions.Timeout:
            yield "[ERROR] Ollama timed out."
        except Exception as e:
            yield f"[ERROR] {e}"

    def _update_memory(self, query: str, response: str):
        self._memory.append({"query": query, "response": response})
        if len(self._memory) > self._MAX_MEM:
            self._memory.pop(0)

    def _format_memory(self) -> str:
        lines = [app_text["prompts"]["memory_intro"]]
        for turn in self._memory:
            lines.append(f"Petitioner: {turn['query']}")
            lines.append(f"Remembrancer: {turn['response'][:400]}...")
        return "\n".join(lines)

    def clear_memory(self):
        self._memory = []
        if self.verbose:
            print("   [Agent] Session memory cleared.")

    @property
    def memory(self) -> list[dict]:
        return list(self._memory)
# """
# OmnissiahCore - Core/agent.py
#
# Coordinates retrieval, prompt construction, Ollama calls, and session memory.
# """
#
# import json
# from typing import Generator, Optional
#
# import requests
#
# from Core.app_text import app_text
# from Core.config_loader import ollama_cfg
# from Core.prompt import build_object_explorer_prompt, build_prompt, format_debug
# from Core.retriever import OmnissiahRetriever
#
#
# class OmnissiahAgent:
#     def __init__(
#         self,
#         verbose: bool = False,
#         mode: str = "remembrancer",
#         retriever: Optional[OmnissiahRetriever] = None,
#     ):
#         self.verbose = verbose
#         self.mode = mode
#         self.retriever = retriever or OmnissiahRetriever()
#         self._memory: list[dict] = []
#         self._MAX_MEM = 4
#
#     def ask(
#         self,
#         query: str,
#         book_filter: str = None,
#         source_filter: list[str] = None,
#         top_k: int = None,
#         candidate_pool: int = None,
#         stitching_window: int = None,
#         stream: bool = None,
#     ) -> tuple[str, list[dict]]:
#         intent = self._classify_intent(query)
#         if self.verbose:
#             print(f"   [Agent] Intent: {intent}")
#
#         chunks = self.retriever.search(
#             query=query,
#             top_k=top_k,
#             candidate_pool=candidate_pool,
#             stitching_window=stitching_window,
#             book_filter=book_filter,
#             source_filter=source_filter,
#         )
#
#         if self.verbose:
#             print(f"   [Agent] {len(chunks)} chunks after retrieval + stitching")
#
#         if not chunks:
#             return app_text["agent"]["no_chunks_message"], []
#
#         if self.mode == "explorer":
#             system_prompt, user_msg = build_object_explorer_prompt(query, chunks)
#         else:
#             system_prompt, user_msg = build_prompt(query, chunks)
#
#         if self._memory:
#             user_msg = self._format_memory() + "\n\n" + user_msg
#
#         use_stream = stream if stream is not None else ollama_cfg.get("stream", True)
#         if use_stream:
#             response = self._call_ollama_stream(system_prompt, user_msg)
#         else:
#             response = self._call_ollama_sync(system_prompt, user_msg)
#
#         self._update_memory(query, response)
#
#         if self.verbose:
#             print(format_debug(query, chunks, response))
#
#         return response, chunks
#
#     def ask_stream(
#         self,
#         query: str,
#         book_filter: str = None,
#         source_filter: list[str] = None,
#         top_k: int = None,
#         candidate_pool: int = None,
#         stitching_window: int = None,
#     ) -> Generator[str, None, None]:
#         _ = self._classify_intent(query)
#
#         chunks = self.retriever.search(
#             query=query,
#             top_k=top_k,
#             candidate_pool=candidate_pool,
#             stitching_window=stitching_window,
#             book_filter=book_filter,
#             source_filter=source_filter,
#         )
#
#         if not chunks:
#             yield app_text["agent"]["stream_no_chunks_message"]
#             return
#
#         if self.mode == "explorer":
#             system_prompt, user_msg = build_object_explorer_prompt(query, chunks)
#         else:
#             system_prompt, user_msg = build_prompt(query, chunks)
#
#         if self._memory:
#             user_msg = self._format_memory() + "\n\n" + user_msg
#
#         full_response = ""
#         for token in self._stream_ollama(system_prompt, user_msg):
#             full_response += token
#             yield token
#
#         sources_data = [
#             {
#                 "source": c.get("source", "?"),
#                 "chapter": c.get("chapter", "?"),
#                 "stitch_range": c.get("stitch_range", ""),
#                 "score": round(
#                     c.get("rerank_score")
#                     or c.get("query_overlap_score")
#                     or c.get("rrf_score")
#                     or c.get("faiss_score")
#                     or 0.0,
#                     4,
#                 ),
#             }
#             for c in chunks
#         ]
#         self._update_memory(query, full_response)
#         yield f"__SOURCES__:{json.dumps(sources_data)}"
#
#     def _classify_intent(self, query: str) -> str:
#         q = query.lower()
#         if any(w in q for w in ["who is", "who was", "what is", "what was", "tell me about", "describe"]):
#             return "lore_lookup"
#         if any(w in q for w in ["compare", "difference between", "vs", "versus", "how does", "how did"]):
#             return "comparison"
#         if any(w in q for w in ["summarize", "summary", "overview", "arc", "timeline", "entire story"]):
#             return "summarization"
#         if any(w in q for w in ["fight", "duel", "battle", "siege", "war"]):
#             return "battle_reconstruction"
#         return "narrative_generation"
#
#     def _call_ollama_sync(self, system_prompt: str, user_msg: str) -> str:
#         payload = {
#             "model": ollama_cfg["model"],
#             "stream": False,
#             "options": {
#                 "num_ctx": ollama_cfg["num_ctx"],
#                 "temperature": ollama_cfg["temperature"],
#                 "top_p": ollama_cfg["top_p"],
#             },
#             "messages": [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_msg},
#             ],
#         }
#         try:
#             resp = requests.post(
#                 ollama_cfg["url"],
#                 json=payload,
#                 timeout=ollama_cfg.get("timeout", 180),
#             )
#             resp.raise_for_status()
#             return resp.json()["message"]["content"].strip()
#         except requests.exceptions.ConnectionError:
#             return "[ERROR] Ollama is not running. Start with: ollama serve"
#         except requests.exceptions.Timeout:
#             return "[ERROR] Ollama timed out. Try a smaller model or reduce num_ctx."
#         except Exception as e:
#             return f"[ERROR] Ollama call failed: {e}"
#
#     def _call_ollama_stream(self, system_prompt: str, user_msg: str) -> str:
#         return "".join(self._stream_ollama(system_prompt, user_msg))
#
#     def _stream_ollama(self, system_prompt: str, user_msg: str) -> Generator[str, None, None]:
#         payload = {
#             "model": ollama_cfg["model"],
#             "stream": True,
#             "options": {
#                 "num_ctx": ollama_cfg["num_ctx"],
#                 "temperature": ollama_cfg["temperature"],
#                 "top_p": ollama_cfg["top_p"],
#             },
#             "messages": [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_msg},
#             ],
#         }
#         try:
#             with requests.post(
#                 ollama_cfg["url"],
#                 json=payload,
#                 stream=True,
#                 timeout=ollama_cfg.get("timeout", 180),
#             ) as resp:
#                 resp.raise_for_status()
#                 for raw_line in resp.iter_lines():
#                     if not raw_line:
#                         continue
#                     try:
#                         data = json.loads(raw_line)
#                     except json.JSONDecodeError:
#                         continue
#                     token = data.get("message", {}).get("content", "")
#                     if token:
#                         yield token
#                     if data.get("done", False):
#                         break
#         except requests.exceptions.ConnectionError:
#             yield "[ERROR] Ollama is not running. Start with: ollama serve"
#         except requests.exceptions.Timeout:
#             yield "[ERROR] Ollama timed out."
#         except Exception as e:
#             yield f"[ERROR] {e}"
#
#     def _update_memory(self, query: str, response: str):
#         self._memory.append({"query": query, "response": response})
#         if len(self._memory) > self._MAX_MEM:
#             self._memory.pop(0)
#
#     def _format_memory(self) -> str:
#         lines = [app_text["prompts"]["memory_intro"]]
#         for turn in self._memory:
#             lines.append(f"Petitioner: {turn['query']}")
#             lines.append(f"Remembrancer: {turn['response'][:400]}...")
#         return "\n".join(lines)
#
#     def clear_memory(self):
#         self._memory = []
#         if self.verbose:
#             print("   [Agent] Session memory cleared.")
#
#     @property
#     def memory(self) -> list[dict]:
#         return list(self._memory)
