"""
OmnissiahCoreOld - Core/agent.py

Coordinates retrieval, prompt construction, LLM calls, and session memory.

LLM backend: LM Studio, via its OpenAI-compatible /v1/chat/completions endpoint.
"""

import json
from typing import Generator, Optional
import requests

from Core.app_text import app_text
from Core.config_loader import llm_cfg
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

        use_stream = stream if stream is not None else llm_cfg.get("stream", True)
        if use_stream:
            response = self._call_llm_stream(system_prompt, user_msg)
        else:
            response = self._call_llm_sync(system_prompt, user_msg)

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
        for token in self._stream_llm(system_prompt, user_msg):
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

    def _build_payload(self, system_prompt: str, user_msg: str, stream: bool) -> dict:
        """
        Builds an OpenAI-compatible chat completions payload for LM Studio.

        LM Studio's /v1/chat/completions endpoint takes sampling params as
        top-level fields (no "options" wrapper, no "num_ctx" — context length
        is a server/model-load setting in LM Studio, not a per-request field).
        """
        payload = {
            "model": llm_cfg["model"],
            "stream": stream,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": llm_cfg["temperature"],
            "top_p": llm_cfg["top_p"],
        }
        max_tokens = llm_cfg.get("max_tokens")
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def _call_llm_sync(self, system_prompt: str, user_msg: str) -> str:
        payload = self._build_payload(system_prompt, user_msg, stream=False)
        try:
            resp = requests.post(
                llm_cfg["url"],
                json=payload,
                timeout=llm_cfg.get("timeout", 300),
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            return "[ERROR] LM Studio is not running or the server is not started. Open LM Studio, go to the Server tab, and press Start Server."
        except requests.exceptions.Timeout:
            return "[ERROR] LM Studio timed out. Try a smaller model or reduce max_tokens."
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if status == 404:
                return "[ERROR] LM Studio returned 404. No model is currently loaded — open LM Studio and load a model first."
            return f"[ERROR] LM Studio HTTP {status}: {e}"
        except (KeyError, IndexError, ValueError) as e:
            return f"[ERROR] Unexpected LM Studio response shape: {e}"
        except Exception as e:
            return f"[ERROR] LM Studio call failed: {e}"

    def _call_llm_stream(self, system_prompt: str, user_msg: str) -> str:
        return "".join(self._stream_llm(system_prompt, user_msg))

    def _stream_llm(self, system_prompt: str, user_msg: str) -> Generator[str, None, None]:
        """
        Streams from LM Studio's OpenAI-compatible endpoint.

        LM Studio sends Server-Sent-Events lines prefixed with "data: ".
        Each line (except the terminator) is JSON shaped like:
            {"choices": [{"delta": {"content": "..."}}]}
        The stream ends with a literal line: "data: [DONE]"
        (NOT a JSON field like Ollama's `"done": true`.)
        """
        payload = self._build_payload(system_prompt, user_msg, stream=True)
        try:
            with requests.post(
                llm_cfg["url"],
                json=payload,
                stream=True,
                timeout=llm_cfg.get("timeout", 300),
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data:"):
                        continue

                    chunk_str = raw_line[len("data:"):].strip()
                    if chunk_str == "[DONE]":
                        break

                    try:
                        data = json.loads(chunk_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    token = choices[0].get("delta", {}).get("content", "")
                    if token:
                        yield token
        except requests.exceptions.ConnectionError:
            yield "[ERROR] LM Studio is not running or the server is not started. Open LM Studio, go to the Server tab, and press Start Server."
        except requests.exceptions.Timeout:
            yield "[ERROR] LM Studio timed out."
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if status == 404:
                yield "[ERROR] LM Studio returned 404. No model is currently loaded — open LM Studio and load a model first."
            else:
                yield f"[ERROR] LM Studio HTTP {status}: {e}"
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