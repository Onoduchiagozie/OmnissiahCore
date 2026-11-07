"""
OmnissiahCore - Core/agent.py

Coordinates retrieval, prompt construction, LLM calls, and session memory.

LLM backend: LM Studio, via its OpenAI-compatible /v1/chat/completions endpoint.
"""

import json
from typing import Generator, Optional
import requests

from Core.app_text import app_text
from Core.config_loader import llm_cfg
from Core.gemini_provider import generate, ProviderError
from Core.prompt import (
    build_focus_prompt,
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
        max_sources: int = None,
        stream: bool = None,
    ) -> tuple[str, list[dict]]:
        intent = self._classify_intent(query)
        if self.verbose:
            print(f"   [Agent] Intent: {intent}  Mode: {self.mode}")

        # CHANGED: book_filter/source_filter now forwarded into search_focus()
        # (previously silently dropped here even though this method accepted
        # them — focus mode never actually saw them).
        if self.mode == "focus":
            chunks = self.retriever.search_focus(
                query=query,
                max_sources=max_sources,
                book_filter=book_filter,
                source_filter=source_filter,
            )
        else:
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

        if self.mode == "focus" and self._is_pure_scene_response(chunks):
            # Pre-woven chronicles are already finished, curated prose.
            # Handing them to the LLM a second time to "narrate" them again
            # only adds a second chance to invent details that aren't in the
            # source — pure risk, no benefit. Return them directly instead.
            # Trade-off: two chronicles read as two connected accounts rather
            # than one seamlessly blended narrative across both.
            response = self._concat_scene_chronicles(chunks)
            self._update_memory(query, response)
            if self.verbose:
                print(format_debug(query, chunks, response))
            return response, chunks

        system_prompt, user_msg = self._build_prompt(query, chunks)

        if self._memory:
            user_msg = self._format_memory() + "\n\n" + user_msg

        use_stream = stream if stream is not None else llm_cfg.get("stream", True)
        if use_stream:
            response = self._call_llm_stream(system_prompt, user_msg)
        else:
            response = self._call_llm_sync(system_prompt, user_msg)

        if "[ERROR]" not in response:
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
            max_sources: int = None,
            on_stage=None,  # optional callable(str) -> None
    ) -> Generator[str, None, None]:
        _ = self._classify_intent(query)

        # CHANGED: same fix as ask() above — book_filter/source_filter now
        # forwarded into search_focus() instead of being dropped.
        if self.mode == "focus":
            chunks = self.retriever.search_focus(
                query=query,
                max_sources=max_sources,
                book_filter=book_filter,
                source_filter=source_filter,
                on_stage=on_stage,
            )
        else:
            chunks = self.retriever.search(
                query=query,
                top_k=top_k,
                candidate_pool=candidate_pool,
                stitching_window=stitching_window,
                book_filter=book_filter,
                source_filter=source_filter,
                on_stage=on_stage,
            )

        if not chunks:
            yield app_text["agent"]["stream_no_chunks_message"]
            return

        if self.mode == "focus" and self._is_pure_scene_response(chunks):
            # Same reasoning as ask(): pre-woven chronicles are already
            # finished prose, so stream them back directly instead of
            # re-narrating through the LLM a second time.
            full_response = self._concat_scene_chronicles(chunks)
            yield full_response

            sources_data = [
                {
                    "source": c.get("source", "?"),
                    "chapter": c.get("chapter", "?"),
                    "stitch_range": c.get("stitch_range", ""),
                    "score": round(c.get("scene_score") or c.get("rrf_score") or 0.0, 4),
                }
                for c in chunks
            ]
            self._update_memory(query, full_response)
            yield f"__SOURCES__:{json.dumps(sources_data)}"
            return

        system_prompt, user_msg = self._build_prompt(query, chunks)

        if self._memory:
            user_msg = self._format_memory() + "\n\n" + user_msg

        if on_stage:
            on_stage("generating")

        full_response = ""
        for token in self._stream_llm(system_prompt, user_msg):
            full_response += token
            yield token
            if token.startswith("[ERROR]"):
                return

        sources_data = [
            {
                "source": c.get("source", "?"),
                "chapter": c.get("chapter", "?"),
                "stitch_range": c.get("stitch_range", ""),
                "score": round(
                    c.get("rerank_score")
                    or c.get("scene_score")
                    or c.get("query_overlap_score")
                    or c.get("rrf_score")
                    or c.get("faiss_score")
                    or 0.0,
                    4,
                ),
            }
            for c in chunks
        ]
        if "[ERROR]" not in full_response:
            self._update_memory(query, full_response)
        yield f"__SOURCES__:{json.dumps(sources_data)}"

    def _build_prompt(self, query: str, chunks: list[dict]) -> tuple[str, str]:
        """Dispatch to the right prompt builder based on current mode."""
        if self.mode == "narrator":
            return build_narrate_prompt(query, chunks)
        elif self.mode == "explorer":
            return build_object_explorer_prompt(query, chunks)
        elif self.mode == "focus":
            # Only reached for the FALLBACK path (raw corpus chunks) — the
            # scene-library path returns chronicles directly and never
            # calls the LLM at all (see _is_pure_scene_response check above).
            return build_focus_prompt(query, chunks)
        else:
            return build_prompt(query, chunks)

    @staticmethod
    def _is_pure_scene_response(chunks: list[dict]) -> bool:
        """True when every chunk came from the pre-woven scene library
        (retriever._scenes_to_chunks tags them file_type='scene'), as
        opposed to the live-fallback corpus search path."""
        return bool(chunks) and all(c.get("file_type") == "scene" for c in chunks)

    @staticmethod
    def _concat_scene_chronicles(chunks: list[dict]) -> str:
        """Joins pre-woven chronicle texts as-is — no LLM pass, no
        additional invention risk. Plain double-linebreak join, no
        markdown, so it stays TTS-clean like every other response path."""
        parts = [c.get("text", "").strip() for c in chunks if c.get("text", "").strip()]
        return "\n\n".join(parts)

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
        """Build a request payload for the selected LLM provider."""
        provider = llm_cfg.get("provider", "openai").lower()

        if provider == "gemini":
            generation_config = {
                "temperature": llm_cfg["temperature"],
                "topP": llm_cfg["top_p"],
            }
            max_tokens = llm_cfg.get("max_tokens")
            if max_tokens is not None:
                generation_config["maxOutputTokens"] = max_tokens

            return {
                "system_instruction": {
                    "parts": [{"text": system_prompt}]
                },
                "contents": [{
                    "role": "user",
                    "parts": [{"text": user_msg}]
                }],
                "generationConfig": generation_config,
            }

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

    def _gemini_request_details(self, stream: bool = False):
        """Build the native Gemini REST URL and authentication headers."""
        import os

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        model = llm_cfg["model"]
        action = "streamGenerateContent?alt=sse" if stream else "generateContent"
        base_url = llm_cfg.get(
            "api_base",
            "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")

        return (
            f"{base_url}/models/{model}:{action}",
            {"x-goog-api-key": api_key},
        )

    def _call_llm_sync(self, system_prompt: str, user_msg: str) -> str:
        provider = llm_cfg.get("provider", "openai").lower()
        payload = self._build_payload(system_prompt, user_msg, stream=False)

        if provider == "gemini":
            try:
                return "".join(generate(llm_cfg, payload, stream=False)).strip()
            except ProviderError as exc:
                return f"[ERROR] {exc}"

        try:
            if provider == "gemini":
                url, headers = self._gemini_request_details()
            else:
                url, headers = llm_cfg["url"], {}

            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=llm_cfg.get("timeout", 300),
            )
            resp.raise_for_status()
            data = resp.json()

            if provider == "gemini":
                candidates = data.get("candidates") or []
                if not candidates:
                    return "[ERROR] Gemini returned no response candidates."
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(
                    part.get("text", "") for part in parts
                ).strip()

            return data["choices"][0]["message"]["content"].strip()

        except RuntimeError as e:
            return f"[ERROR] {e}"
        except requests.exceptions.ConnectionError:
            return f"[ERROR] Could not connect to {provider}."
        except requests.exceptions.Timeout:
            return f"[ERROR] {provider} request timed out."
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            detail = e.response.text[:1000] if e.response is not None else str(e)
            return f"[ERROR] {provider} HTTP {status}: {detail}"
        except (KeyError, IndexError, ValueError) as e:
            return f"[ERROR] Unexpected {provider} response: {e}"
        except Exception as e:
            return f"[ERROR] {provider} call failed: {e}"

    def _call_llm_stream(self, system_prompt: str, user_msg: str) -> str:
        return "".join(self._stream_llm(system_prompt, user_msg))

    def _stream_llm(self, system_prompt: str, user_msg: str) -> Generator[str, None, None]:
        """Stream from Gemini native SSE or an OpenAI-compatible endpoint."""
        provider = llm_cfg.get("provider", "openai").lower()
        payload = self._build_payload(system_prompt, user_msg, stream=True)

        if provider == "gemini":
            try:
                yield from generate(llm_cfg, payload, stream=True)
            except ProviderError as exc:
                yield f"[ERROR] {exc}"
            return

        try:
            if provider == "gemini":
                url, headers = self._gemini_request_details(stream=True)
            else:
                url, headers = llm_cfg["url"], {}

            with requests.post(
                url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=llm_cfg.get("timeout", 600),
            ) as resp:
                resp.raise_for_status()

                for raw_line_bytes in resp.iter_lines():
                    if not raw_line_bytes:
                        continue

                    raw_line = raw_line_bytes.decode("utf-8", errors="replace")
                    if not raw_line.startswith("data:"):
                        continue

                    chunk_str = raw_line[len("data:"):].strip()
                    if chunk_str == "[DONE]":
                        break

                    try:
                        data = json.loads(chunk_str)
                    except json.JSONDecodeError:
                        continue

                    if provider == "gemini":
                        candidates = data.get("candidates") or []
                        if not candidates:
                            continue
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            token = part.get("text", "")
                            if token:
                                yield token
                    else:
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        token = choices[0].get("delta", {}).get("content", "")
                        if token:
                            yield token

        except RuntimeError as e:
            yield f"[ERROR] {e}"
        except requests.exceptions.ConnectionError:
            yield f"[ERROR] Could not connect to {provider}."
        except requests.exceptions.Timeout:
            yield f"[ERROR] {provider} request timed out."
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            detail = e.response.text[:1000] if e.response is not None else str(e)
            yield f"[ERROR] {provider} HTTP {status}: {detail}"
        except Exception as e:
            yield f"[ERROR] {provider} streaming failed: {e}"

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
