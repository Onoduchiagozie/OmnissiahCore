"""Isolated FastAPI contract checks; no GPU, corpus, network, or real API keys."""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

import test_graph_workflow  # requires the real LangGraph runtime

ROOT = Path(__file__).resolve().parents[2]


class FakeQueryRequest(BaseModel):
    query: str
    session_id: str | None = "default"
    book_filter: str | None = None
    source_filter: list[str] | None = None
    top_k: int | None = None
    candidate_pool: int | None = None
    stitching_window: int | None = None
    response_style: str | None = None
    retrieval_scope: str | None = None
    max_sources: int = 2


class StubAgent:
    def __init__(self):
        self._memory = []
        self.calls = []

    @property
    def memory(self):
        return list(self._memory)

    def _build_prompt(self, question, chunks):
        self.calls.append((question, chunks))
        return "Respect grounded evidence", question

    def _format_memory(self):
        return "memory"

    def _call_llm_sync(self, sys_prompt, usr_prompt):
        return "Ferrus Manus and Fulgrim appear in the supplied sources."

    def _update_memory(self, question, answer):
        self._memory.append({"query": question, "response": answer})


class StubRuntime:
    @staticmethod
    def _resolve_session(session_id):
        if not session_id or session_id == "default":
            return "anon-" + uuid.uuid4().hex
        if "/" in session_id:
            raise ValueError("Invalid session")
        return session_id

    def __init__(self):
        self.agent = StubAgent()
        self.sources = [
            {"source": "Fulgrim.pdf", "chapter": "Chapter 1",
             "text": "Ferrus Manus and Fulgrim faced one another in the battle as described by the witnesses."}
        ]
        self.calls = []
        self.memory = None

    def ensure_ready(self):
        pass

    @property
    def retriever(self):
        return self

    def search(self, question, **options):
        self.calls.append((question, options))
        return list(self.sources)

    def _build_agent(self, session, mode, req=None):
        return self.agent

    def _set_session_memory(self, session, memory):
        self.memory = (session, memory)


class GraphApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = StubRuntime()
        cls.held = [0]

        api_models = types.ModuleType("Api.models")
        api_models.QueryRequest = FakeQueryRequest
        config = types.ModuleType("Core.config_loader")
        config.retrieval_cfg = {"top_k": 5}
        rt = types.ModuleType("Api.services.runtime_service")
        rt.runtime_service = cls.runtime
        rt._acquire_llm_slot = lambda: cls.held.__setitem__(0, cls.held[0] + 1)
        rt._release_llm_slot = lambda: cls.held.__setitem__(0, cls.held[0] - 1)
        cls.patch = patch.dict(sys.modules, {
            "Api.models": api_models,
            "Core.config_loader": config,
            "Api.services.runtime_service": rt,
        })
        cls.patch.start()
        spec = importlib.util.spec_from_file_location("isolated_graph_routes", ROOT / "Api/routes/graph_routes.py")
        cls.route = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.route)
        app = FastAPI()
        app.include_router(cls.route.router)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.patch.stop()

    def setUp(self):
        self.env_patch = patch.dict(os.environ, {"OMNISSIAH_GRAPH_VERIFY": "0"})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.runtime.agent = StubAgent()
        self.runtime.calls = []
        self.runtime.memory = None
        self.runtime.sources = [{"source": "Fulgrim.pdf", "chapter": "Chapter 1",
                                 "text": "Ferrus Manus and Fulgrim faced one another in the battle as described by the witnesses."}]
        self.held[0] = 0

    def test_end_to_end_route_return_contract(self):
        response = self.client.post("/query/graph", json={"query": "Ferrus vs Fulgrim", "session_id": "z"})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["attempts"], 1)
        self.assertTrue(data["sources"])
        self.assertEqual(self.held[0], 0)
        self.assertEqual(self.runtime.memory[0], "z")

    def test_missing_sessions_are_distinct_and_returned(self):
        first = self.client.post("/query/graph", json={"query": "Ferrus vs Fulgrim"}).json()
        second = self.client.post("/query/graph", json={"query": "Ferrus vs Fulgrim"}).json()
        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertTrue(first["session_id"].startswith("anon-"))

    def test_invalid_session_is_client_error(self):
        response = self.client.post("/query/graph", json={
            "query": "Ferrus vs Fulgrim", "session_id": "../invalid"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.held[0], 0)

    def test_memory_load_and_commit_share_exclusive_generation_slot(self):
        observations = []
        build = self.runtime._build_agent
        save = self.runtime._set_session_memory
        def checked_build(*args, **kwargs):
            observations.append(("load", self.held[0]))
            return build(*args, **kwargs)
        def checked_save(*args):
            observations.append(("save", self.held[0]))
            return save(*args)
        with patch.object(self.runtime, "_build_agent", checked_build), patch.object(
                self.runtime, "_set_session_memory", checked_save):
            response = self.client.post("/query/graph", json={"query": "Ferrus vs Fulgrim"})
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(observations, [("load", 1), ("save", 1)])
        self.assertEqual(self.held[0], 0)

    def test_inner_retrieval_graph_does_not_get_outer_retries(self):
        self.runtime.sources = []
        with patch.dict(self.route.retrieval_cfg, {"langgraph_retrieval": True}):
            response = self.client.post("/query/graph/inspect", json={"query": "Why did Fulgrim change?"})
        self.assertEqual(response.json()["attempts"], 1)
        self.assertEqual(len(self.runtime.calls), 1)

    def test_filter_reaches_actual_retrieval_call(self):
        self.client.post("/query/graph", json={"query": "Ferrus", "book_filter": "Fulgrim",
                                               "source_filter": ["Fulgrim.pdf"]})
        self.assertEqual(self.runtime.calls[0][1]["book_filter"], "Fulgrim")
        self.assertEqual(self.runtime.calls[0][1]["source_filter"], ["Fulgrim.pdf"])
        with patch.object(self.runtime, "search_focus", create=True, return_value=self.runtime.sources) as focused:
            self.client.post("/query/graph/inspect", json={
                "query": "Ferrus", "book_filter": "Fulgrim", "source_filter": ["Fulgrim.pdf"],
                "retrieval_scope": "focus", "max_sources": 1})
            self.assertEqual(focused.call_args.kwargs["book_filter"], "Fulgrim")
            self.assertEqual(focused.call_args.kwargs["source_filter"], ["Fulgrim.pdf"])
            self.assertEqual(focused.call_args.kwargs["max_sources"], 1)

    def test_blank_query_is_http_400(self):
        response = self.client.post("/query/graph", json={"query": "    "})
        self.assertEqual(response.status_code, 400)

    def test_model_exception_does_not_leak_semaphore(self):
        self.runtime.agent._call_llm_sync = lambda *args: (_ for _ in ()).throw(RuntimeError("model unavailable"))
        response = self.client.post("/query/graph", json={"query": "Ferrus vs Fulgrim"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "generation_failed")
        self.assertEqual(self.held[0], 0)
        self.assertIsNone(self.runtime.memory)

    def test_insufficient_evidence_no_model_no_memory(self):
        self.runtime.sources = []
        response = self.client.post("/query/graph", json={"query": "Who is unknown?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        self.assertEqual(self.runtime.agent.calls, [])
        self.assertIsNone(self.runtime.memory)
        self.assertEqual(self.held[0], 0)



    def test_graph_inspect_does_not_call_llm_or_write_memory(self):
        self.runtime.agent._call_llm_sync = lambda *a: (_ for _ in ()).throw(
            RuntimeError("LLM called in inspection"))
        response = self.client.post("/query/graph/inspect", json={"query": "Ferrus vs Fulgrim"})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["status"], "inspection_complete")
        self.assertTrue(data["passages"])
        self.assertEqual(data["verification"], "not_run")
        self.assertIsNone(self.runtime.memory)
        self.assertEqual(self.held[0], 0)

    def test_verified_route_accepts_strict_valid_json_only(self):
        os.environ["OMNISSIAH_GRAPH_VERIFY"] = "1"
        def generate_or_verify(system, user):
            if "strict factual consistency auditor" in system:
                return '{"supported":true,"evidence_ids":[1]}'
            return "Ferrus Manus and Fulgrim are recorded together in the battle."
        self.runtime.agent._call_llm_sync = generate_or_verify
        response = self.client.post("/query/graph", json={"query": "Ferrus vs Fulgrim"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["verification"], "passed")
        self.assertEqual(self.held[0], 0)
        self.assertIsNotNone(self.runtime.memory)

    def test_verifier_rejects_unsupported_response_and_memory(self):
        os.environ["OMNISSIAH_GRAPH_VERIFY"] = "1"
        def generate_or_verify(system, user):
            if "strict factual consistency auditor" in system:
                return '{"supported":false,"evidence_ids":[]}'
            return "An unsupported plot claim."
        self.runtime.agent._call_llm_sync = generate_or_verify
        response = self.client.post("/query/graph", json={"query": "Ferrus vs Fulgrim"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "verification_failed")
        self.assertNotIn("unsupported plot claim", response.json()["response"])
        self.assertIsNone(self.runtime.memory)
        self.assertEqual(self.held[0], 0)

    def test_verifier_malformed_json_rejects_without_memory(self):
        os.environ["OMNISSIAH_GRAPH_VERIFY"] = "1"
        self.runtime.agent._call_llm_sync = lambda system, user: "definitely true"
        response = self.client.post("/query/graph", json={"query": "Ferrus vs Fulgrim"})
        self.assertEqual(response.json()["status"], "verification_failed")
        self.assertIsNone(self.runtime.memory)
        self.assertEqual(self.held[0], 0)

    def test_verifier_out_of_range_citation_rejected(self):
        os.environ["OMNISSIAH_GRAPH_VERIFY"] = "1"
        self.runtime.agent._call_llm_sync = lambda system, user: (
            '{"supported":true,"evidence_ids":[999]}' if "strict factual" in system
            else "Ferrus Manus and Fulgrim fought.")
        response = self.client.post("/query/graph", json={"query": "Ferrus vs Fulgrim"})
        self.assertEqual(response.json()["status"], "verification_failed")
        self.assertEqual(self.held[0], 0)

    def test_inspection_invalid_query_is_400(self):
        self.assertEqual(self.client.post("/query/graph/inspect", json={"query": " "}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
