"""Real runtime/HTTP bridge, mocked ML only. Never uses an API key or corpus."""
import asyncio
import importlib.util
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi import FastAPI
from fastapi.testclient import TestClient
from Api.services.stream_bridge import stream_events


def load_runtime():
    config = types.ModuleType("Core.config_loader")
    for name in ("llm_cfg", "paths", "retrieval_cfg"):
        setattr(config, name, {})
    config.active_profile = config.machine_role = "test"
    agent = types.ModuleType("Core.agent")
    agent.OmnissiahAgent = object
    retriever = types.ModuleType("Core.retriever")
    retriever.OmnissiahRetriever = object
    prompt = types.ModuleType("Core.prompt")
    prompt.build_prompt = lambda *args: ("", "")
    with patch.dict(sys.modules, {
        "faiss": types.ModuleType("faiss"), "numpy": types.ModuleType("numpy"),
        "Core.config_loader": config, "Core.agent": agent,
        "Core.retriever": retriever, "Core.prompt": prompt,
    }), patch.dict(os.environ, {"OMNISSIAH_CONVERSATION_DB": ""}):
        path = Path(__file__).resolve().parents[2] / "Api/services/runtime_service.py"
        spec = importlib.util.spec_from_file_location("_runtime_stream_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class RuntimeStreamTests(unittest.TestCase):
    def setUp(self):
        self.module = load_runtime()
        self.runtime = self.module.RuntimeService()
        self.request = types.SimpleNamespace(
            session_id="test-session", query="question", book_filter=None,
            source_filter=None, top_k=None, candidate_pool=None,
            stitching_window=None, max_sources=None)
        self.saved = []
        self.closed = False
        self.runtime._set_session_memory = lambda *args: self.saved.append(args)

    def stub(self, tokens):
        def generate(**kwargs):
            try:
                yield from tokens
            finally:
                self.closed = True
        agent = types.SimpleNamespace(ask_stream=generate, memory=[{"query": "q", "response": "a"}])
        self.runtime._build_agent = lambda *args, **kwargs: agent
        return agent

    def test_success_preserves_text_and_saves_once_before_completion(self):
        self.stub(["first/second\n\nnext", "__SOURCES__:[]"])
        frames = list(self.runtime.stream_query_mode(self.request))
        self.assertIn("data: first/second\ndata: \ndata: next\n\n", frames)
        self.assertEqual(frames[-1], "data: __STAGE__:complete\n\n")
        self.assertEqual(len(self.saved), 1)
        self.assertEqual(self.module.get_llm_status()["active"], 0)
        self.assertTrue(self.closed)

    def test_empty_and_provider_errors_never_save_or_complete(self):
        for tokens in ([], ["   ", "__SOURCES__:[]"], ["Partial", "[ERROR] quota"]):
            with self.subTest(tokens=tokens):
                self.stub(tokens)
                text = "".join(self.runtime.stream_query_mode(self.request))
                self.assertIn("[ERROR]", text)
                self.assertNotIn("__STAGE__:complete", text)
                self.assertFalse(self.saved)
                self.assertEqual(self.module.get_llm_status()["active"], 0)

    def test_closing_partial_stream_releases_slot_without_memory_write(self):
        self.stub(["partial", " rest"])
        stream = self.runtime.stream_query_mode(self.request)
        next(stream)
        stream.close()
        self.assertTrue(self.closed)
        self.assertFalse(self.saved)
        self.assertEqual(self.module.get_llm_status()["active"], 0)

    def test_cancelled_waiter_does_not_start_generation(self):
        self.module._acquire_llm_slot()
        cancelled = threading.Event()
        worker_done = threading.Event()
        result = []
        def run():
            result.extend(self.runtime.stream_query_mode(self.request, cancel_event=cancelled))
            worker_done.set()
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        cancelled.set()
        try:
            self.assertTrue(worker_done.wait(2))
            self.assertFalse(result)
            self.assertEqual(self.module.get_llm_status()["queued"], 0)
            self.assertEqual(self.module.get_llm_status()["active"], 1)
        finally:
            self.module._release_llm_slot()

    def test_invalid_sync_session_is_client_error(self):
        self.request.session_id = "x"
        with self.assertRaises(HTTPException) as failure:
            self.runtime.run_query(self.request, "narrator", False)
        self.assertEqual(failure.exception.status_code, 400)


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_closes_producer_and_bounds_buffer(self):
        closed = threading.Event()
        produced = []
        def factory(stopped):
            try:
                for i in range(10000):
                    produced.append(i)
                    yield f"data: {i}\n\n"
            finally:
                closed.set()
        stream = stream_events(factory, buffer_size=2)
        await anext(stream)
        await stream.aclose()
        self.assertTrue(await asyncio.to_thread(closed.wait, 2))
        self.assertLessEqual(len(produced), 4)

    async def test_error_closes_upstream_and_hides_provider_details(self):
        closed = threading.Event()
        def factory(stopped):
            try:
                yield "data: [ERROR] private-model secret quota\n\n"
                self.fail("Must not read more after terminal error")
            finally:
                closed.set()
        result = "".join([frame async for frame in stream_events(factory)])
        self.assertIn("[ERROR]", result)
        self.assertNotIn("private-model", result)
        self.assertNotIn("[DONE]", result)
        self.assertTrue(closed.is_set())


class HttpIntegrationTests(unittest.TestCase):
    def setUp(self):
        module = load_runtime()
        self.runtime = module.RuntimeService()
        self.runtime._retriever = types.SimpleNamespace(
            search=lambda **kwargs: [{"source": "test-book", "chunk_id": 1, "text": "Anchor evidence."}],
            search_focus=lambda **kwargs: [{"source": "test-book", "chunk_id": 1, "text": "Anchor evidence."}])
        self.tokens = ["An answer.\n\nAnother paragraph.", "__SOURCES__:[]"]
        self.modes = []
        def build(session, mode, req=None):
            self.modes.append(mode)
            return types.SimpleNamespace(
                memory=[{"query": "question", "response": "answer"}],
                ask_stream=lambda **kwargs: iter(self.tokens),
                ask=lambda **kwargs: ("An answer.", []))
        # A closeable generator, as supplied by the real agent.
        def build_generator(session, mode, req=None):
            agent = build(session, mode, req)
            def tokens(**kwargs):
                yield from self.tokens
            agent.ask_stream = tokens
            return agent
        self.runtime._build_agent = build_generator
        module.runtime_service = self.runtime
        config = types.ModuleType("Core.config_loader")
        config.paths = {}
        with patch.dict(sys.modules, {
            "Api.services.runtime_service": module, "Core.config_loader": config,
        }):
            path = Path(__file__).resolve().parents[2] / "Api/routes/query_routes.py"
            spec = importlib.util.spec_from_file_location("_query_http_test", path)
            routes = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(routes)
        app = FastAPI()
        app.include_router(routes.router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_all_stream_routes_share_terminal_contract_over_http(self):
        for path, mode in (("/query/stream", "remembrancer"), ("/query/narrate", "narrator"),
                           ("/query/focus/stream", "focus"), ("/query/respond/stream", "adaptive")):
            with self.subTest(path=path):
                response = self.client.post(path, json={"query": "question"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.text.count("data: [DONE]"), 1)
                self.assertIn("data: An answer.\ndata: \ndata: Another paragraph.", response.text)
                self.assertEqual(self.modes[-1], mode)

    def test_empty_stream_is_terminal_error_over_http(self):
        self.tokens = ["__SOURCES__:[]"]
        response = self.client.post("/query/narrate", json={"query": "question"})
        self.assertIn("[ERROR]", response.text)
        self.assertNotIn("[DONE]", response.text)
        self.assertFalse(self.runtime._session_memory)

    def test_invalid_session_rejected_before_stream_headers(self):
        response = self.client.post("/query/narrate", json={"query": "question", "session_id": "x"})
        self.assertEqual(response.status_code, 400)

    def test_style_and_scope_are_independent_real_model_validation(self):
        result = self.client.post("/query/voice-preview", json={
            "query": "Why did he leave?", "response_style": "narrator", "retrieval_scope": "focus"})
        self.assertEqual(result.json()["response_style"], "narrator")
        self.assertEqual(result.json()["retrieval_scope"], "focus")
        self.assertFalse(result.json()["llm_used"])
        self.assertEqual(self.client.post("/query/voice-preview", json={
            "query": "question", "response_style": "invalid"}).status_code, 422)

    def test_final_evidence_route_never_generates(self):
        result = self.client.post("/query/evidence", json={"query": "question"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["hits"][0]["text"], "Anchor evidence.")
        self.assertEqual(self.modes, [])
