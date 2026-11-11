"""Exercise the *actual FastAPI route functions* without heavy ML imports.

A minimal FastAPI/Runtime dependency stand-in lets this run on Python 3.10
with requirements-test.txt. No external LLM, index or cloud access.
"""
import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
ROUTE = ROOT / "Api" / "routes" / "query_routes.py"


class DummyRequest:
    query = "Narrate the Burning of Prospero"
    session_id = "test"

    def model_copy(self, update=None):
        return self


class DummyRuntime:
    def __init__(self, frames=(), exception=None):
        self.frames = list(frames)
        self.exception = exception
        self.modes = []

    def ensure_ready(self):
        pass

    def stream_query_mode(self, request, mode="narrator"):
        self.modes.append(mode)
        for frame in self.frames:
            yield frame
        if self.exception:
            raise self.exception


class FakeRouter:
    def __init__(self, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return lambda fn: fn

    def post(self, *args, **kwargs):
        return lambda fn: fn


class FakeStreamResponse:
    def __init__(self, body_iterator, **kwargs):
        self.body_iterator = body_iterator
        self.media_type = kwargs.get("media_type")


def get_route_module(runtime):
    fastapi = types.ModuleType("fastapi")
    fastapi.APIRouter = FakeRouter
    fastapi.HTTPException = type("HTTPException", (Exception,), {})
    responses = types.ModuleType("fastapi.responses")
    responses.StreamingResponse = FakeStreamResponse

    models = types.ModuleType("Api.models")
    models.QueryRequest = DummyRequest
    models.QueryResponse = object
    models.SourceInfo = object

    services = types.ModuleType("Api.services.runtime_service")
    services.runtime_service = runtime

    config = types.ModuleType("Core.config_loader")
    config.paths = {"book_manifest": ""}

    modules = {
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "Api.models": models,
        "Api.services.runtime_service": services,
        "Core.config_loader": config,
    }
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "_isolated_fastapi_stream_routes", ROUTE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def run_stream(runtime, endpoint="narrator"):
    routes = get_route_module(runtime)

    async def collect():
        fn = routes.query_narrate_stream if endpoint == "narrator" else routes.query_focus_stream
        response = await fn(DummyRequest())
        return "".join([part async for part in response.body_iterator])

    return asyncio.run(collect())


class RealRouteContractTests(unittest.TestCase):
    def test_narrator_success_produces_exactly_one_done(self):
        runtime = DummyRuntime([
            "data: __STAGE__:generating\n\n",
            "data: The Space Wolves advanced.\n\n",
            'data: __SOURCES__:[{"source":"Prospero.pdf"}]\n\n',
            "data: __STAGE__:complete\n\n",
        ])
        text = run_stream(runtime)
        self.assertEqual(text.count("data: [DONE]\n\n"), 1)
        self.assertIn("The Space Wolves advanced", text)
        self.assertEqual(runtime.modes, ["narrator"])

    def test_focus_failed_provider_has_no_false_done(self):
        runtime = DummyRuntime([
            "data: __STAGE__:generating\n\n",
            "data: [ERROR] Gemini HTTP 503\n\n",
            "data: __STAGE__:failed\n\n",
        ])
        text = run_stream(runtime, endpoint="focus")
        self.assertIn("[ERROR]", text)
        self.assertNotIn("[DONE]", text)
        self.assertEqual(runtime.modes, ["focus"])

    def test_narrator_sources_without_text_fails(self):
        runtime = DummyRuntime([
            'data: __SOURCES__:[{"source":"Nuceria.pdf"}]\n\n',
            "data: __STAGE__:complete\n\n",
        ])
        text = run_stream(runtime)
        self.assertIn("no answer text", text)
        self.assertNotIn("[DONE]", text)

    def test_focus_premature_eof_is_error(self):
        text = run_stream(DummyRuntime(["data: Partial.\n\n"]), endpoint="focus")
        self.assertIn("without confirmed completion", text)
        self.assertNotIn("[DONE]", text)

    def test_narrator_worker_exception_is_terminal(self):
        runtime = DummyRuntime(
            ["data: The blade fell.\n\n"],
            exception=RuntimeError("private backend internals"),
        )
        text = run_stream(runtime)
        self.assertIn("[ERROR]", text)
        self.assertNotIn("[DONE]", text)
        self.assertNotIn("private backend internals", text)


if __name__ == "__main__":
    unittest.main()
