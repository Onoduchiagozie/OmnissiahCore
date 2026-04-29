"""
test_query_routes.py
====================
Tests every endpoint in Api/routes/query_routes.py.

Run from project root:
    pytest tests/test_query_routes.py -v

Requirements:
    pip install pytest httpx --break-system-packages

The FastAPI app must be running:
    uvicorn Api.server:app --port 8000
OR the tests hit the live app via httpx.AsyncClient(app=app).

Two query tiers are tested each time:
  EASY  - short, single-entity lookup   → fast, low token load
  HARD  - complex cross-source narrative → stress-tests retrieval + Ollama

WHY THE TIMEOUT HAPPENED
-------------------------
Your response contained "[ERROR] Ollama timed out."
Root causes (in order of likelihood):

1. num_ctx is too large for your hardware.
   - num_ctx controls the context window Ollama loads into VRAM/RAM.
   - A 19 GB model with a large num_ctx (e.g. 8192+) can take 60–120 s on a
     mid-range GPU or CPU-only machine.
   FIX: lower num_ctx to 2048 or 4096 in your config profile.

2. The Ollama timeout in your config_loader is too short.
   - If ollama_cfg["timeout"] is e.g. 30s, the model simply hasn't finished.
   FIX: raise timeout to 120–180 s, or set it to 0 (no timeout) while testing.

3. Too many chunks being stitched into the prompt.
   - 8 chunks × stitching_window=2 can produce a very long system prompt.
   FIX: reduce top_k to 3 and candidate_pool to 10 while testing.

4. The model itself is too large for real-time use.
   FIX: switch to a smaller quantized variant (e.g. mistral:7b-q4 instead of
        a 19 GB full-precision model) for development.

HOW TO CONFIRM WHICH CAUSE:
   Hit GET /config/runtime — it shows current timeout, num_ctx, model.
   If timeout < 60  → raise it first.
   If num_ctx > 4096 → lower it.
   If model is >13B → try a smaller one.
"""

import pytest
import httpx

BASE = "http://localhost:8000"

# ── Test payloads ─────────────────────────────────────────────────────────────

EASY_QUERY = {
    "query": "Who is Leman Russ?",
    "session_id": "test-easy",
}

HARD_QUERY = {
    "query": "Espionage plot involving Leman Russ",
    "session_id": "test-hard",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_query_response(data: dict, query_text: str):
    """Shared shape-check for QueryResponse."""
    assert "query" in data, f"Missing 'query' field: {data}"
    assert "response" in data, f"Missing 'response' field: {data}"
    assert "sources" in data, f"Missing 'sources' field: {data}"
    assert "chunks_used" in data, f"Missing 'chunks_used' field: {data}"

    # Detect the Ollama error string and fail with a clear message
    if "[ERROR] Ollama timed out" in data["response"]:
        pytest.fail(
            "\n\nOllama TIMEOUT detected.\n"
            "Diagnosis steps:\n"
            "  1. GET /config/runtime and check 'timeout' and 'num_ctx'.\n"
            "  2. Raise timeout to 120s in your config profile.\n"
            "  3. Lower num_ctx to 2048 or 4096.\n"
            "  4. Try a smaller model (e.g. mistral:7b-q4).\n"
            f"Query that triggered it: {query_text!r}"
        )

    if "[ERROR]" in data["response"]:
        pytest.fail(
            f"\n\nOllama returned an error for query {query_text!r}:\n"
            f"  response = {data['response']}\n"
            "Check Ollama logs: ollama serve (in a separate terminal)."
        )

    assert isinstance(data["sources"], list), "sources must be a list"
    assert data["chunks_used"] >= 0


# ── POST /query ───────────────────────────────────────────────────────────────

class TestQuerySync:
    def test_easy_query_shape(self):
        """EASY: single-entity lookup — should be fast and low-load."""
        r = httpx.post(f"{BASE}/query", json=EASY_QUERY, timeout=120)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        _assert_query_response(data, EASY_QUERY["query"])

    def test_hard_query_shape(self):
        """HARD: cross-source narrative — stress-tests retrieval + Ollama."""
        r = httpx.post(f"{BASE}/query", json=HARD_QUERY, timeout=180)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        _assert_query_response(data, HARD_QUERY["query"])

    def test_empty_query_returns_400(self):
        """Validation: empty query must be rejected before hitting Ollama."""
        r = httpx.post(f"{BASE}/query", json={"query": "   ", "session_id": "test"}, timeout=30)
        assert r.status_code == 400, (
            f"Expected 400 for empty query, got {r.status_code}. "
            "Add a .strip() guard in query_routes.py if missing."
        )

    def test_sources_have_required_fields(self):
        """Each source must carry source, chapter, stitch_range, score."""
        r = httpx.post(f"{BASE}/query", json=EASY_QUERY, timeout=120)
        assert r.status_code == 200
        sources = r.json().get("sources", [])
        for s in sources:
            for field in ("source", "chapter", "stitch_range", "score"):
                assert field in s, f"Source missing field '{field}': {s}"


# ── POST /query/narrate ───────────────────────────────────────────────────────

class TestQueryNarrate:
    def test_easy_narrate(self):
        """EASY: narrator mode on a simple entity should return prose."""
        r = httpx.post(f"{BASE}/query/narrate", json=EASY_QUERY, timeout=120)
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        data = r.json()
        _assert_query_response(data, EASY_QUERY["query"])

    def test_hard_narrate(self):
        """HARD: narrator mode on a complex plot — high token load."""
        r = httpx.post(f"{BASE}/query/narrate", json=HARD_QUERY, timeout=180)
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        data = r.json()
        _assert_query_response(data, HARD_QUERY["query"])


# ── POST /query/explore ───────────────────────────────────────────────────────

class TestQueryExplore:
    def test_easy_explore(self):
        """EASY: explorer mode — object inspection, lower generation load."""
        r = httpx.post(f"{BASE}/query/explore", json=EASY_QUERY, timeout=120)
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        data = r.json()
        assert "query" in data
        assert "response" in data

    def test_hard_explore(self):
        """HARD: explorer mode on a complex narrative query."""
        r = httpx.post(f"{BASE}/query/explore", json=HARD_QUERY, timeout=180)
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        data = r.json()
        assert "query" in data
        assert "response" in data


# ── POST /query/stream ────────────────────────────────────────────────────────

class TestQueryStream:
    def test_easy_stream_returns_tokens(self):
        """EASY: SSE stream — check we receive at least one data: line."""
        tokens = []
        with httpx.stream(
            "POST", f"{BASE}/query/stream", json=EASY_QUERY, timeout=120
        ) as r:
            assert r.status_code == 200, f"{r.status_code}: {r.text}"
            for line in r.iter_lines():
                if line.startswith("data: "):
                    token = line[len("data: "):]
                    if token == "[DONE]":
                        break
                    tokens.append(token)

        assert len(tokens) > 0, (
            "Stream returned no tokens. "
            "Check Ollama is running and the model is loaded: ollama list"
        )

    def test_hard_stream_returns_tokens(self):
        """HARD: SSE stream on a complex query — longer wait expected."""
        tokens = []
        with httpx.stream(
            "POST", f"{BASE}/query/stream", json=HARD_QUERY, timeout=180
        ) as r:
            assert r.status_code == 200, f"{r.status_code}: {r.text}"
            for line in r.iter_lines():
                if line.startswith("data: "):
                    token = line[len("data: "):]
                    if token == "[DONE]":
                        break
                    tokens.append(token)

        assert len(tokens) > 0, "Hard stream query returned no tokens."

    def test_stream_ends_with_done(self):
        """Stream must terminate with [DONE], not hang."""
        done_seen = False
        with httpx.stream(
            "POST", f"{BASE}/query/stream", json=EASY_QUERY, timeout=120
        ) as r:
            for line in r.iter_lines():
                if line == "data: [DONE]":
                    done_seen = True
                    break

        assert done_seen, (
            "[DONE] sentinel never arrived. "
            "The stream_query generator in runtime_service.py may have thrown "
            "an exception before yielding 'data: [DONE]\\n\\n'."
        )


# ── POST /query/inspect ───────────────────────────────────────────────────────

class TestQueryInspect:
    def test_inspect_easy(self):
        """EASY: inspect retrieval without running Ollama — safest health check."""
        r = httpx.post(f"{BASE}/query/inspect", json=EASY_QUERY, timeout=60)
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        data = r.json()
        assert "inspection" in data, f"Missing 'inspection': {data}"
        assert "prompt_preview" in data, f"Missing 'prompt_preview': {data}"

    def test_inspect_hard(self):
        """HARD: inspect a complex query — validates retrieval stack under load."""
        r = httpx.post(f"{BASE}/query/inspect", json=HARD_QUERY, timeout=60)
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        data = r.json()
        assert "inspection" in data