"""
full_suite_test.py
==================
Comprehensive integration test suite for OmnissiahCore RAG API.

Profile target : lenovo_build  (48 GB RAM, qwen3:14b, timeout=600)
Run            : pytest Tests/full_suite_test.py -v -s
Pre-requisite  : uvicorn Api.server:app --port 8000  (API must be live)

Test order matters — the WARMUP class runs first via alphabetical sort
(class name "A_Warmup" sorts before all others).  Every test prints a
[STAGE] banner so you can follow progress in the terminal.

Timeout policy
--------------
  - Non-LLM endpoints  →  30 s
  - Inspect / system   →  60 s
  - Sync query / narrate / explore  →  600 s  (qwen3:14b can be slow)
  - Stream endpoints   →  None  (let httpx block until [DONE])
"""

import json
import time
from typing import Generator

import httpx
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE = "http://localhost:8000"

# Timeouts (seconds).  None = no timeout (used for streams).
T_FAST       = 30
T_INSPECT    = 60
T_WARMUP     = 120
T_INFERENCE  = 600   # qwen3:14b on lenovo_build can take up to 10 min

# ──────────────────────────────────────────────────────────────────────────────
# Lore query payloads  – all Warhammer 40K / Horus Heresy specific
# ──────────────────────────────────────────────────────────────────────────────

# Simple single-entity lookups (light retrieval, moderate generation)
QUERY_EASY = {
    "query": "Who is Ferrus Manus and what Legion does he command?",
    "session_id": "suite-easy",
}

QUERY_ISTVAAN_III = {
    "query": "What occurred during the Istvaan III Atrocity?",
    "session_id": "suite-istvaan3",
}

# Complex multi-source narratives (heavy retrieval + long generation)
QUERY_DROPSITE = {
    "query": "Describe the full sequence of events at the Istvaan V Dropsite Massacre.",
    "session_id": "suite-dropsite",
}

QUERY_FULGRIM = {
    "query": "How was Fulgrim of the Emperor's Children corrupted by Chaos and Slaanesh?",
    "session_id": "suite-fulgrim",
}

QUERY_FERRUS_DUEL = {
    "query": "Recount the duel between Ferrus Manus and Fulgrim at Istvaan V.",
    "session_id": "suite-ferrus-duel",
}

# Cross-source synthesis
QUERY_IRON_HANDS = {
    "query": "What happened to the Iron Hands Legion after Ferrus Manus was killed?",
    "session_id": "suite-iron-hands",
}

# Object / artefact query — suited for explorer mode
QUERY_ARTEFACT = {
    "query": "Describe the blade Fulgrim wielded when he killed Ferrus Manus.",
    "session_id": "suite-artefact",
}

# Memory test session
SESSION_MEMORY = "suite-memory-test"
QUERY_MEMORY = {
    "query": "Who is Corvus Corax?",
    "session_id": SESSION_MEMORY,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _stage(label: str) -> None:
    """Print a visible stage banner in the terminal."""
    print(f"\n{'─' * 60}")
    print(f"  [STAGE: {label}]")
    print(f"{'─' * 60}")


def _assert_ok(r: httpx.Response, stage: str) -> dict:
    """Assert 200, pretty-print body on failure, return parsed JSON."""
    if r.status_code != 200:
        print(f"\n[FAIL] {stage} — HTTP {r.status_code}")
        try:
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print(r.text[:2000])
        pytest.fail(f"{stage}: expected 200, got {r.status_code}")
    return r.json()


def _assert_query_shape(data: dict, query_text: str, stage: str) -> None:
    """Validate the standard QueryResponse envelope."""
    for field in ("query", "response", "sources", "chunks_used"):
        assert field in data, f"[{stage}] Missing field '{field}' in response: {list(data.keys())}"

    resp = data["response"]

    if "[ERROR] Ollama timed out" in resp:
        pytest.fail(
            f"[{stage}] Ollama TIMEOUT for query: {query_text!r}\n"
            "Fix: raise timeout in config.json → ollama.timeout\n"
            "     or lower num_ctx / top_k for testing."
        )

    if "[ERROR]" in resp:
        pytest.fail(
            f"[{stage}] Ollama returned an error for query: {query_text!r}\n"
            f"  response = {resp[:500]}"
        )

    assert isinstance(data["sources"], list), f"[{stage}] 'sources' must be a list"
    assert isinstance(data["chunks_used"], int), f"[{stage}] 'chunks_used' must be int"
    assert data["chunks_used"] >= 0


def _assert_source_fields(sources: list, stage: str) -> None:
    """Every source entry must carry all four required fields."""
    required = ("source", "chapter", "stitch_range", "score")
    for i, s in enumerate(sources):
        for field in required:
            assert field in s, f"[{stage}] Source[{i}] missing '{field}': {s}"


def _collect_sse(r: httpx.Response, stage: str) -> tuple[list[str], dict | None]:
    """
    Consume a streaming SSE response.
    Returns (token_list, sources_payload_or_None).
    Raises if [DONE] is never received.
    """
    tokens: list[str] = []
    sources_payload = None
    done_seen = False

    for line in r.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]

        if payload == "[DONE]":
            done_seen = True
            break
        if payload.startswith("__SOURCES__:"):
            try:
                sources_payload = json.loads(payload[len("__SOURCES__:"):])
            except json.JSONDecodeError:
                print(f"[{stage}] WARNING: could not parse __SOURCES__ payload")
            continue
        if payload.startswith("[ERROR]"):
            pytest.fail(f"[{stage}] Stream returned error token: {payload}")

        tokens.append(payload)

    assert done_seen, (
        f"[{stage}] SSE stream never sent [DONE]. "
        "The generator in runtime_service.py may have thrown mid-stream."
    )
    return tokens, sources_payload


# ──────────────────────────────────────────────────────────────────────────────
# A — Warm-up  (runs first, ensures Ollama model is in VRAM before heavy tests)
# ──────────────────────────────────────────────────────────────────────────────

class TestA_Warmup:
    """
    Stage 0: Verify the server is live and model is ready.
    Hits /health with a long timeout so Ollama has time to load qwen3:14b.
    """

    def test_warmup_health(self):
        _stage("WARMUP — pinging /health (allow 120 s for model to load into VRAM)")
        start = time.time()
        try:
            r = httpx.get(f"{BASE}/health", timeout=T_WARMUP)
        except httpx.ConnectError:
            pytest.fail(
                "Cannot connect to localhost:8000. "
                "Start the API first: uvicorn Api.server:app --port 8000"
            )
        elapsed = time.time() - start
        data = _assert_ok(r, "WARMUP/health")

        print(f"  status          : {data.get('status')}")
        print(f"  active_profile  : {data.get('active_profile')}")
        print(f"  machine_role    : {data.get('machine_role')}")
        print(f"  ollama_model    : {data.get('ollama_model')}")
        print(f"  metadata_loaded : {data.get('metadata_loaded')}")
        print(f"  response time   : {elapsed:.2f}s")

        assert data.get("status") == "online", f"Server not online: {data}"
        assert data.get("metadata_loaded", 0) > 0, (
            "metadata_loaded is 0 — Db/metadata.json may be missing or empty. "
            "Run: python main.py build"
        )


# ──────────────────────────────────────────────────────────────────────────────
# B — System endpoints  (no Ollama)
# ──────────────────────────────────────────────────────────────────────────────

class TestB_System:
    """Non-LLM endpoints: /health, /info, /config/runtime, /sources, /memory."""

    def test_health_shape(self):
        _stage("SYSTEM — /health shape")
        r = httpx.get(f"{BASE}/health", timeout=T_FAST)
        data = _assert_ok(r, "GET /health")
        for field in ("status", "active_profile", "machine_role", "ollama_model", "metadata_loaded"):
            assert field in data, f"Missing '{field}' in /health: {data}"

    def test_info_shape(self):
        _stage("SYSTEM — /info shape")
        r = httpx.get(f"{BASE}/info", timeout=T_FAST)
        data = _assert_ok(r, "GET /info")
        # /info returns index stats — at minimum it must not be empty
        assert isinstance(data, dict) and len(data) > 0, f"/info returned empty response: {data}"
        print(f"  /info payload: {json.dumps(data, indent=2)[:400]}")

    def test_runtime_config_shape(self):
        _stage("SYSTEM — /config/runtime")
        r = httpx.get(f"{BASE}/config/runtime", timeout=T_FAST)
        data = _assert_ok(r, "GET /config/runtime")

        # Warn loudly if lenovo settings look wrong
        timeout_val = data.get("timeout")
        num_ctx     = data.get("num_ctx")
        reranker    = data.get("use_reranker")

        print(f"  timeout      : {timeout_val}")
        print(f"  num_ctx      : {num_ctx}")
        print(f"  use_reranker : {reranker}")
        print(f"  model        : {data.get('model') or data.get('ollama_model')}")

        if timeout_val and timeout_val < 120:
            pytest.fail(
                f"ollama.timeout={timeout_val} is too low for qwen3:14b. "
                "Set it to 600 in lenovo_build profile."
            )

    def test_sources_list(self):
        _stage("SYSTEM — /sources list")
        r = httpx.get(f"{BASE}/sources", timeout=T_FAST)
        data = _assert_ok(r, "GET /sources")
        assert isinstance(data, (dict, list)), f"Unexpected /sources type: {type(data)}"
        print(f"  sources response (first 300 chars): {str(data)[:300]}")

    def test_sources_by_name(self):
        """Fetch a known source by a substring of its name."""
        _stage("SYSTEM — /sources/{source_name}")
        # First get the full sources list to pick a real name
        r_list = httpx.get(f"{BASE}/sources", timeout=T_FAST)
        if r_list.status_code != 200:
            pytest.skip("Cannot fetch /sources list — skipping by-name test.")

        sources_data = r_list.json()
        # Handle both list-of-strings and dict-with-sources-key shapes
        if isinstance(sources_data, list):
            all_names = sources_data
        elif isinstance(sources_data, dict):
            all_names = sources_data.get("sources", [])
        else:
            pytest.skip("Unrecognised /sources shape — skipping by-name test.")

        if not all_names:
            pytest.skip("No sources indexed yet — skipping by-name test.")

        # Pick first source and URL-encode the name
        first = all_names[0] if isinstance(all_names[0], str) else str(all_names[0])
        print(f"  fetching source: {first[:80]}")
        r = httpx.get(f"{BASE}/sources/{first}", timeout=T_FAST)
        # 200 or 404 (if API doesn't support exact match) are both acceptable here
        assert r.status_code in (200, 404), (
            f"Unexpected status {r.status_code} for /sources/{{name}}: {r.text[:200]}"
        )
        if r.status_code == 200:
            data = r.json()
            assert isinstance(data, (dict, list)), f"Unexpected shape: {type(data)}"


# ──────────────────────────────────────────────────────────────────────────────
# C — Inspect  (retrieval only, no Ollama — best health check for the index)
# ──────────────────────────────────────────────────────────────────────────────

class TestC_Inspect:
    """POST /query/inspect — validates FAISS + BM25 + stitching stack."""

    def test_inspect_easy(self):
        _stage("INSPECT — easy entity query (Ferrus Manus)")
        r = httpx.post(f"{BASE}/query/inspect", json=QUERY_EASY, timeout=T_INSPECT)
        data = _assert_ok(r, "POST /query/inspect [easy]")

        assert "inspection" in data, f"Missing 'inspection' key: {list(data.keys())}"
        assert "prompt_preview" in data, f"Missing 'prompt_preview' key: {list(data.keys())}"

        inspection = data["inspection"]
        assert "query_terms" in inspection, "'query_terms' missing from inspection"
        assert "faiss_hits" in inspection, "'faiss_hits' missing from inspection"
        assert "bm25_hits" in inspection, "'bm25_hits' missing from inspection"
        assert "stitched_hits" in inspection, "'stitched_hits' missing from inspection"

        faiss = inspection["faiss_hits"]
        bm25  = inspection["bm25_hits"]
        stitched = inspection["stitched_hits"]

        print(f"  query_terms  : {inspection['query_terms']}")
        print(f"  faiss_hits   : {len(faiss)}")
        print(f"  bm25_hits    : {len(bm25)}")
        print(f"  stitched_hits: {len(stitched)}")

        assert len(faiss) > 0, (
            "faiss_hits is empty — FAISS index may not have loaded. "
            "Check Db/faiss_index.bin exists."
        )
        assert len(bm25) > 0, (
            "bm25_hits is empty — BM25 index may not have loaded. "
            "Check Db/bm25_index.pkl exists and rank_bm25 is installed."
        )

        # Validate chunk_id is a plain Python int (not numpy.int64)
        for hit in faiss:
            assert isinstance(hit.get("chunk_id"), int), (
                f"chunk_id is not a plain int: {type(hit.get('chunk_id'))} — "
                "numpy.int64 serialisation bug may still be present."
            )

    def test_inspect_istvaan_iii(self):
        _stage("INSPECT — Istvaan III Atrocity")
        r = httpx.post(f"{BASE}/query/inspect", json=QUERY_ISTVAAN_III, timeout=T_INSPECT)
        data = _assert_ok(r, "POST /query/inspect [Istvaan III]")
        inspection = data["inspection"]

        print(f"  query_terms  : {inspection.get('query_terms')}")
        print(f"  faiss_hits   : {len(inspection.get('faiss_hits', []))}")
        print(f"  bm25_hits    : {len(inspection.get('bm25_hits', []))}")
        print(f"  stitched_hits: {len(inspection.get('stitched_hits', []))}")

        assert len(inspection.get("faiss_hits", [])) > 0, "No FAISS hits for Istvaan III query."

    def test_inspect_dropsite(self):
        _stage("INSPECT — Dropsite Massacre (stress-tests hybrid retrieval)")
        r = httpx.post(f"{BASE}/query/inspect", json=QUERY_DROPSITE, timeout=T_INSPECT)
        data = _assert_ok(r, "POST /query/inspect [Dropsite]")
        inspection = data["inspection"]

        faiss    = inspection.get("faiss_hits", [])
        bm25     = inspection.get("bm25_hits", [])
        stitched = inspection.get("stitched_hits", [])

        print(f"  query_terms  : {inspection.get('query_terms')}")
        print(f"  faiss_hits   : {len(faiss)}")
        print(f"  bm25_hits    : {len(bm25)}")
        print(f"  stitched_hits: {len(stitched)}")

        assert len(faiss) > 0, "No FAISS hits for Dropsite Massacre query."
        assert len(bm25)  > 0, "No BM25 hits for Dropsite Massacre query."

        # Stitched hits should have stitch_range populated
        for hit in stitched:
            assert hit.get("stitch_range"), (
                f"stitch_range is empty in a stitched hit: {hit.get('chunk_id')}"
            )

        # Cross-source check: ideally hits span multiple books
        sources_seen = {h.get("source") for h in stitched if h.get("source")}
        print(f"  distinct sources in stitched hits: {len(sources_seen)}")
        if len(sources_seen) < 2:
            print(
                "  WARNING: all stitched hits are from one source — "
                "RRF fusion may not be blending FAISS and BM25 results."
            )

    def test_inspect_empty_query_rejected(self):
        _stage("INSPECT — empty query must return 400")
        r = httpx.post(
            f"{BASE}/query/inspect",
            json={"query": "   ", "session_id": "suite-empty"},
            timeout=T_FAST,
        )
        assert r.status_code == 400, (
            f"Expected 400 for empty query on /inspect, got {r.status_code}. "
            "Add a .strip() guard to the inspect route handler."
        )

    def test_inspect_prompt_preview_not_empty(self):
        _stage("INSPECT — prompt_preview must contain real lore text")
        r = httpx.post(f"{BASE}/query/inspect", json=QUERY_EASY, timeout=T_INSPECT)
        data = _assert_ok(r, "POST /query/inspect [prompt preview]")

        system_prompt = data.get("prompt_preview", {}).get("system_prompt", "")
        print(f"  system_prompt length: {len(system_prompt)} chars")

        # Check it's not the fallback "no passages found" message
        no_archive = "[No relevant passages were found in the Archives" in system_prompt
        if no_archive:
            print(
                "  WARNING: system_prompt contains the fallback 'no passages found' message — "
                "stitching may have returned 0 chunks for this query."
            )
        assert len(system_prompt) > 100, (
            "system_prompt is too short — retrieval or stitching may have failed."
        )


# ──────────────────────────────────────────────────────────────────────────────
# D — Sync Query  (POST /query)
# ──────────────────────────────────────────────────────────────────────────────

class TestD_SyncQuery:
    """POST /query — remembrancer mode, synchronous inference."""

    def test_easy_query(self):
        _stage("SYNC_QUERY — easy: Ferrus Manus")
        r = httpx.post(f"{BASE}/query", json=QUERY_EASY, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query [easy]")
        _assert_query_shape(data, QUERY_EASY["query"], "SYNC_QUERY/easy")
        _assert_source_fields(data["sources"], "SYNC_QUERY/easy")

        print(f"  chunks_used : {data['chunks_used']}")
        print(f"  sources     : {len(data['sources'])}")
        print(f"  response[:200]: {data['response'][:200]}")

    def test_istvaan_iii_query(self):
        _stage("SYNC_QUERY — Istvaan III Atrocity")
        r = httpx.post(f"{BASE}/query", json=QUERY_ISTVAAN_III, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query [Istvaan III]")
        _assert_query_shape(data, QUERY_ISTVAAN_III["query"], "SYNC_QUERY/Istvaan3")
        print(f"  chunks_used  : {data['chunks_used']}")
        print(f"  response[:200]: {data['response'][:200]}")

    def test_empty_query_rejected(self):
        _stage("SYNC_QUERY — empty query must return 400")
        r = httpx.post(
            f"{BASE}/query",
            json={"query": "", "session_id": "suite-empty"},
            timeout=T_FAST,
        )
        assert r.status_code == 400, (
            f"Expected 400 for empty query, got {r.status_code}."
        )

    def test_sources_all_have_required_fields(self):
        _stage("SYNC_QUERY — source field completeness check")
        r = httpx.post(f"{BASE}/query", json=QUERY_EASY, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query [source fields]")
        _assert_source_fields(data["sources"], "SYNC_QUERY/sources")

    def test_overridden_top_k_respected(self):
        _stage("SYNC_QUERY — override top_k=3, candidate_pool=15")
        payload = {**QUERY_ISTVAAN_III, "top_k": 3, "candidate_pool": 15}
        r = httpx.post(f"{BASE}/query", json=payload, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query [top_k override]")
        assert data["chunks_used"] <= 3, (
            f"chunks_used={data['chunks_used']} exceeds requested top_k=3."
        )
        print(f"  chunks_used (expect ≤ 3): {data['chunks_used']}")


# ──────────────────────────────────────────────────────────────────────────────
# E — Stream Query  (POST /query/stream)
# ──────────────────────────────────────────────────────────────────────────────

class TestE_Stream:
    """POST /query/stream — SSE format, token-by-token delivery."""

    def test_easy_stream_format(self):
        _stage("STREAM — easy query: SSE format validation")
        with httpx.stream(
            "POST", f"{BASE}/query/stream", json=QUERY_EASY, timeout=None
        ) as r:
            assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.read()[:200]}"
            assert "text/event-stream" in r.headers.get("content-type", ""), (
                f"Content-Type is not text/event-stream: {r.headers.get('content-type')}"
            )
            tokens, sources = _collect_sse(r, "STREAM/easy")

        print(f"  tokens received : {len(tokens)}")
        print(f"  sources present : {sources is not None}")
        assert len(tokens) > 1, (
            "Stream yielded ≤1 token — model may not be streaming properly. "
            "Check ollama.stream=true in config.json."
        )

    def test_dropsite_stream_yields_tokens(self):
        _stage("STREAM — Dropsite Massacre: heavy narrative stream")
        with httpx.stream(
            "POST", f"{BASE}/query/stream", json=QUERY_DROPSITE, timeout=None
        ) as r:
            assert r.status_code == 200, f"{r.status_code}: {r.read()[:200]}"
            tokens, sources = _collect_sse(r, "STREAM/dropsite")

        full_text = "".join(tokens)
        print(f"  tokens received   : {len(tokens)}")
        print(f"  total chars       : {len(full_text)}")
        print(f"  first 200 chars   : {full_text[:200]}")
        assert len(tokens) > 5, "Dropsite stream yielded suspiciously few tokens."

    def test_stream_sources_payload_present(self):
        _stage("STREAM — __SOURCES__ payload must arrive before [DONE]")
        with httpx.stream(
            "POST", f"{BASE}/query/stream", json=QUERY_EASY, timeout=None
        ) as r:
            assert r.status_code == 200
            _, sources = _collect_sse(r, "STREAM/sources_payload")

        assert sources is not None, (
            "No __SOURCES__:{...} payload detected in stream. "
            "Check stream_query in runtime_service.py appends it before [DONE]."
        )
        assert isinstance(sources, (dict, list)), f"Sources payload is not dict/list: {type(sources)}"

    def test_stream_no_swagger_filter_poison(self):
        """
        Sending book_filter='string' / source_filter=['string'] (Swagger defaults)
        should return 'Chronicles do not record this' if filters are not stripped.
        This test confirms clean payloads (None filters) work correctly.
        """
        _stage("STREAM — clean payload (no Swagger filter pollution)")
        payload = {
            "query": "Who is Leman Russ?",
            "session_id": "suite-filter-clean",
            # book_filter and source_filter intentionally absent
        }
        with httpx.stream(
            "POST", f"{BASE}/query/stream", json=payload, timeout=None
        ) as r:
            assert r.status_code == 200
            tokens, _ = _collect_sse(r, "STREAM/no_filter_poison")

        full = "".join(tokens)
        assert "Chronicles do not record this" not in full, (
            "Stream returned 'Chronicles do not record this' even on a clean query — "
            "retrieval is returning 0 chunks. Check FAISS/BM25 indices."
        )


# ──────────────────────────────────────────────────────────────────────────────
# F — Narrate  (POST /query/narrate and /query/narrate/stream)
# ──────────────────────────────────────────────────────────────────────────────

class TestF_Narrate:
    """Narrator mode — prose scene reconstruction."""

    def test_narrate_fulgrim_corruption(self):
        _stage("NARRATE — Fulgrim's corruption (sync)")
        r = httpx.post(f"{BASE}/query/narrate", json=QUERY_FULGRIM, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query/narrate [Fulgrim]")
        _assert_query_shape(data, QUERY_FULGRIM["query"], "NARRATE/fulgrim")
        print(f"  chunks_used  : {data['chunks_used']}")
        print(f"  response[:200]: {data['response'][:200]}")

    def test_narrate_ferrus_duel(self):
        _stage("NARRATE — Ferrus Manus vs Fulgrim duel (sync)")
        r = httpx.post(f"{BASE}/query/narrate", json=QUERY_FERRUS_DUEL, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query/narrate [Ferrus duel]")
        _assert_query_shape(data, QUERY_FERRUS_DUEL["query"], "NARRATE/ferrus_duel")
        print(f"  chunks_used  : {data['chunks_used']}")
        print(f"  response[:200]: {data['response'][:200]}")

    def test_narrate_stream_fulgrim(self):
        _stage("NARRATE_STREAM — Fulgrim's corruption (SSE)")
        with httpx.stream(
            "POST", f"{BASE}/query/narrate/stream", json=QUERY_FULGRIM, timeout=None
        ) as r:
            assert r.status_code == 200, f"{r.status_code}: {r.read()[:200]}"
            tokens, sources = _collect_sse(r, "NARRATE_STREAM/fulgrim")

        full = "".join(tokens)
        print(f"  tokens received : {len(tokens)}")
        print(f"  total chars     : {len(full)}")
        print(f"  first 200 chars : {full[:200]}")
        assert len(tokens) > 1, "Narrate stream returned ≤1 token."

    def test_narrate_stream_dropsite(self):
        _stage("NARRATE_STREAM — Dropsite Massacre (real-time narrative)")
        with httpx.stream(
            "POST", f"{BASE}/query/narrate/stream", json=QUERY_DROPSITE, timeout=None
        ) as r:
            assert r.status_code == 200, f"{r.status_code}: {r.read()[:200]}"
            tokens, _ = _collect_sse(r, "NARRATE_STREAM/dropsite")

        full = "".join(tokens)
        print(f"  tokens received  : {len(tokens)}")
        print(f"  total chars      : {len(full)}")
        assert len(tokens) > 5, "Dropsite narrate stream yielded too few tokens."


# ──────────────────────────────────────────────────────────────────────────────
# G — Explore  (POST /query/explore)
# ──────────────────────────────────────────────────────────────────────────────

class TestG_Explore:
    """Explorer mode — object / artefact analysis."""

    def test_explore_artefact(self):
        _stage("EXPLORE — Fulgrim's blade (artefact query)")
        r = httpx.post(f"{BASE}/query/explore", json=QUERY_ARTEFACT, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query/explore [artefact]")
        _assert_query_shape(data, QUERY_ARTEFACT["query"], "EXPLORE/artefact")
        print(f"  chunks_used  : {data['chunks_used']}")
        print(f"  response[:200]: {data['response'][:200]}")

    def test_explore_iron_hands(self):
        _stage("EXPLORE — Iron Hands after Ferrus (cross-source)")
        r = httpx.post(f"{BASE}/query/explore", json=QUERY_IRON_HANDS, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query/explore [Iron Hands]")
        _assert_query_shape(data, QUERY_IRON_HANDS["query"], "EXPLORE/iron_hands")
        print(f"  chunks_used  : {data['chunks_used']}")
        print(f"  response[:200]: {data['response'][:200]}")


# ──────────────────────────────────────────────────────────────────────────────
# H — Memory  (GET + DELETE /memory)
# ──────────────────────────────────────────────────────────────────────────────

class TestH_Memory:
    """
    Session memory lifecycle:
      1. Seed a session via a real query.
      2. GET /memory to verify context was stored.
      3. DELETE /memory to clear it.
      4. GET /memory again to confirm it is empty.
    """

    def test_memory_lifecycle(self):
        _stage("MEMORY — full lifecycle (seed → get → delete → verify)")

        # 1. Seed the session
        r = httpx.post(f"{BASE}/query", json=QUERY_MEMORY, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query [memory seed]")
        _assert_query_shape(data, QUERY_MEMORY["query"], "MEMORY/seed")
        print(f"  Seeded session '{SESSION_MEMORY}' — chunks_used: {data['chunks_used']}")

        # 2. GET memory — should have content
        r_get = httpx.get(f"{BASE}/memory", params={"session_id": SESSION_MEMORY}, timeout=T_FAST)
        _assert_ok(r_get, "GET /memory [after seed]")
        mem_before = r_get.json()
        print(f"  Memory after seed: {str(mem_before)[:200]}")

        # 3. DELETE memory
        r_del = httpx.delete(f"{BASE}/memory", params={"session_id": SESSION_MEMORY}, timeout=T_FAST)
        assert r_del.status_code in (200, 204), (
            f"DELETE /memory returned {r_del.status_code}: {r_del.text}"
        )
        print(f"  DELETE /memory returned {r_del.status_code}")

        # 4. GET memory — should be empty / cleared
        r_get2 = httpx.get(f"{BASE}/memory", params={"session_id": SESSION_MEMORY}, timeout=T_FAST)
        _assert_ok(r_get2, "GET /memory [after delete]")
        mem_after = r_get2.json()
        print(f"  Memory after delete: {str(mem_after)[:200]}")

        # API returns {"session_id": "...", "memory": []} after DELETE
        # Unwrap the inner memory list before checking emptiness
        if isinstance(mem_after, dict) and "memory" in mem_after:
            inner = mem_after["memory"]
        else:
            inner = mem_after
        is_empty = inner is None or inner == [] or inner == {}
        assert is_empty, (
            f"Memory was not cleared after DELETE — inner memory: {inner}"
        )

    def test_memory_delete_unknown_session_does_not_crash(self):
        _stage("MEMORY — DELETE on unknown session_id should not 500")
        r = httpx.delete(
            f"{BASE}/memory",
            params={"session_id": "suite-nonexistent-session-xyz"},
            timeout=T_FAST,
        )
        assert r.status_code in (200, 204, 404), (
            f"Unexpected status deleting unknown session: {r.status_code}: {r.text}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# I — Lore Integrity  (end-to-end response quality)
# ──────────────────────────────────────────────────────────────────────────────

class TestI_LoreIntegrity:
    """
    Smoke-test that the LLM is actually grounded in the lore.
    These tests check for key proper nouns in responses — they do NOT
    enforce exact wording, just that known lore terms appear.
    """

    def test_ferrus_manus_response_mentions_iron_hands(self):
        _stage("LORE — Ferrus Manus response must reference Iron Hands")
        r = httpx.post(f"{BASE}/query", json=QUERY_EASY, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query [lore: Ferrus Manus]")
        response_lower = data["response"].lower()
        print(f"  response[:300]: {data['response'][:300]}")
        assert "iron hands" in response_lower or "iron hand" in response_lower, (
            "Response about Ferrus Manus doesn't mention the Iron Hands Legion — "
            "retrieval may be pulling off-topic chunks."
        )

    def test_dropsite_response_mentions_betrayal(self):
        _stage("LORE — Dropsite response must mention key lore terms")
        r = httpx.post(f"{BASE}/query", json=QUERY_DROPSITE, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query [lore: Dropsite]")
        response_lower = data["response"].lower()
        print(f"  response[:300]: {data['response'][:300]}")

        lore_terms = ["istvaan", "isstvan", "traitor", "massacre", "betrayal", "heresy"]
        found = [t for t in lore_terms if t in response_lower]
        print(f"  lore terms found: {found}")
        assert len(found) >= 2, (
            f"Dropsite response contains too few lore-specific terms ({found}). "
            "The model may be hallucinating or the retrieval returned wrong chunks."
        )

    def test_fulgrim_response_not_hallucinated(self):
        _stage("LORE — Fulgrim response must not be pure hallucination")
        r = httpx.post(f"{BASE}/query", json=QUERY_FULGRIM, timeout=T_INFERENCE)
        data = _assert_ok(r, "POST /query [lore: Fulgrim]")
        response_lower = data["response"].lower()
        print(f"  response[:300]: {data['response'][:300]}")

        lore_terms = ["fulgrim", "slaanesh", "chaos", "emperor's children", "corruption"]
        found = [t for t in lore_terms if t in response_lower]
        print(f"  lore terms found: {found}")
        assert len(found) >= 2, (
            f"Fulgrim response lacks expected lore terms ({found}). "
            "Check that Fulgrim / Emperor's Children books are embedded in the index."
        )