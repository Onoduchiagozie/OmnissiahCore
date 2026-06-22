# API And Service Contracts

## Base Service

Default API process:

```text
python main.py api
```

`main.py:41` starts Uvicorn with:

```text
Api.server:app
host 0.0.0.0
port 8000
reload enabled
```

FastAPI OpenAPI docs are available because `Api/server.py:25` sets `docs_url="/docs"` and `redoc_url="/redoc"`.

## Request Models

### `QueryRequest`

Defined at `Api/models.py:6`.

```json
{
  "query": "string, required",
  "session_id": "string, optional, default: default",
  "book_filter": "string or null, optional",
  "source_filter": ["string", "optional exact source names"],
  "top_k": "integer 1..20 or null",
  "candidate_pool": "integer 1..100 or null",
  "stitching_window": "integer 0..6 or null"
}
```

Field behavior:

- `query` is validated by route code for non-empty text.
- `session_id` is used by `RuntimeService` as the key for process-local memory.
- `book_filter` performs case-insensitive substring matching against chunk `source`.
- `source_filter` performs exact source-name matching.
- `top_k`, `candidate_pool`, and `stitching_window` override `retrieval_cfg` for the request.

The model has no `mode` field. To select behavior, call the mode-specific route.

### `SourceInfo`

Defined at `Api/models.py:17`.

```json
{
  "source": "string",
  "chapter": "string",
  "stitch_range": "string",
  "score": "number"
}
```

`Api/routes/query_routes.py:16` computes `score` by choosing the first available value from `rerank_score`, `query_overlap_score`, `rrf_score`, `faiss_score`, then rounding to four decimals.

### `QueryResponse`

Defined at `Api/models.py:24`.

```json
{
  "query": "string",
  "response": "string",
  "sources": [
    {
      "source": "string",
      "chapter": "string",
      "stitch_range": "string",
      "score": 0.0
    }
  ],
  "chunks_used": 0
}
```

## Query Endpoints

### `POST /query`

Registered at `Api/routes/query_routes.py:53`.

Purpose: standard grounded remembrancer answer.

Runtime path:

```text
query_sync()
  -> runtime_service.ensure_ready()
  -> runtime_service.run_query(req, mode="remembrancer", stream=False)
  -> OmnissiahAgent.ask()
  -> OmnissiahRetriever.search()
  -> build_prompt()
  -> Ollama /api/chat stream=false
```

Example request:

```json
{
  "query": "Who was Ferrus Manus?",
  "session_id": "integration-user-42",
  "top_k": 6,
  "candidate_pool": 30,
  "stitching_window": 3
}
```

Example response shape:

```json
{
  "query": "Who was Ferrus Manus?",
  "response": "Generated grounded answer text.",
  "sources": [
    {
      "source": "BookOrFile.pdf",
      "chapter": "unknown",
      "stitch_range": "chunk_id 100-106",
      "score": 3.0
    }
  ],
  "chunks_used": 1
}
```

Error behavior:

- Empty query: HTTP 400 from `Api/routes/query_routes.py:64`.
- Ollama connection/timeout/generic failures: HTTP 200 with a response string beginning `[ERROR]`, because `Core/agent.py:181`, `:183`, and `:185` return error text rather than raising.
- Runtime not ready: `RuntimeService.ensure_ready()` at `Api/services/runtime_service.py:69` raises `RuntimeError`; route does not catch it.

### `POST /query/narrate`

Registered at `Api/routes/query_routes.py:111`.

Purpose: long-form scene reconstruction. The route is identical to `/query` except it passes `mode="narrator"` at `Api/routes/query_routes.py:144`, causing `Core/agent.py:141` to call `build_narrate_prompt()` at `Core/prompt.py:113`.

Recommended request shape:

```json
{
  "query": "Narrate the confrontation between Ferrus Manus and Fulgrim",
  "session_id": "story-session-1",
  "top_k": 15,
  "candidate_pool": 80,
  "stitching_window": 6
}
```

Response model is `QueryResponse`.

### `POST /query/explore`

Registered at `Api/routes/query_routes.py:226`.

Purpose: object, weapon, vehicle, relic, or artefact analysis. This route uses `mode="explorer"` at `Api/routes/query_routes.py:250`, causing `Core/agent.py:141` to call `build_object_explorer_prompt()` at `Core/prompt.py:125`.

Response shape is manually constructed rather than annotated as `QueryResponse`, but fields match the same practical contract:

```json
{
  "query": "Describe the laer blade",
  "response": "Generated object analysis.",
  "chunks_used": 6,
  "sources": [
    {
      "source": "string",
      "chapter": "string",
      "stitch_range": "string",
      "score": 0.0
    }
  ]
}
```

### `POST /query/inspect`

Registered at `Api/routes/query_routes.py:36`.

Purpose: retrieval debugging without LLM generation.

Runtime path:

```text
query_inspect()
  -> runtime_service.inspect_query()
  -> OmnissiahRetriever.inspect()
  -> sanitize numpy values
  -> build_prompt() preview
```

Response shape:

```json
{
  "inspection": {
    "query": "string",
    "query_terms": ["term"],
    "faiss_hits": ["top five raw dense hit objects"],
    "bm25_hits": ["top five raw sparse hit objects"],
    "grounded_hits": ["grounded top hits"],
    "stitched_hits": ["final stitched chunk objects"]
  },
  "prompt_preview": {
    "system_prompt": "first 4000 characters",
    "user_message": "string"
  }
}
```

Use this endpoint in integration tests before expensive generation.

## Streaming Endpoints

### `POST /query/stream`

Registered at `Api/routes/query_routes.py:85`.

Purpose: standard remembrancer generation over Server-Sent Events.

Content type:

```text
text/event-stream
```

Frame contract:

```text
data: token text

data: token text

data: __SOURCES__:[{"source":"...","chapter":"...","stitch_range":"...","score":0.0}]

data: [DONE]

```

Implementation details:

- `Api/routes/query_routes.py:99` returns `StreamingResponse(runtime_service.stream_query(req), ...)`.
- `Api/services/runtime_service.py:120` uses remembrancer mode only.
- `Core/agent.py:88` retrieves and starts streaming.
- `Core/agent.py:131` yields a `__SOURCES__:` marker after Ollama completes.

### `POST /query/narrate/stream`

Registered at `Api/routes/query_routes.py:159`.

Purpose: narrator generation over SSE.

Implementation difference:

- The route creates a copy of the request with a fallback session ID at `Api/routes/query_routes.py:173`.
- It creates an async generator and queue at `Api/routes/query_routes.py:176`.
- It starts a daemon producer thread at `Api/routes/query_routes.py:196`.
- The thread calls `runtime_service.stream_query_mode(req_narrator, mode="narrator")` at `Api/routes/query_routes.py:183`.

Client caution: `RuntimeService.stream_query_mode()` yields `[DONE]` at `Api/services/runtime_service.py:117`, and `_stream()` also yields `[DONE]` at `Api/routes/query_routes.py:211`. Robust clients should treat the first `[DONE]` as terminal and ignore duplicate completion frames.

## System Endpoints

### `GET /health`

Registered at `Api/routes/system_routes.py:9`.

Returns `RuntimeService.health_payload()` from `Api/services/runtime_service.py:162`.

Response shape:

```json
{
  "status": "online",
  "active_profile": "lenovo_build",
  "machine_role": "build",
  "ollama_model": "qwen3:30b-a3b",
  "ollama_url": "http://localhost:11434/api/chat",
  "metadata_loaded": 498850
}
```

This endpoint does not call Ollama. It verifies runtime state, not LLM availability.

### `GET /info`

Registered at `Api/routes/system_routes.py:14`.

Returns `RuntimeService.info_payload()` from `Api/services/runtime_service.py:172`.

Response shape:

```json
{
  "index_vectors": 498850,
  "index_dim": 1024,
  "machine_role": "build",
  "manifest": {
    "build_date": "2026-04-29T18:49:42.062000",
    "total_chunks": 498850,
    "total_files": 2350,
    "model": "BAAI/bge-m3",
    "status": "healthy"
  },
  "retrieval": {
    "use_faiss": true,
    "use_bm25": true,
    "use_reranker": true,
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "candidate_pool": 50,
    "top_k": 10,
    "stitching_window": 5,
    "rrf_k": 60
  },
  "cached_sources": 0
}
```

If FAISS cannot be read, `index_vectors` and `index_dim` become `-1` at `Api/services/runtime_service.py:177`.

### `GET /config/runtime`

Registered at `Api/routes/system_routes.py:19`.

Returns sanitized active runtime config from `Api/services/runtime_service.py:195`.

Use this endpoint for .NET client diagnostics and to display currently active profile behavior.

### `GET /sources`

Registered at `Api/routes/system_routes.py:24`.

Response shape:

```json
{
  "total": 2350,
  "sources": ["file1.pdf", "file2.epub"]
}
```

Uses `_metadata_cache`, not the retriever metadata, through `RuntimeService.list_sources_payload()` at `Api/services/runtime_service.py:215`.

### `GET /sources/{source_name}?limit=20`

Registered at `Api/routes/system_routes.py:32`.

Response shape:

```json
{
  "source": "fulgrim",
  "matched": 20,
  "chunks": [
    {
      "text": "chunk text",
      "source": "Fulgrim.pdf",
      "chapter": "unknown"
    }
  ]
}
```

Matching is case-insensitive substring over `source`, implemented at `Api/services/runtime_service.py:220`. `limit` is constrained to `1..100` by FastAPI `Query` at `Api/routes/system_routes.py:33`.

### `GET /memory?session_id=default`

Registered at `Api/routes/system_routes.py:45`.

Response shape:

```json
{
  "session_id": "default",
  "memory": [
    {
      "query": "previous question",
      "response": "previous response"
    }
  ]
}
```

### `DELETE /memory?session_id=default`

Registered at `Api/routes/system_routes.py:40`.

Response shape:

```json
{
  "status": "Session memory cleared for 'default'."
}
```

## Internal Service Contracts

### `RuntimeService.run_query(req, mode, stream)`

Defined at `Api/services/runtime_service.py:73`.

Input:

- `req`: `QueryRequest`.
- `mode`: one of `remembrancer`, `narrator`, or `explorer`.
- `stream`: boolean passed to `OmnissiahAgent.ask()`.

Output:

```python
tuple[str, list[dict]]
```

Contract:

- Reads session memory before inference.
- Does not hold the lock during Ollama generation.
- Writes the updated memory after inference completes.

### `OmnissiahRetriever.search(...)`

Defined at `Core/retriever.py:118`.

Input:

```python
query: str
top_k: int | None
candidate_pool: int | None
stitching_window: int | None
book_filter: str | None
source_filter: list[str] | None
```

Output chunk fields:

```json
{
  "chunk_id": 123,
  "text": "string",
  "source": "string",
  "chapter": "string",
  "file_type": "pdf",
  "faiss_rank": 0,
  "faiss_score": 0.8,
  "bm25_rank": 0,
  "bm25_score": 12.3,
  "rrf_score": 0.032,
  "query_overlap_terms": ["ferrus"],
  "query_overlap_score": 1.0,
  "stitch_range": "chunk_id 120-126",
  "rerank_score": 4.2
}
```

Not every field is present on every chunk. Downstream code must tolerate missing score fields, which `_source_list()` already does.

### `OmnissiahAgent.ask(...)`

Defined at `Core/agent.py:41`.

Output:

```python
tuple[response_text: str, chunks: list[dict]]
```

If retrieval returns no chunks, returns `app_text["agent"]["no_chunks_message"]` and an empty chunk list at `Core/agent.py:67`.
