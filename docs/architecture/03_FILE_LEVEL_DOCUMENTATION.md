# File-Level Documentation

## Root Files

### `main.py`

Ownership: process entry dispatcher.

Key symbols:

- `COMMANDS` at `main.py:16`.
- `usage()` at `main.py:24`.
- API `os.execv` branch at `main.py:41`.
- Non-API script `os.execv` branch at `main.py:57`.

Responsibilities:

- Defines the supported command surface: `cli`, `api`, `verify`, `build`.
- Prints usage if the user omits or mistypes the command.
- Replaces the current Python process with the target script/process.

Runtime relationships:

- `api` dispatches to `uvicorn Api.server:app`.
- `cli` dispatches to `Scripts/query_test.py`.
- `verify` dispatches to `Scripts/verify_db.py`.
- `build` dispatches to `Scripts/build_db.py`.

Danger:

- The file labels the project `OmnissiahCoreOld`, but the API metadata says `OmnissiahCore API`.
- API mode always uses `--reload`; in production this creates extra reload processes and should be disabled.

### `config.json`

Ownership: runtime and build configuration.

Key sections:

- Active profile at `config.json:2`.
- `lenovo_build` profile at `config.json:5`.
- Lenovo retrieval settings at `config.json:17`.
- Lenovo Ollama settings at `config.json:27`.
- `dell_query` profile at `config.json:48`.
- Dell retrieval settings at `config.json:60`.
- Dell Ollama settings at `config.json:70`.

Responsibilities:

- Selects machine role, embedding model, retrieval strategy, Ollama model and options, chunking behavior, and data paths.
- Serves both the API query runtime and offline build script.

Danger:

- `Scripts/build_db.py:84` reads `full_config["active_profile"]` directly and does not apply environment overrides from `Core/config_loader.py`.
- Query runtime and build runtime can diverge if environment overrides are used for API but not for build.

### `app_text.json`

Ownership: application text and prompt templates.

Key sections:

- API metadata consumed by `Api/server.py:25`.
- CLI strings consumed by `Scripts/query_test.py:18` and `:56`.
- Agent fallback messages consumed by `Core/agent.py:67` and `:106`.
- Remembrancer prompt at `app_text.json:22`.
- Object explorer prompt at `app_text.json:23`.
- Memory prefix at `app_text.json:28`.
- Narrator prompt at `app_text.json:29`.

Responsibilities:

- Keeps prompt wording and UI strings editable without changing Python code.
- Defines the model behavior contract more strongly than the route names do.

Danger:

- Prompt changes are production behavior changes. They can alter hallucination risk, response format, and integration expectations.

### `requirements.txt`

Ownership: Python dependency declaration.

Runtime dependency groups:

- RAG: `faiss-cpu`, `sentence-transformers`, `numpy`, `torch`.
- Sparse search: `rank-bm25`.
- Ingestion: `pypdf`, `python-docx`, `beautifulsoup4`, `lxml`.
- API: `fastapi`, `uvicorn[standard]`, `pydantic`.
- HTTP: `requests`, `httpx`.
- Utility/test: `tqdm`, `pytest`.

Danger:

- `Scripts/build_db.py` imports additional packages not declared here: `nltk`, `pdf2image`, `pytesseract`, `ebooklib`, `PIL`, `optimum.onnxruntime`, `transformers`, and `onnxruntime`.

## `Api` Package

### `Api/server.py`

Ownership: FastAPI composition root.

Key symbols:

- `create_app()` at `Api/server.py:25`.
- Startup event at `Api/server.py:47`.
- Module-level `app = create_app()` at `Api/server.py:54`.

Responsibilities:

- Creates `FastAPI` app with title, description, version from `Core.app_text`.
- Configures CORS from `OMNISSIAH_CORS_ORIGINS`.
- Calls `runtime_service.startup()` on FastAPI startup.
- Includes system and query routers.

Relationships:

- Imports query router from `Api.routes.query_routes`.
- Imports system router from `Api.routes.system_routes`.
- Imports runtime singleton from `Api.services.runtime_service`.

### `Api/models.py`

Ownership: public API data models.

Key symbols:

- `QueryRequest` at `Api/models.py:6`.
- `SourceInfo` at `Api/models.py:17`.
- `QueryResponse` at `Api/models.py:24`.

Responsibilities:

- Validates request overrides for `top_k`, `candidate_pool`, and `stitching_window`.
- Defines the structured response returned by `/query` and `/query/narrate`.

### `Api/routes/query_routes.py`

Ownership: query-facing HTTP endpoints.

Key symbols:

- `_ollama_pool` at `Api/routes/query_routes.py:12`.
- `_source_list()` at `Api/routes/query_routes.py:16`.
- `query_inspect()` at `Api/routes/query_routes.py:37`.
- `query_sync()` at `Api/routes/query_routes.py:54`.
- `query_stream()` at `Api/routes/query_routes.py:86`.
- `query_narrate()` at `Api/routes/query_routes.py:112`.
- `query_narrate_stream()` at `Api/routes/query_routes.py:160`.
- `query_explore()` at `Api/routes/query_routes.py:227`.

Responsibilities:

- Converts HTTP requests into `RuntimeService` calls.
- Enforces blank-query validation.
- Uses a thread pool for blocking synchronous work.
- Converts chunks into public source metadata.
- Exposes SSE streaming for token-by-token generation.

Danger:

- `RuntimeError` from `ensure_ready()` is not caught.
- `/query/narrate/stream` can send duplicate `[DONE]`.
- `_ollama_pool` is named for Ollama but also runs retrieval and prompt construction.

### `Api/routes/QUERY_ROUTES_IMPORVED.py`

Ownership: duplicate/unused route file.

Evidence:

- It contains the same route symbols and line structure as `Api/routes/query_routes.py`.
- `Api/server.py:13` imports only `Api.routes.query_routes`, not this file.

Responsibility:

- None at runtime unless imported manually.

Danger:

- Future fixes can be applied to one file but not the other. Treat this as dead duplicate code unless intentionally promoted.

### `Api/routes/system_routes.py`

Ownership: health, metadata, and memory HTTP endpoints.

Key symbols:

- `health()` at `Api/routes/system_routes.py:10`.
- `info()` at `Api/routes/system_routes.py:15`.
- `runtime_config()` at `Api/routes/system_routes.py:20`.
- `list_sources()` at `Api/routes/system_routes.py:25`.
- `get_source_chunks()` at `Api/routes/system_routes.py:33`.
- `clear_memory()` at `Api/routes/system_routes.py:41`.
- `get_memory()` at `Api/routes/system_routes.py:46`.

Responsibilities:

- Exposes operational metadata for clients and operators.
- Lets clients inspect indexed sources.
- Lets clients inspect or clear in-memory session state.

### `Api/services/runtime_service.py`

Ownership: API runtime singleton and application service layer.

Key symbols:

- `_sanitize_numpy()` at `Api/services/runtime_service.py:16`.
- `RuntimeService` at `Api/services/runtime_service.py:33`.
- `startup()` at `Api/services/runtime_service.py:44`.
- `run_query()` at `Api/services/runtime_service.py:73`.
- `stream_query_mode()` at `Api/services/runtime_service.py:97`.
- `stream_query()` at `Api/services/runtime_service.py:120`.
- `inspect_query()` at `Api/services/runtime_service.py:143`.
- `runtime_service = RuntimeService()` at `Api/services/runtime_service.py:237`.

Responsibilities:

- Owns one shared `OmnissiahRetriever` instance per API process.
- Loads metadata cache for system/source APIs.
- Creates short-lived `OmnissiahAgent` instances per request.
- Manages process-local session memory.
- Adapts core outputs to API payloads.

Relationships:

- Imports config exports from `Core.config_loader`.
- Imports `OmnissiahRetriever` and `OmnissiahAgent`.
- Imports prompt builder for inspect preview.

Danger:

- Loads metadata twice in memory: once inside retriever and once in `_metadata_cache`.
- Session memory concurrency is not per-session serialized.

## `Core` Package

### `Core/config_loader.py`

Ownership: canonical runtime config loader for API, CLI, verification, and retriever.

Key symbols:

- `BASE_DIR` and `CONFIG_PATH` at `Core/config_loader.py:10`.
- `_fatal()` at `Core/config_loader.py:17`.
- `_apply_env_overrides()` at `Core/config_loader.py:48`.
- `_load_config()` at `Core/config_loader.py:75`.
- Exported `embedding_cfg`, `retrieval_cfg`, `ollama_cfg`, `chunking_cfg`, `machine_role`, `paths`, `cfg`.

Responsibilities:

- Loads `config.json`.
- Selects profile using `OMNISSIAH_ACTIVE_PROFILE` or file default.
- Validates required profile keys.
- Applies environment overrides for machine role, embedding device, Ollama settings, and retrieval settings.
- Builds absolute filesystem paths for FAISS, metadata, manifest, processed files, failure log, and base dir.

Danger:

- Imports have side effects: importing this module loads config and can exit the process.

### `Core/retriever.py`

Ownership: hybrid retrieval engine.

Key symbols:

- `OmnissiahRetriever` at `Core/retriever.py:56`.
- Constructor at `Core/retriever.py:57`.
- `search()` at `Core/retriever.py:118`.
- `inspect()` at `Core/retriever.py:151`.
- `_faiss_search()` at `Core/retriever.py:185`.
- `_bm25_search()` at `Core/retriever.py:210`.
- `_rrf_merge()` at `Core/retriever.py:231`.
- `_apply_query_grounding()` at `Core/retriever.py:261`.
- `_stitch_chunks()` at `Core/retriever.py:286`.
- `_rerank()` at `Core/retriever.py:321`.
- `_check_file()` at `Core/retriever.py:357`.

Responsibilities:

- Validates and loads retrieval artifacts.
- Loads embedding model and optional reranker.
- Builds BM25 in memory from metadata text.
- Performs multi-stage retrieval and returns stitched candidate chunks.

Danger:

- BM25 is built over every metadata text at startup; large corpora increase startup memory and time.
- The builder currently stores minimal chunk metadata, so `chapter` and `file_type` often fall back to defaults.

### `Core/agent.py`

Ownership: RAG orchestration and Ollama client.

Key symbols:

- `OmnissiahAgent` at `Core/agent.py:28`.
- `ask()` at `Core/agent.py:41`.
- `ask_stream()` at `Core/agent.py:88`.
- `_build_prompt()` at `Core/agent.py:141`.
- `_classify_intent()` at `Core/agent.py:150`.
- `_call_ollama_sync()` at `Core/agent.py:162`.
- `_stream_ollama()` at `Core/agent.py:194`.
- `_update_memory()` at `Core/agent.py:235`.
- `memory` property at `Core/agent.py:253`.

Responsibilities:

- Calls retriever with request parameters.
- Selects prompt builder based on mode.
- Prepends short previous-turn memory if present.
- Calls Ollama sync or streaming.
- Emits `__SOURCES__` marker after streaming generation.

Danger:

- `_classify_intent()` result is logged only; it does not alter mode or retrieval behavior.
- Ollama failures are encoded as response text, not raised exceptions.

### `Core/prompt.py`

Ownership: prompt assembly.

Key symbols:

- `_format_context_block()` at `Core/prompt.py:14`.
- `_infer_viewpoint()` at `Core/prompt.py:50`.
- `build_prompt()` at `Core/prompt.py:105`.
- `build_narrate_prompt()` at `Core/prompt.py:113`.
- `build_object_explorer_prompt()` at `Core/prompt.py:125`.
- `format_debug()` at `Core/prompt.py:140`.

Responsibilities:

- Formats retrieved chunks into source-tagged context blocks.
- Adds viewpoint hints for narrator mode.
- Selects prompt templates from `app_text.json`.
- Provides CLI debug formatting.

### `Core/app_text.py`

Ownership: `app_text.json` loader.

Key symbols:

- `_load_app_text()` at `Core/app_text.py:19`.
- `app_text` module export at `Core/app_text.py:29`.

Responsibilities:

- Loads editable prompt/API/CLI text once at import time.
- Exits process if file is missing or malformed.

## `Scripts` Package

### `Scripts/build_db.py`

Ownership: offline ingestion and index build pipeline.

Key symbols:

- `OmnissiahBuilder` at `Scripts/build_db.py:62`.
- `load_config()` at `Scripts/build_db.py:84`.
- `load_state()` at `Scripts/build_db.py:90`.
- `init_models()` at `Scripts/build_db.py:106`.
- `get_embeddings()` at `Scripts/build_db.py:127`.
- `extract_text()` at `Scripts/build_db.py:135`.
- `extract_pdf()` at `Scripts/build_db.py:143`.
- `extract_ocr_turbo()` at `Scripts/build_db.py:153`.
- `extract_epub()` at `Scripts/build_db.py:161`.
- `extract_azw3()` at `Scripts/build_db.py:166`.
- `extract_cbr()` at `Scripts/build_db.py:172`.
- `chunk_text()` at `Scripts/build_db.py:180`.
- `build()` at `Scripts/build_db.py:195`.
- `save()` at `Scripts/build_db.py:233`.

Responsibilities:

- Reads active profile from `config.json`.
- Loads prior processed files, metadata, and FAISS index for incremental builds.
- Uses ONNX GPU path if local `bge_m3_onnx` exists and CUDA provider is available.
- Extracts text from PDF, EPUB, AZW3, CBR, CBZ, and TXT.
- Falls back to OCR for PDFs with too little extracted text.
- Chunks text by NLTK sentence boundaries and configured target token count.
- Embeds chunk text and appends vectors to FAISS.
- Writes metadata, processed-file state, manifest, and failure report.

Danger:

- Hardcoded Windows tool paths at top of file.
- Sets `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`, requiring local model caches.
- Does not assign explicit `chunk_id`, `chapter`, or `file_type` in `chunk_text()`, even though retriever and API expose those fields.
- Does not implement advertised `--retry-failed`.

### `Scripts/query_test.py`

Ownership: local interactive CLI.

Key symbols:

- `_print_banner()` at `Scripts/query_test.py:18`.
- `_list_books()` at `Scripts/query_test.py:35`.
- `main()` at `Scripts/query_test.py:56`.
- `_print_sources()` at `Scripts/query_test.py:165`.

Responsibilities:

- Creates an `OmnissiahAgent` directly, bypassing FastAPI.
- Supports commands for memory clearing, book filter, source listing, explorer/remembrancer modes, verbosity, top-k, stitching window, battle mode, and quit.
- Uses API `/sources` when available, falling back to local metadata.

Danger:

- CLI mode constructs its own retriever and can duplicate memory/model load if API is running separately.

### `Scripts/verify_db.py`

Ownership: DB validation script.

Key symbols:

- `check()` at `Scripts/verify_db.py:35`.
- Top-level verification script body after imports.

Responsibilities:

- Checks required DB artifact existence.
- Loads FAISS and metadata.
- Verifies vector count versus metadata count.
- Loads embedding model to verify dimension.
- Checks chunk ID continuity.
- Prints source distribution and sample chunks.
- Prints failed files and manifest data.

Danger:

- It imports `Core.config_loader`, so config failures terminate verification before any checks.
- It can load the embedding model, which is expensive on low-memory machines.

## `Db`

Ownership: generated retrieval artifacts.

Observed files:

- `Db/faiss.index`: large FAISS vector index.
- `Db/metadata.json`: large metadata/chunk corpus.
- `Db/manifest.json`: build summary showing `total_chunks` 498,850 and `total_files` 2,350.
- `Db/processed_files.json`: incremental build state.
- `Db/failed_files.json` and `Db/failure_report.json`: failed extraction state.

`Db` is ignored by `.gitignore` at `.gitignore:6` and `.gitignore:8`, though files exist locally.

## `Data`

Ownership: raw and failed input documents.

Configured through `config.json` profile `paths`.

Expected folders:

- `Data/raw_pdfs`
- `Data/failed_pdfs`

The current listing returned no files under `Data`.

## `bge_m3_onnx`

Ownership: optional local ONNX embedding acceleration artifacts used by `Scripts/build_db.py`.

Observed files:

- `config.json`
- `model.onnx`
- `model.onnx_data`
- tokenizer files such as `tokenizer.json`, `tokenizer_config.json`, `sentencepiece.bpe.model`, and `special_tokens_map.json`.

Runtime relationship:

- `Scripts/build_db.py:106` checks for this directory.
- `Scripts/build_db.py:110` requires `CUDAExecutionProvider`.
- `Scripts/build_db.py:114` loads `ORTModelForFeatureExtraction`.

The API retriever does not use this ONNX directory; it uses `SentenceTransformer` directly in `Core/retriever.py:79`.

## `Tests/test.py`

Ownership: HTTP integration test harness.

Key symbols:

- `BASE_URL = "http://localhost:8000"`.
- `test_health()` at `Tests/test.py:74`.
- `test_sources()` at `Tests/test.py:113`.
- `test_inspect()` at `Tests/test.py:169`.
- `test_query_narrator()` at `Tests/test.py:249`.
- `test_query_stream_narrator()` at `Tests/test.py:346`.
- `main()` at `Tests/test.py:445`.

Responsibilities:

- Exercises health, source listing, inspect, sync query, and stream query.
- Requires an already running API server.

Danger:

- File header says `python Tests/test_comprehensive.py`, but actual file is `Tests/test.py`.
- `test_query_narrator()` posts to `/query` with `"mode": "narrator"`, but the backend ignores that field and uses remembrancer mode.
- `test_query_stream_narrator()` posts to `/query/stream`, not `/query/narrate/stream`.
