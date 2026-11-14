# OmnissiahCore

**A from-scratch RAG backend for a 2,350-volume Warhammer 40K corpus.**

OmnissiahCore is the AI service behind **TheForge**, a two-service lore engine that turns a private library of Warhammer 40,000 and Horus Heresy books into grounded, streamed answers. It retrieves evidence from **498,850 searchable passages**, reranks it with a cross-encoder, stitches neighbouring chunks back into continuous scenes, and streams the answer token by token with live retrieval stages.

The project is deliberately built without LangChain or LlamaIndex. The retrieval pipeline, prompt construction, streaming protocol, scene archive, and deployment workflow are first-party Python. The same codebase runs on a local GPU laptop and on Oracle Cloud CPU through profile-driven configuration.

---

## At a glance

| | |
|---|---|
| **Corpus** | 2,245 catalogued books; 2,350 unique source volumes; 498,850 searchable passages |
| **Scene archive** | 6,180 battle scenes in SQLite plus a dedicated FAISS index |
| **Embeddings** | `BAAI/bge-m3`, 1,024 dimensions |
| **Dense retrieval** | FAISS inner-product index |
| **Sparse retrieval** | BM25 (`bm25s`), optional |
| **Fusion** | Reciprocal Rank Fusion |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Generation** | LM Studio OpenAI-compatible server locally; Gemini on Oracle |
| **API** | FastAPI, Server-Sent Events, stage-aware streaming |
| **Frontend** | TheForge — ASP.NET Core 8 / Blazor Server (separate repository) |
| **Deployment** | Local loopback, LAN testing, Oracle systemd with backup/rollback |
| **Tests** | 129 passed, 10 skipped in the non-LangGraph suite; GitHub Actions on Windows and Linux |

The manifest records 2,245 catalogued books while the searchable metadata contains 2,350 unique source volumes; the difference reflects works held in more than one source format.

---

## What this project demonstrates

This is not a notebook prototype. It is a complete retrieval and serving system with production-shaped boundaries:

- **Hybrid retrieval** — dense semantic search and exact keyword search are merged with RRF so a query can match both meaning and rare names.
- **Two-stage ranking** — a cross-encoder reads each query/passage pair and reorders the fused candidates before they reach the model.
- **Scene stitching** — neighbouring chunks are recombined around each hit so a battle or conversation reads as continuous prose instead of disconnected fragments.
- **Grounded prompting** — source headers, chapter labels, and stitch ranges are carried into the prompt and returned to the client as `__SOURCES__`.
- **Live streaming** — SSE frames expose `embedding`, `retrieving`, `reranking`, `generating`, and `complete` before and during answer generation.
- **Multiple query modes** — Explorer, Narrator, Focus, Ghost, and an evidence-only endpoint.
- **Optional adaptive retrieval** — a feature-gated LangGraph graph can assess coverage and retry retrieval when the evidence is thin.
- **Operational discipline** — profile-driven settings, private data outside Git, `doctor`, `smoke`, tests, a tracked-file credential scan, and an explicit Oracle publish/rollback path.

---

## Architecture

```text
TheForge (ASP.NET Core 8 / Blazor Server)
        │  HTTP + SSE
        ▼
FastAPI application (Api/server.py)
        │
        ├── Runtime service
        │     └── Retriever
        │           ├── BGE-M3 query/document embeddings
        │           ├── FAISS dense search
        │           ├── BM25 sparse search (optional)
        │           ├── Reciprocal Rank Fusion
        │           ├── Cross-encoder reranking
        │           └── Scene stitching
        │
        ├── Agent
        │     ├── LM Studio / OpenAI-compatible endpoint (local)
        │     └── Gemini native API (Oracle)
        │
        └── Scene archive
              ├── Db/battle_scenes.db
              ├── Db/scene_faiss.index
              └── Db/scene_metadata.json
```

The frontend lives in the separate **TheForge** repository. This repository owns retrieval, generation, streaming, scene indexing, and deployment.

---

## Request lifecycle

A streamed Explorer request returns progress before the answer:

```text
POST /query/explore/stream
  data: __STAGE__:embedding
  data: __STAGE__:retrieving
  data: __STAGE__:reranking
  data: __STAGE__:generating
  data: The
  data:  provided
  ...
  data: __SOURCES__:[{"source": "...", "stitch_range": "chunk_id ...", "score": ...}]
  data: __STAGE__:complete
  data: [DONE]
```

A representative local run on an RTX 5050 Laptop GPU (43/43 layers offloaded) processed an 18,586-token prompt and generated 301 tokens at 51.4 tok/s, with GPU utilization reaching 98–99% during generation. Those numbers are machine-specific; they are evidence that the local path is real, not a benchmark guarantee.

---

## Query modes and API surface

| Endpoint | Purpose |
|---|---|
| `GET /health`, `/version`, `/info` | Runtime health, build identity, index metadata |
| `POST /query`, `/query/stream` | General adaptive query |
| `POST /query/explore`, `/query/explore/stream` | Object and artefact analysis |
| `POST /query/narrate`, `/query/narrate/stream` | Scene reconstruction |
| `POST /query/focus`, `/query/focus/stream` | Narrow, source-limited answer |
| `POST /query/ghost` | Scene-library lookup without an LLM |
| `POST /query/evidence` | Inspect retrieved chunks without generation |
| `POST /query/graph`, `/query/graph/inspect` | Feature-gated LangGraph inspection |
| `GET /books`, `/sources`, `/memory`, `/llm/status` | Catalogue, source, session, and model status |

---

## Tech stack

**AI / retrieval:** Python, BGE-M3, FAISS, BM25 (`bm25s`), Reciprocal Rank Fusion, `sentence-transformers` cross-encoder, ONNX (optional), LangGraph (optional).

**Backend:** FastAPI, Pydantic, Uvicorn, Server-Sent Events, SQLite, `requests`, `numpy`.

**Frontend (separate repo):** ASP.NET Core 8, Blazor Server, C#.

**Operations:** Git, GitHub Actions, pytest, tracked-file credential scanning, Oracle Cloud, systemd, SSH publish/rollback.

---

## Repository layout

| Path | Purpose |
|---|---|
| `Api/` | FastAPI composition root, routes, runtime service |
| `Core/` | Retrieval, prompting, generation, providers, graphs, configuration |
| `Scripts/` | Corpus build, scene index build, database verification, CLI |
| `Tests/` | Unit, provider, release, and integration tests |
| `tools/` | Evaluation, smoke checks, deployment, documentation verification |
| `config/` | Safe example settings; no real credentials |
| `deployments/` | Historical Oracle release snapshots |
| `staging/` | Archived source; not deployed or imported by the API |
| `Db/` | Private indexes and databases; ignored by Git |
| `Data/` | Private source documents; ignored by Git |

`staging/legacy-build/` and `staging/legacy-scene-pipeline/` hold retired scripts for reference. The current build path is `Scripts/build_db.py` plus `Scripts/buildSceneIndex.py`.

---

## Quick start

Install Git, Python 3.10 or 3.12, and OpenSSH. Clone the private repository and create the environment:

```powershell
git clone https://github.com/Onoduchiagozie/OmnissiahCore.git
cd OmnissiahCore
python manage.py init
python manage.py setup
```

On Windows use `.venv\Scripts\python.exe`; on Linux/macOS use `.venv/bin/python`.

Run the dependency-light checks:

```powershell
python manage.py test
python manage.py scan
```

Install the runtime dependencies:

```powershell
python manage.py setup --full --build
```

CPU PyTorch is the default. For a supported GPU, pass the wheel index from <https://pytorch.org/get-started/locally/>:

```powershell
python manage.py setup --full --build --torch-index YOUR_OFFICIAL_PYTORCH_INDEX
```

FAISS runs on CPU. Embeddings and reranking can use a supported GPU, but a machine having a GPU does not guarantee CUDA support.

Start the local API:

```powershell
python manage.py doctor
python manage.py api --reload
```

Open <http://127.0.0.1:8000/docs>. Local startup binds to loopback only and never starts a public tunnel.

To test from another device on the same trusted network, bind all interfaces explicitly:

```powershell
python manage.py api --host 0.0.0.0 --port 8000
```

Then use `http://<laptop-ip>:8000`. The query endpoints have no authentication; do not expose them to an untrusted network.

Run the non-generating smoke check and the CLI:

```powershell
python manage.py smoke
python manage.py cli
```

`smoke` checks health and retrieval without spending LLM quota. Passing it is not proof that an answer is accurate. Use `tools/verify_scenes.py` to review returned passages before generation.

---

## Private settings and data

`init` creates `~/.config/omnissiahcore/settings.json` outside the source checkout. Edit that file to select the local model, private paths, and machine settings. The examples under `config/` contain no real credentials.

- `active_profile`: `lenovo_build` for local build/query; `dell_query` for Oracle query.
- `data_root`: parent containing `Db/` and `Data/raw_pdfs/`.
- `profiles`: partial overrides of the safe defaults in `config.json`.
- `environment`: optional strings such as `GEMINI_API_KEY` or `HF_HOME`.
- Process environment variables override the private environment file.
- `OMNISSIAH_PRIVATE_CONFIG` selects a different private JSON file.
- `OMNISSIAH_DATA_DIR` overrides the data root.
- `OMNISSIAH_ACTIVE_PROFILE` selects the profile without editing tracked files.

For local generation, load a model in LM Studio (or another OpenAI-compatible server) and set its identifier and chat URL. The local path does not require a paid OpenAI API. Oracle can use its existing Gemini credentials; the workflow does not enable billing.

Copy matching `Db/faiss.index` and `Db/metadata.json` from an existing installation, or build a new index. Keep the embedding model and dimensions compatible. Optional scene database/index/metadata and book manifests must also be copied to retain those features. Model caches download on first use unless offline mode is enabled.

The source checkout does **not** include books, indexes, model weights, API keys, SSH private keys, or private machine notes. Back those up separately. Removing tracked data now does not erase older Git history.

---

## Build a private index

Use the build profile, install `--build` dependencies, and put documents in the private `Data/raw_pdfs` directory:

```powershell
python manage.py build
```

OCR and archive formats additionally need Tesseract, Poppler, 7-Zip, and/or Calibre on `PATH`. Override unusual locations through `OMNISSIAH_TESSERACT`, `OMNISSIAH_POPPLER_PATH`, `OMNISSIAH_7ZIP`, and `OMNISSIAH_CALIBRE` in private settings. ONNX acceleration is optional. Never run `build` against Oracle's production index.

---

## Testing and verification

```powershell
python manage.py test
python manage.py scan
python manage.py doctor
python manage.py smoke
```

The dependency-light suite covers configuration, provider completion, release packaging, and rollback simulations. It does not run a GPU or the private full corpus. Two graph test modules require the optional LangGraph runtime (`requirements-graph.txt`, `langgraph==1.2.20`); the rest of the suite runs without it.

Current local result:

```text
129 passed, 10 skipped, 112 subtests passed
```

The two graph modules are excluded from that count because LangGraph is not installed in the base environment. Install the optional graph requirements before running them.

GitHub Actions runs the Windows and Linux checks. Those checks do not exercise a GPU or the private corpus; test those separately before merging retrieval-affecting changes.

---

## Edit, review, and merge

Start from a clean, synchronized checkout. Preserve outstanding local work before switching branches.

```powershell
git switch main
git pull --ff-only origin main
git switch -c codex/my-change
```

Edit, then run the relevant tests, `doctor`, and the live local smoke check. Stage specific source files, review the diff, and scan it:

```powershell
git add Core/the_file_you_changed.py
git diff --cached
python manage.py scan
git commit -m "Describe the change"
git push -u origin codex/my-change
```

Open a pull request against `main`, read the diff and checks, merge, then delete the branch. On the laptop:

```powershell
git switch main
git pull --ff-only origin main
git branch -d codex/my-change
git fetch --prune
```

Only `main` is permanent. Machine profiles select behavior; GitHub's `origin` is a repository address, not an Oracle server.

---

## Publish to Oracle

Copy `config/publish.example.json` to `~/.config/omnissiahcore/publish.json` and fill in the SSH host, existing private key path, app root, and service name. Keep it outside Git. The server needs Python, the working app virtual environment, and permission for the SSH user to run `sudo -n systemctl` for this service.

Publishing is explicit, not an automatic side effect of pushing code. This keeps SSH keys on the laptop and avoids storing production secrets in CI.

```powershell
python manage.py status
python manage.py publish
```

`publish` requires clean `main` exactly matching `origin/main`, reruns tests and the tracked-file scan, and uploads only allowlisted runtime source from that commit. Preparation compiles the candidate, checks its configuration against existing data paths, creates a timestamped exact source backup on Oracle, and reports the files that would change. Production keeps running.

After reviewing the prepared commit, apply it explicitly:

```powershell
python manage.py publish --apply FULL_MAIN_COMMIT_SHA
```

Activation refuses changed live source or active/queued generation requests, stops the service, installs the source, records `release.json`, starts the service, and waits for health plus the matching `/version` commit. Failed readiness restores the exact previous source and restarts it. A failure to restart the previous service is surfaced as an error; inspect server logs rather than assuming success.

```powershell
python manage.py status
python manage.py smoke --url YOUR_PUBLIC_API_URL
```

To restore the backup path printed by the deployment:

```powershell
python manage.py rollback --backup /absolute/backup/path
```

Rollback changes the running version. `status` makes this visible; revert the bad change through a new pull request before the next publish so `main` and production converge again. Source deployments do not install packages, rebuild indexes, or migrate data. Dependency and data changes need their own tested preparation.

---

## Known limits and design decisions

- **No query authentication.** Do not expose the API directly to untrusted public users without a gateway.
- **Single LLM slot.** One generation runs at a time; concurrent users wait. Monitor `/llm/status`.
- **Answer accuracy is still under investigation.** Retrieval quality and grounding are evaluated, not assumed. The Horus cause question is a known hard case.
- **Scene index coverage.** Oracle's scene index has historically been incomplete. Verify `Db/scene_faiss.index`; `/query/ghost` can return `matched: false`.
- **BM25 is optional.** If `bm25s` is not installed, the retriever logs a warning and continues with dense retrieval plus reranking.
- **LangGraph is optional.** It is absent from the base runtime and only enabled with `retrieval_cfg["langgraph_retrieval"]` plus the graph requirements.
- **Gemini is Oracle-only in practice.** The local path is self-hosted; the Oracle profile can use Gemini. Health returns the configured URL, so do not infer provider health from that field alone.
- **No LangChain or LlamaIndex.** The retrieval pipeline is custom Python. LangGraph is an optional orchestration layer, not a replacement for the pipeline.
- **Archived source is not deployed.** `staging/` is reference material, not a certified fix or a second runtime.

---

## Why this is a useful portfolio project

It shows the parts of AI engineering that are easy to skip in a demo:

- designing retrieval and ranking as separate, testable stages;
- measuring and streaming each stage to a client;
- keeping private data and credentials out of the repository;
- making the same code run on a GPU laptop and a CPU cloud host;
- shipping with tests, health checks, backups, and rollback instead of a one-way deploy.

TheForge and OmnissiahCore are deliberately separate services with strict data contracts, so the frontend and the AI backend can evolve independently.
