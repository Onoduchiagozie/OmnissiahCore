# OmnissiahCore — Complete System Documentation
### Warhammer 40K Lore RAG Engine v2.0

---

## Table of Contents
1. [What This System Is](#1-what-this-system-is)
2. [Project Structure](#2-project-structure)
3. [Machine Roles — Lenovo vs Dell](#3-machine-roles--lenovo-vs-dell)
4. [Step-by-Step System Flow](#4-step-by-step-system-flow)
5. [Configuration Reference](#5-configuration-reference)
6. [Context Depth Control](#6-context-depth-control)
7. [FastAPI vs Django — Why FastAPI Wins Here](#7-fastapi-vs-django--why-fastapi-wins-here)
8. [ML.NET Integration — Now and at Scale](#8-mlnet-integration--now-and-at-scale)
9. [AnythingLLM Integration](#9-anythingllm-integration)
10. [Git Setup — CMD Instructions](#10-git-setup--cmd-instructions)
11. [Running the System](#11-running-the-system)
12. [Performance Tuning](#12-performance-tuning)
13. [Multi-Format Extension Guide](#13-multi-format-extension-guide)
14. [Interview Talking Points](#14-interview-talking-points)

---

## 1. What This System Is

OmnissiahCore is a **production-grade Retrieval-Augmented Generation (RAG) system** built specifically for querying a large corpus of Warhammer 40K lore across 2,000+ books, codexes, and short stories.

**In plain terms:**
- You have 295,000+ text chunks extracted and embedded from lore PDFs.
- When you ask a question, the system finds the most relevant chunks using two strategies (semantic + keyword).
- It stitches neighbouring chunks together to preserve narrative continuity.
- It feeds that context into a locally-running LLM (via Ollama) and generates an immersive response.

**What makes it different from a generic RAG chatbot:**
- It reconstructs full battle scenes (Horus vs Emperor, Ferrus vs Fulgrim).
- It merges fragments from multiple books into one coherent narrative.
- It runs entirely locally — no OpenAI, no cloud, no data leaving your machines.

---

## 2. Project Structure

```
OmnissiahCore/
│
├── config.json              ← MASTER CONFIG — all settings live here
│
├── Core/
│   ├── config_loader.py     ← Loads config.json, exported to all modules
│   ├── retriever.py         ← FAISS + BM25 + chunk stitching + reranker
│   ├── agent.py             ← Full pipeline: retrieve → prompt → LLM
│   └── prompt.py            ← Remembrancer & Object Explorer prompts
│
├── Api/
│   └── server.py            ← FastAPI REST + streaming SSE endpoints
│
├── Scripts/
│   ├── build_db.py          ← Lenovo ONLY — ingests PDFs, builds FAISS
│   ├── query_test.py        ← Interactive CLI, works on both machines
│   └── verify_db.py         ← Health check — safe on Dell
│
├── Db/
│   ├── faiss.index          ← NOT in Git — transfer manually
│   ├── metadata.json        ← NOT in Git — transfer manually
│   ├── manifest.json        ← Index stats and build config
│   ├── processed_files.json ← Which files have been ingested
│   └── failed_files.json    ← Files that failed extraction
│
├── Data/
│   ├── raw_pdfs/            ← Your lore PDFs — NOT in Git
│   └── failed_pdfs/         ← Auto-moved failed files
│
├── requirements.txt
├── .gitignore
└── docs/
    └── README.md            ← This file
```

---

## 3. Machine Roles — Lenovo vs Dell

| Task | Lenovo (48GB) | Dell (8GB) |
|---|---|---|
| Build FAISS index | ✅ Yes | ❌ Never |
| Re-embed documents | ✅ Yes | ❌ Never |
| Query the index | ✅ Yes | ✅ Yes |
| Run Ollama LLM | ✅ Yes | ✅ Yes |
| Cross-encoder reranker | ✅ Enabled | ❌ Disabled (RAM) |
| BM25 search | ✅ Yes | ✅ Yes |

**How Dell gets its data:**
1. Lenovo builds `Db/faiss.index` and `Db/metadata.json`.
2. You copy those two files to Dell via USB drive or local network.
3. Dell pulls all code via Git.
4. On Dell: set `"active_profile": "dell_query"` in `config.json`.
5. Run `python Scripts/verify_db.py` to confirm everything is healthy.

**Files never committed to Git:**
- `Db/faiss.index` (~1-4GB binary file)
- `Db/metadata.json` (~500MB+ JSON)
- `Data/raw_pdfs/` (your books)

---

## 4. Step-by-Step System Flow

### The full query pipeline — teacher level explanation

```
User types: "Describe the duel between Ferrus Manus and Fulgrim"
```

**Step 1 — Query Enrichment**
The agent prepends a BGE-M3 instruction prefix to the query:
```
"Represent this sentence for searching relevant passages: Describe the duel..."
```
This is required because BGE-M3 was trained with this prefix for retrieval tasks.
The same prefix must be used consistently between build and query.

**Step 2 — FAISS Dense Search**
Your query is converted to a 1024-dimensional vector by the embedding model.
FAISS performs a dot-product similarity search across all 295,537 stored vectors.
It returns the top `candidate_pool` (e.g. 30) most semantically similar chunks.
This catches meaning-based matches: "primarch's blade" matches "Fulgrim's sword".

**Step 3 — BM25 Sparse Search**
The same query is tokenised into words: ["duel", "ferrus", "manus", "fulgrim"]
BM25 searches for documents containing those exact terms with TF-IDF weighting.
This catches exact matches that embeddings might miss: character names, specific places.
Returns top 30 candidates independently.

**Step 4 — Reciprocal Rank Fusion (RRF) Merge**
Both candidate lists are merged using RRF:
```
score(chunk) = 1/(60 + faiss_rank) + 1/(60 + bm25_rank)
```
Chunks appearing high in BOTH lists get boosted significantly.
Duplicates are detected by MD5 hash of the text content.
Result: a single ranked list of the best 30 candidates.

**Step 5 — Chunk Stitching**
This is the most important step for narrative continuity.
For each top chunk retrieved, we look up its `chunk_id` and expand outward:
```
chunk_id 4821  ← the retrieved hit
chunk_id 4819  ← 2 before (context: the betrayal begins)
chunk_id 4820  ← 1 before (context: Fulgrim raises his blade)
chunk_id 4822  ← 1 after  (context: Ferrus falls)
chunk_id 4823  ← 2 after  (context: the aftermath)
```
These are joined in order to produce one continuous passage.
Without stitching, the model gets fragments. With stitching, it gets the scene.

**Step 6 — Cross-Encoder Rerank (Lenovo only)**
On Lenovo, the merged + stitched chunks are re-scored using a CrossEncoder.
Unlike the embedding model, the CrossEncoder reads query + document *together*,
scoring their relevance with full attention. This is more accurate but slower.
On Dell (8GB), this step is skipped — FAISS + BM25 is sufficient.

**Step 7 — Prompt Construction**
The Remembrancer system prompt is assembled with all stitched passages inserted.
The prompt instructs the model to narrate the scene as a witness, never summarise,
never use bullet points, and never hallucinate beyond the provided context.

**Step 8 — Ollama LLM Call**
The system + user messages are sent to Ollama via HTTP POST.
Streaming mode sends tokens as they're generated (SSE format).
The agent accumulates tokens and returns the final response.

**Step 9 — Response + Sources**
The user receives:
- The Remembrancer's narration (the full answer)
- The source list: book name, chapter, chunk_id range, score

---

## 5. Configuration Reference

All settings live in `config.json`. Switch profiles by changing `active_profile`.

### Lenovo profile key settings:
```json
"candidate_pool": 30     ← how many raw hits to retrieve per search
"top_k": 8               ← how many chunks reach the LLM
"stitching_window": 2    ← ±2 neighbouring chunks added per hit
"use_reranker": true     ← CrossEncoder enabled
"num_ctx": 8192          ← Ollama context window
```

### Dell profile key settings:
```json
"candidate_pool": 20     ← smaller pool, less RAM
"top_k": 6               ← fewer chunks, less context window usage
"stitching_window": 2    ← same stitching, different machine
"use_reranker": false    ← disabled for memory
"num_ctx": 6144          ← smaller context window
```

---

## 6. Context Depth Control

Three parameters control how much lore context the LLM receives:

| Parameter | Effect | Trade-off |
|---|---|---|
| `top_k` | How many chunks reach the LLM | Higher = broader coverage, more context window |
| `stitching_window` | Neighbouring chunks added per hit | Higher = deeper narrative, more tokens |
| `candidate_pool` | Initial retrieval size | Higher = better recall, same final output size |

**For the Horus vs Emperor fight:**
Use `/battle Horus Emperor Throne Room` in CLI, which sets:
- `top_k=12`, `candidate_pool=40`, `stitching_window=3`
This gives ~36 stitched chunks covering the full confrontation from multiple books.

**For exact quote retrieval:**
BM25 handles this. If you remember a specific phrase, BM25 will surface it even if
the embedding model misses it. The RRF merge ensures exact matches are promoted.

---

## 7. FastAPI vs Django — Why FastAPI Wins Here

This is a common interview question. Here is the honest comparison for an ML/RAG system:

### FastAPI Advantages (why we chose it)

| Feature | FastAPI | Django |
|---|---|---|
| **Streaming (SSE/WebSocket)** | Native, first-class | Requires channels + Daphne setup |
| **Async performance** | Built on Starlette + asyncio | Sync-first, async bolted on |
| **Speed** | One of the fastest Python frameworks | Significantly slower in benchmarks |
| **ML/AI integration** | Zero friction with torch, numpy, generators | ORM-focused, ML is an afterthought |
| **Auto-generated API docs** | Swagger + ReDoc built in (`/docs`) | Needs django-rest-framework + drf-spectacular |
| **Type safety** | Pydantic validation on every request | Manual serializers |
| **Startup overhead** | Minimal | Heavy ORM, migrations, admin loading |
| **Learning curve** | Flat — pure Python + type hints | Steeper — ORM, migrations, settings |
| **Token streaming** | `StreamingResponse` + generators | Complex to implement correctly |

### When Django would be better:
- You need a full admin panel for non-technical users.
- You are building a multi-user app with complex database relationships.
- You need Django ORM's migration system for relational data.

### The killer reason for our project:
Streaming LLM responses require the server to send tokens as they arrive from Ollama.
In FastAPI: `return StreamingResponse(generator, media_type="text/event-stream")`
In Django: this requires Django Channels, Redis, Daphne, and significant extra config.
FastAPI does it in one line. For an AI backend, this is decisive.

---

## 8. ML.NET Integration — Now and at Scale

ML.NET is Microsoft's machine learning framework for .NET (C#/F#). Here is how
it fits into OmnissiahCore at different stages.

### Right Now — Potential Uses

**Option A: C# CLI client calling the FastAPI backend**
If you want a Windows-native UI or .NET integration:
```csharp
// OmnissiahClient.cs
var client = new HttpClient();
var payload = new { query = "Who is Horus Lupercal?", top_k = 6 };
var json = JsonSerializer.Serialize(payload);
var content = new StringContent(json, Encoding.UTF8, "application/json");
var response = await client.PostAsync("http://localhost:8000/query", content);
var result = await response.Content.ReadAsStringAsync();
```
This requires zero changes to the Python backend.

**Option B: ML.NET Text Featurisation for BM25 preprocessing**
ML.NET has `Microsoft.ML.Text` with `NormalizeText`, `TokenizeIntoWords`, and
`RemoveDefaultStopWords` transforms. You could preprocess text on the build machine
in C# before passing chunks to the Python embedder.

**Option C: ML.NET Sentiment + Intent Classification**
Instead of the current Python keyword-based intent classifier in `agent.py`,
train an ML.NET `TextClassificationTrainer` on a small labeled set of lore queries.
Export the model, wrap it in a REST endpoint, and call it from the Python agent.

### At Scale — Where ML.NET Becomes Powerful

| Scenario | ML.NET Role |
|---|---|
| **High-traffic production** | Deploy ML.NET as the primary API layer; call Python retriever as microservice |
| **Windows/Azure deployment** | ML.NET integrates natively with Azure ML, Functions, and Service Bus |
| **Real-time reranking** | Train a custom ML.NET ranker on user feedback data (clicks, ratings) |
| **Query classification** | Replace the keyword classifier with a proper ML.NET text classifier |
| **Named Entity Recognition** | Use ML.NET + custom NER to extract character names, battles, places from queries |
| **A/B testing** | ML.NET Model Builder makes it easy to experiment with different rankers |
| **Blazor Server integration** | Pair with your Argus/Blazor experience to build a full .NET lore dashboard |

### Architecture at Scale with ML.NET:
```
User (Blazor UI)
    ↓
ASP.NET Core API Gateway (C#)
    ↓ calls
ML.NET Intent Classifier → Python FastAPI RAG Backend
                                    ↓
                             FAISS + BM25 + Ollama
                                    ↓
                             ML.NET Reranker (trained on feedback)
                                    ↓
                             Response to user
```

This is the architecture you could present in an interview as a multi-language,
production ML system. Your Blazor + .NET background from Argus makes this natural.

---

## 9. AnythingLLM Integration

### Option A — Custom Backend (FastAPI) + AnythingLLM as UI

This is the recommended approach. You keep full control of retrieval quality.

**Setup:**
1. Start the FastAPI backend: `uvicorn Api.server:app --port 8000`
2. In AnythingLLM: Settings → LLM Provider → Custom
3. Set API endpoint to `http://localhost:8000/query/stream`
4. AnythingLLM sends `{"query": "..."}` and receives SSE tokens
5. Ollama is shared: both systems point to `http://localhost:11434`

**Data flow:**
```
AnythingLLM UI → POST /query/stream → FastAPI → FAISS+BM25 → Ollama → SSE tokens → UI
```

**Object Explorer with AnythingLLM:**
- Point AnythingLLM to `POST /query/explore`
- This activates the technical analyst persona
- For images/visual descriptions ingested by AnythingLLM, add them to the metadata

### Option B — AnythingLLM as Full Replacement

Useful for processing the failed PDFs that have image-heavy content
(the codexes and graphic novels that pypdf can't extract text from).

**How AnythingLLM handles failed PDFs:**
- AnythingLLM uses OCR (via Tesseract) to extract text from image-based PDFs
- It generates text descriptions of images and diagrams
- These descriptions enter AnythingLLM's own vector store

**To use both systems together:**
1. Run AnythingLLM for the 138 failed files (codexes, graphic novels)
2. Export AnythingLLM's extracted chunks as JSON
3. Add those chunks to your existing `metadata.json` and re-embed into FAISS
4. Result: FAISS covers both text PDFs and previously-failed image PDFs

---

## 10. Git Setup — CMD Instructions

### First-time setup on Lenovo (project origin)

```cmd
:: Open CMD in the OmnissiahCore project folder

:: 1. Initialise git
git init

:: 2. Stage all files (respects .gitignore — large files excluded)
git add .

:: 3. First commit
git commit -m "Initial commit: OmnissiahCore v2.0 production system"

:: 4. Create repo on GitHub (do this in browser: github.com → New repository)
::    Name it: OmnissiahCore   |   Private   |   No README (we have one)

:: 5. Connect local repo to GitHub
git remote add origin https://github.com/YOUR_USERNAME/OmnissiahCore.git

:: 6. Push
git branch -M main
git push -u origin main
```

### Clone on Dell

```cmd
:: On the Dell machine:
cd C:\Users\YourName\Projects

git clone https://github.com/YOUR_USERNAME/OmnissiahCore.git
cd OmnissiahCore

:: Create virtual environment
python -m venv venv
venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt

:: IMPORTANT: Copy the index files from Lenovo (USB or network share)
:: Copy: Db\faiss.index  →  OmnissiahCore\Db\faiss.index
:: Copy: Db\metadata.json → OmnissiahCore\Db\metadata.json

:: Switch to Dell profile
:: Edit config.json: "active_profile": "dell_query"

:: Verify
python Scripts/verify_db.py
```

### Keeping Dell in sync with Lenovo code changes

```cmd
:: On Dell (pulls latest code — does NOT touch Db/ files per .gitignore)
git pull origin main

:: If Lenovo built a new index, copy new Db/ files manually then:
python Scripts/verify_db.py
```

### Pushing code changes from Lenovo

```cmd
git add .
git commit -m "Description of what changed"
git push origin main
```

### Useful Git commands

```cmd
:: See what's changed
git status

:: See commit history
git log --oneline

:: Undo last commit (keep file changes)
git reset HEAD~1

:: See what's being ignored
git check-ignore -v Db/faiss.index

:: Create a branch for new features
git checkout -b feature/streaming-improvements
git push -u origin feature/streaming-improvements
```

---

## 11. Running the System

### Lenovo — Build and Query

```cmd
:: Activate venv
venv\Scripts\activate

:: Build / update the index
python Scripts/build_db.py

:: Retry failed files
python Scripts/build_db.py --retry-failed

:: Test with CLI
python Scripts/query_test.py

:: Start FastAPI server
uvicorn Api.server:app --host 0.0.0.0 --port 8000

:: Open API docs in browser
:: http://localhost:8000/docs
```

### Dell — Query Only

```cmd
:: Activate venv
venv\Scripts\activate

:: Verify index is healthy
python Scripts/verify_db.py

:: CLI query
python Scripts/query_test.py

:: Start FastAPI server
uvicorn Api.server:app --host 0.0.0.0 --port 8000
```

### Ollama (both machines — run before anything else)

```cmd
:: Start Ollama in the background
ollama serve

:: Pull a model (if not already pulled)
ollama pull llama3

:: For better lore responses on Lenovo:
ollama pull llama3:70b   ← if you have the VRAM
```

---

## 12. Performance Tuning

### Lenovo (48GB RAM, likely GPU)

- Set `candidate_pool: 30`, `top_k: 8`, `stitching_window: 2`
- Enable reranker: `"use_reranker": true`
- Use `llama3:70b` or `mistral` for better responses
- For the Horus/Emperor battle: use `/battle` command (sets top_k=12, window=3)

### Dell (8GB RAM, CPU only)

- Set `candidate_pool: 20`, `top_k: 6`, `stitching_window: 2`
- Disable reranker: `"use_reranker": false`
- Use `llama3:8b` or `phi3:mini` — they fit in 8GB
- Set `"num_ctx": 4096` if responses are slow (less context = faster)
- BGE-M3 loads to ~1.2GB RAM on CPU — acceptable

### Memory breakdown on Dell (8GB):

```
OS + background:          ~1.5 GB
BGE-M3 embedding model:   ~1.2 GB
metadata.json (in RAM):   ~0.5 GB
BM25 index (in RAM):      ~0.3 GB
Ollama (llama3:8b):       ~4.5 GB
────────────────────────────────
Total (approx):           ~8.0 GB  ← tight but workable
```

If you hit memory pressure: disable BM25 (`"use_bm25": false`) to save ~300MB.

---

## 13. Multi-Format Extension Guide

To add a new file format to the build pipeline, add a loader to `Scripts/build_db.py`:

```python
# Example: Adding EPUB support
def load_epub(filepath: str) -> str | None:
    """Requires: pip install ebooklib beautifulsoup4"""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        book  = epub.read_epub(filepath)
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "lxml")
            parts.append(soup.get_text())
        text = "\n".join(parts).strip()
        return text if len(text) > 100 else None
    except Exception as e:
        print(f"  EPUB error: {e}")
        return None
```

Then add `".epub": load_epub` to the `loaders` dict in `load_file()` and
add `".epub"` to the `SUPPORTED_EXT` set. Zero changes to any other file.

---

## 14. Interview Talking Points

**"Tell me about this RAG project."**
> "I built a local RAG system over 295,000 text chunks from 2,000+ Warhammer lore books.
> The retrieval uses hybrid search — FAISS for semantic similarity and BM25 for exact
> keyword matching — merged with Reciprocal Rank Fusion. I implemented chunk stitching
> to preserve narrative continuity, which is critical for reconstructing full battle scenes.
> The backend is FastAPI with Server-Sent Events for real-time token streaming.
> The system runs entirely locally using Ollama, split across two machines: a build machine
> for indexing and a low-memory query machine. All config is centralised with no hardcoding."

**"Why not LangChain or LlamaIndex?"**
> "I built the retrieval pipeline from scratch so I could implement chunk stitching —
> pulling neighbouring chunks by chunk_id for narrative continuity. Existing frameworks
> don't support this well out of the box. It also gave me a deep understanding of every
> component: what FAISS actually does, how RRF scoring works, why BM25 complements
> dense retrieval. That understanding is more valuable than knowing a framework's API."

**"Can this already do what an LLM can do natively?"**
> "For general knowledge — yes, a large model often knows the answer. But for specific
> lore details from 300+ books, models hallucinate confidently. RAG grounds every answer
> in source text. The model can't invent a battle that isn't in the retrieved passages.
> That's the core value proposition: accuracy over fluency."
