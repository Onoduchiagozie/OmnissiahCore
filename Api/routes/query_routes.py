import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from Api.models import QueryRequest, QueryResponse, SourceInfo
from Api.services.runtime_service import runtime_service
from Api.services.stream_contract import validated_sse_frames
from Core.config_loader import paths


router = APIRouter(tags=["Query"])
logger = logging.getLogger(__name__)

_ollama_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ollama")

_book_manifest_cache: list[dict] | None = None


# ---------------------------------------------------------------------------
# GET /books — canonical, deduplicated book list (224 entries)
#
# Deliberately separate from GET /sources, which returns distinct raw
# `source` strings out of metadata.json — that count is inflated because
# the same book can appear under several differently-formatted filenames
# (cross-format duplicates from mixed .mobi/.pdf/.epub ingestion). This
# endpoint returns the hand-curated manifest instead: real book count,
# clean display titles, and the exact filename each title maps to for use
# as book_filter (confirmed to match metadata.json's `source` field
# verbatim for PDF-derived books).
# ---------------------------------------------------------------------------

@router.get("/books")
async def list_canonical_books():
    """
    Returns the canonical book manifest: [{filename, book_order, title}].

    `title` is for display (e.g. a filter dropdown). `filename` is what
    should actually be sent as `book_filter` on a query — it's the value
    that reliably substring-matches the real `source` field used by the
    retriever, unlike the messy/inconsistent title strings.
    """
    global _book_manifest_cache
    if _book_manifest_cache is None:
        manifest_path = paths.get("book_manifest")
        if not manifest_path:
            raise HTTPException(status_code=503, detail="Book manifest path not configured.")
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _book_manifest_cache = data.get("books", [])
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=f"Book manifest not found at {manifest_path}. Place book_manifest.json there.",
            )
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Book manifest is malformed: {e}")

    return {"total": len(_book_manifest_cache), "books": _book_manifest_cache}


def _source_list(chunks: list[dict]) -> list[SourceInfo]:
    return [
        SourceInfo(
            source=c.get("source", "?"),
            chapter=c.get("chapter", "?"),
            stitch_range=c.get("stitch_range", ""),
            score=round(
                c.get("rerank_score")
                or c.get("scene_score")
                or c.get("query_overlap_score")
                or c.get("rrf_score")
                or c.get("faiss_score")
                or 0.0,
                4,
            ),
        )
        for c in chunks
    ]



# ---------------------------------------------------------------------------
# Backward-compatible Remembrancer endpoints used by TheForge's general
# Query and Chat clients. These were absent from the registered router.
# ---------------------------------------------------------------------------

@router.post("/query")
async def query_default(req: QueryRequest):
    """General synchronous RAG answer (legacy Forge API contract)."""
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    loop = asyncio.get_running_loop()
    answer, chunks = await loop.run_in_executor(
        _ollama_pool,
        lambda: runtime_service.run_query(req, mode="remembrancer", stream=False),
    )
    return {
        "query": req.query,
        "response": answer,
        "chunks_used": len(chunks),
        "sources": _source_list(chunks),
    }


@router.post("/query/stream")
async def query_default_stream(req: QueryRequest):
    """General streamed RAG answer with the same SSE contract as Narrator."""
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def _stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _producer():
            try:
                for event in validated_sse_frames(
                    runtime_service.stream_query_mode(req, mode="remembrancer")
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            except Exception:
                logger.exception("FastAPI Remembrancer producer failed")
                asyncio.run_coroutine_threadsafe(
                    queue.put("data: [ERROR] Remembrancer stream failed. See server logs.\n\n"), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_producer, daemon=True).start()
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# POST /query/inspect — retrieval diagnostics, no LLM call
# ---------------------------------------------------------------------------

@router.post("/query/inspect")
async def query_inspect(req: QueryRequest):
    """
    Debug endpoint: shows retrieval results WITHOUT calling the LLM.
    Use this to verify chunk quality before committing to a full generation.
    Returns FAISS hits, BM25 hits, grounded hits, stitched hits, and a
    prompt preview.
    """
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _ollama_pool,
        lambda: runtime_service.inspect_query(req)
    )


# ---------------------------------------------------------------------------
# POST /query/narrate — primary "cast a wide net" experience (SSE)
#
# Reverted back to TRUE broad narrator mode (mode="narrator") — pools
# across however many books are relevant, no source cap, full CrossEncoder
# rerank. This is what it originally did before being temporarily swapped
# to run focus mode internally. Focus now lives exclusively at its own
# dedicated endpoints (/query/focus, /query/focus/stream) so the two modes
# are reachable independently instead of narrate secretly being focus.
# ---------------------------------------------------------------------------

@router.post("/query/narrate")
async def query_narrate_stream(req: QueryRequest):
    """
    Broad narrator mode, streamed via Server-Sent Events.

    Reconstructs events, battles, character arcs, and chronicles as a
    single flowing cinematic narrative, pooling from however many books
    are relevant — no source cap. Use this when you want breadth (a
    faction's whole arc, a "why did X happen" question) rather than one
    specific scene. For a narrowed, accuracy-first single-scene answer,
    use /query/focus or /query/focus/stream instead.

    SSE frame contract:
      data: token text
      data: __SOURCES__:[{...}]
      data: [DONE]
      data: [ERROR] message

    Recommended parameters for deep narratives:
      top_k: 15
      candidate_pool: 80
      stitching_window: 6
    """
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    req_narrator = req.model_copy(
        update={"session_id": req.session_id or "narrate-default"}
    )

    async def _stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _producer():
            try:
                for token in validated_sse_frames(runtime_service.stream_query_mode(
                    req_narrator, mode="narrator"
                )):
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop)
            except Exception:
                logger.exception("FastAPI streaming producer failed")
                err_token = "data: [ERROR] Narrator stream failed. See server logs.\n\n"
                asyncio.run_coroutine_threadsafe(queue.put(err_token), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

        # The stream contract emits [DONE] only after a real answer and
        # upstream completion. Exceptions and empty responses emit [ERROR].

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# POST /query/explore — object and artifact analysis (sync)
# ---------------------------------------------------------------------------

@router.post("/query/explore")
async def query_explore(req: QueryRequest):
    """
    Explorer mode. Synchronous.

    Use for object, weapon, vehicle, relic, and artefact analysis.
    Returns a structured encyclopedic response describing physical properties,
    origin, function, and lore significance.

    Example queries:
      "Describe the laer blade that Fulgrim recovered"
      "What is known about the Anathame blade"
      "Describe the Vengeful Spirit flagship"
    """
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    loop = asyncio.get_event_loop()
    response, chunks = await loop.run_in_executor(
        _ollama_pool,
        lambda: runtime_service.run_query(req, mode="explorer", stream=False),
    )
    return {
        "query": req.query,
        "response": response,
        "chunks_used": len(chunks),
        "sources": [
            {
                "source": c.get("source", "?"),
                "chapter": c.get("chapter", "?"),
                "stitch_range": c.get("stitch_range", ""),
                "score": round(
                    c.get("rerank_score")
                    or c.get("query_overlap_score")
                    or c.get("rrf_score")
                    or c.get("faiss_score")
                    or 0.0,
                    4,
                ),
            }
            for c in chunks
        ],
    }


# ---------------------------------------------------------------------------
# POST /query/focus/stream — narrowed search, streamed via SSE
# ---------------------------------------------------------------------------

@router.post("/query/focus/stream")
async def query_focus_stream(req: QueryRequest):
    """
    Focus mode, streamed. Identical SSE frame contract to /query/narrate —
    same "data: token", "data: __STAGE__:x", "data: __SOURCES__:[...]",
    "data: [DONE]", "data: [ERROR] msg" shape — so a client already wired
    for /query/narrate's stream only needs its target URL changed, nothing
    else (parsing logic, controller, UI all stay identical).

    When the scene library has a confident match, the "stream" is a fake
    typewriter effect over already-finished text (see agent.py) since no
    LLM call happens in that path. When it falls back to a live corpus
    search, tokens are real, generated live, same as narrator.

    Narrows to at most `req.max_sources` (default 2) stitched groups.
    """
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    req_focus = req.model_copy(
        update={"session_id": req.session_id or "focus-default"}
    )

    async def _stream():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _producer():
            try:
                for token in validated_sse_frames(runtime_service.stream_query_mode(
                    req_focus, mode="focus"
                )):
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop)
            except Exception:
                logger.exception("FastAPI streaming producer failed")
                err_token = "data: [ERROR] Focus stream failed. See server logs.\n\n"
                asyncio.run_coroutine_threadsafe(queue.put(err_token), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

        # The stream contract emits [DONE] only after a real answer and
        # upstream completion. Exceptions and empty responses emit [ERROR].

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.post("/query/focus")
async def query_focus(req: QueryRequest):
    """
    Focus mode. Synchronous.

    Narrows the search to at most `req.max_sources` (default 2) stitched
    groups instead of scattering across the whole corpus. Tries the
    pre-woven scene library (battle_scenes.db) first for a fast, already-
    curated match; falls back to a narrowed live corpus search — top_k
    pinned to max_sources, stitching window widened — if nothing in the
    scene library scores confidently.

    Example queries:
      "The destruction of Prospero"
      "The betrayal at Isstvan III"
      "The duel between Loken and Abaddon"
    """
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    loop = asyncio.get_event_loop()
    response, chunks = await loop.run_in_executor(
        _ollama_pool,
        lambda: runtime_service.run_query(req, mode="focus", stream=False),
    )
    return {
        "query": req.query,
        "response": response,
        "chunks_used": len(chunks),
        "sources": _source_list(chunks),
    }


# --- Add this route to Api/routes/query_routes.py ---
# Deliberately bypasses OmnissiahAgent and runtime_service.run_query —
# there's no LLM call, so there's no prompt to build and no session
# memory to update. This talks to the retriever directly.

@router.post("/query/ghost")
async def query_ghost(req: QueryRequest):
    """
    Pure-retrieval mode. Never calls an LLM, ever — not even as a
    fallback. If the scene library has a confident, lexically-grounded
    match, its chronicle(s) come back verbatim. If not, `matched` is
    false and `response` is empty — this endpoint does NOT fall back to
    a live search or a generated answer under any circumstance.

    Useful for: testing how much of your query traffic the pre-woven
    scene library alone can answer with zero LLM cost; an extremely
    cheap/fast tier; measuring where "no LLM needed" coverage actually
    stands today.
    """
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    retriever = runtime_service.retriever
    chunks = retriever.search_ghost(
        query=req.query,
        max_sources=req.max_sources,
        book_filter=req.book_filter,
        source_filter=req.source_filter,
    )

    if not chunks:
        return {
            "query": req.query,
            "matched": False,
            "response": "",
            "sources": [],
            "llm_used": False,
        }

    response_text = "\n\n".join(c.get("text", "").strip() for c in chunks if c.get("text"))

    return {
        "query": req.query,
        "matched": True,
        "response": response_text,
        "sources": _source_list(chunks),
        "llm_used": False,
    }