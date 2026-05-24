import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from Api.models import QueryRequest, QueryResponse, SourceInfo
from Api.services.runtime_service import runtime_service


router = APIRouter(tags=["Query"])

_ollama_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ollama")


def _source_list(chunks: list[dict]) -> list[SourceInfo]:
    return [
        SourceInfo(
            source=c.get("source", "?"),
            chapter=c.get("chapter", "?"),
            stitch_range=c.get("stitch_range", ""),
            score=round(
                c.get("rerank_score")
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
# POST /query/narrate/stream — primary user experience (narrator SSE)
# ---------------------------------------------------------------------------

@router.post("/query/narrate/stream")
async def query_narrate_stream(req: QueryRequest):
    """
    Primary endpoint. Narrator mode streamed via Server-Sent Events.

    Reconstructs events, battles, character arcs, and chronicles as a single
    flowing cinematic prose narrative. Tokens arrive one by one.

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
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _producer():
            try:
                for token in runtime_service.stream_query_mode(
                    req_narrator, mode="narrator"
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop)
            except Exception as exc:
                err_token = f"data: [ERROR] Narrator stream failed: {exc}\n\n"
                asyncio.run_coroutine_threadsafe(queue.put(err_token), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

        # Exactly one [DONE] — runtime_service.stream_query_mode does NOT yield it
        yield "data: [DONE]\n\n"

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

