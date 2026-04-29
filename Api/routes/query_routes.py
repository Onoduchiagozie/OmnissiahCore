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


@router.post("/query/inspect")
async def query_inspect(req: QueryRequest):
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_ollama_pool, lambda: runtime_service.inspect_query(req))


@router.post("/query", response_model=QueryResponse)
async def query_sync(req: QueryRequest):
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    loop = asyncio.get_event_loop()
    response, chunks = await loop.run_in_executor(
        _ollama_pool,
        lambda: runtime_service.run_query(req, mode="remembrancer", stream=False),
    )
    return QueryResponse(
        query=req.query,
        response=response,
        sources=_source_list(chunks),
        chunks_used=len(chunks),
    )


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return StreamingResponse(
        runtime_service.stream_query(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/query/narrate", response_model=QueryResponse)
async def query_narrate(req: QueryRequest):
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    loop = asyncio.get_event_loop()
    response, chunks = await loop.run_in_executor(
        _ollama_pool,
        lambda: runtime_service.run_query(req, mode="narrator", stream=False),
    )
    return QueryResponse(
        query=req.query,
        response=response,
        sources=_source_list(chunks),
        chunks_used=len(chunks),
    )


@router.post("/query/narrate/stream")
async def query_narrate_stream(req: QueryRequest):
    runtime_service.ensure_ready()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    req_narrator = req.model_copy(update={"session_id": req.session_id or "narrate-default"})

    async def _stream():
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def _producer():
            try:
                for token in runtime_service.stream_query_mode(req_narrator, mode="narrator"):
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/query/explore")
async def query_explore(req: QueryRequest):
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
        "sources": [
            {
                "source": c.get("source", "?"),
                "chapter": c.get("chapter", "?"),
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