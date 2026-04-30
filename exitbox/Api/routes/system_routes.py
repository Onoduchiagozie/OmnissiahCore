from fastapi import APIRouter, HTTPException, Query

from Api.services.runtime_service import runtime_service


router = APIRouter(tags=["System"])


@router.get("/health")
async def health():
    return runtime_service.health_payload()


@router.get("/info")
async def info():
    return runtime_service.info_payload()


@router.get("/config/runtime")
async def runtime_config():
    return runtime_service.runtime_config_payload()


@router.get("/sources", tags=["Index"])
async def list_sources():
    try:
        return runtime_service.list_sources_payload()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/{source_name}", tags=["Index"])
async def get_source_chunks(source_name: str, limit: int = Query(20, ge=1, le=100)):
    try:
        return runtime_service.source_chunks_payload(source_name, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory", tags=["Session"])
async def clear_memory(session_id: str = "default"):
    return runtime_service.clear_memory(session_id)


@router.get("/memory", tags=["Session"])
async def get_memory(session_id: str = "default"):
    return runtime_service.get_memory(session_id)