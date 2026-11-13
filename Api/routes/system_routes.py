"""
OmnissiahCore - Api/routes/system_routes.py
"""
import json
import os
import secrets

from fastapi import APIRouter, HTTPException, Query, Header

from Api.services.runtime_service import runtime_service, get_llm_status
from Core.config_loader import paths

router = APIRouter(tags=["System"])


@router.get("/version")
async def version():
    """Identify deployed code without exposing providers or private settings."""
    from pathlib import Path
    manifest = Path(__file__).resolve().parents[2] / "release.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return {"commit": data["commit"], "deployed_at": data.get("deployed_at")}
    return {"commit": "development", "deployed_at": None}


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


def _require_memory_admin(header: str | None):
    # Diagnostics are disabled unless an administrator provisions a private
    # token in the server environment. Session IDs do not authorize access.
    expected = os.getenv("OMNISSIAH_MEMORY_ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=404, detail="Not found")
    if not header or not secrets.compare_digest(expected, header):
        raise HTTPException(status_code=403, detail="Forbidden")


@router.delete("/memory", tags=["Session"])
async def clear_memory(
    session_id: str,
    x_omnissiah_memory_admin: str | None = Header(default=None),
):
    _require_memory_admin(x_omnissiah_memory_admin)
    return runtime_service.clear_memory(session_id)


@router.get("/memory", tags=["Session"])
async def get_memory(
    session_id: str,
    x_omnissiah_memory_admin: str | None = Header(default=None),
):
    _require_memory_admin(x_omnissiah_memory_admin)
    return runtime_service.get_memory(session_id)


# ---------------------------------------------------------------------------
# GET /llm/status — traffic light for concurrent query control
#
# Returns green (idle), amber (busy / queued), or red (error).
# TheForge polls this every 3-5 seconds to show the colored indicator
# and block the submit button when the LLM slot is taken.
# ---------------------------------------------------------------------------

@router.get("/llm/status", tags=["System"])
async def llm_status():
    """
    Returns current LLM concurrency state.

    green  — LLM is idle, ready to accept a query
    amber  — LLM is busy or a query is queued; new queries will wait
    red    — LLM backend unreachable (future: health-check ping)

    TheForge should poll this every 3-5 seconds and disable the submit
    button when status is amber, showing a colored indicator to users.
    """
    return get_llm_status()


# ---------------------------------------------------------------------------
# GET /books — canonical 224-entry Horus Heresy book manifest
#
# Returns clean display titles + filenames for the book filter dropdown.
# Source of truth for book count — NOT /sources, which inflates the count
# via cross-format duplicates (.mobi vs .pdf of the same book).
#
# Categories injected at serve time based on title patterns:
#   horus_heresy — numbered HH series books
#   novella      — HH novellas
#   short_story  — standalone short stories
#   anthology    — multi-author collections
#   codex        — Warhammer rulebooks / codices (future)
# ---------------------------------------------------------------------------

_book_manifest_cache: list[dict] | None = None


def _categorise(entry: dict) -> str:
    title = (entry.get("title") or "").lower()
    filename = (entry.get("filename") or "").lower()

    if "codex" in filename or "codex" in title or "[codex]" in filename:
        return "codex"
    if "anthology" in title or "[anthology]" in title:
        return "anthology"
    if "novella" in filename or "novella" in title:
        return "novella"
    if "short story" in filename or "short story" in title:
        return "short_story"
    # Numbered HH main series: filename starts with a 2-digit number
    import re
    if re.match(r"^\d{2}\.", entry.get("filename", "")):
        return "horus_heresy"
    return "other"


@router.get("/books", tags=["Index"])
async def list_books():
    """
    Returns the canonical book manifest.

    `title`    — clean display name for the UI dropdown
    `filename` — exact value to send as `book_filter` on a query request
                 (matches the raw `source` field in metadata.json)
    `category` — UI grouping hint: horus_heresy | novella | short_story |
                 anthology | codex | other
    `book_order` — sort order within the manifest
    """
    global _book_manifest_cache
    if _book_manifest_cache is not None:
        return {"total": len(_book_manifest_cache), "books": _book_manifest_cache}

    manifest_path = paths.get("book_manifest")
    if not manifest_path:
        raise HTTPException(status_code=503, detail="Book manifest path not configured.")

    if not os.path.exists(manifest_path):
        raise HTTPException(
            status_code=503,
            detail=f"Book manifest not found at {manifest_path}. Place book_manifest.json in Db/.",
        )

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Book manifest is malformed: {e}")

    books = data.get("books", [])
    for entry in books:
        entry["category"] = _categorise(entry)

    # Sort: Horus Heresy numbered entries first (by book_order), then rest alphabetically
    def sort_key(e):
        cat = e.get("category", "other")
        order = e.get("book_order", 9999)
        title = (e.get("title") or "").lower()
        if cat == "horus_heresy":
            return (0, order, title)
        elif cat == "novella":
            return (1, order, title)
        elif cat == "short_story":
            return (2, order, title)
        elif cat == "anthology":
            return (3, order, title)
        elif cat == "codex":
            return (4, order, title)
        return (5, order, title)

    books.sort(key=sort_key)
    _book_manifest_cache = books
    return {"total": len(books), "books": books}
