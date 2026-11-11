"""
OmnissiahCore - Api/models.py
"""
import uuid
from datetime import datetime, timezone
from typing import Optional, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The lore question to answer.")
    response_style: Optional[Literal["auto", "explorer", "narrator"]] = Field(
        None, description="How the answer sounds. Auto infers from query; narrator is cinematic, explorer explains."
    )
    retrieval_scope: Optional[Literal["wide", "focus"]] = Field(
        None, description="How broadly to search evidence, independent of response_style."
    )
    session_id: Optional[str] = Field("default", description="Client session identifier.")
    book_filter: Optional[str] = Field(None, description="Substring filter on book name.")
    source_filter: Optional[list[str]] = Field(None, description="Exact source name list.")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Max chunks to return.")
    candidate_pool: Optional[int] = Field(None, ge=1, le=100, description="Initial retrieval pool size.")
    stitching_window: Optional[int] = Field(None, ge=0, le=6, description="Neighbour chunks to stitch.")
    max_sources: Optional[int] = Field(
        2, ge=1, le=3,
        description="Focus/ghost mode: max stitched source groups returned.",
    )


class SourceInfo(BaseModel):
    source: str
    chapter: str
    stitch_range: str
    score: float


class QueryResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = Field(default="ok")
    query: str
    response: str
    sources: list[SourceInfo]
    chunks_used: int
    llm_used: bool = Field(default=True)
    path: Optional[str] = Field(
        None,
        description="Which retrieval path fired: 'scene_library', 'live_search', or 'ghost'."
    )


class LlmStatusResponse(BaseModel):
    status: str          # "green" | "amber" | "red"
    active: int
    queued: int
    message: str
    max_concurrent: int


class BookEntry(BaseModel):
    filename: str
    book_order: int
    title: str
    category: Optional[str] = Field(
        None,
        description="'horus_heresy', 'codex', 'anthology', or None"
    )


class BookListResponse(BaseModel):
    total: int
    books: list[BookEntry]