from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The lore question to answer.")
    session_id: Optional[str] = Field("default", description="Client session identifier.")
    book_filter: Optional[str] = Field(None, description="Substring filter on book name.")
    source_filter: Optional[list[str]] = Field(None, description="Exact source name list.")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Max chunks to return.")
    candidate_pool: Optional[int] = Field(None, ge=1, le=100, description="Initial retrieval pool size.")
    stitching_window: Optional[int] = Field(None, ge=0, le=6, description="Neighbour chunks to stitch.")



class SourceInfo(BaseModel):
    source: str
    chapter: str
    stitch_range: str
    score: float


class QueryResponse(BaseModel):
    query: str
    response: str
    sources: list[SourceInfo]
    chunks_used: int
