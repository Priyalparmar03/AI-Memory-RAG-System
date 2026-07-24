from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ==========================================================
# Upload Response
# ==========================================================

class DocumentUploadResponse(BaseModel):

    document_id: int

    filename: str

    chunks: int

    status: str


# ==========================================================
# Document Metadata
# ==========================================================

class DocumentMetadata(BaseModel):

    id: int

    filename: str

    file_type: str

    size: int

    uploaded_at: str

    pages: Optional[int] = None

    chunks: int


# ==========================================================
# Search Request
# ==========================================================

class DocumentSearchRequest(BaseModel):

    query: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


# ==========================================================
# Search Result
# ==========================================================

class SearchResult(BaseModel):

    chunk: str

    score: float

    source: str

    page: Optional[int] = None


class SearchResponse(BaseModel):

    query: str

    results: List[SearchResult]
