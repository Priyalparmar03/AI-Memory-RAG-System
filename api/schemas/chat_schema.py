from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ==========================================================
# Chat Request
# ==========================================================

class ChatRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    session_id: Optional[str] = None

    model: str = Field(
        default="gemini",
    )

    temperature: float = Field(
        default=0.3,
        ge=0,
        le=2,
    )

    stream: bool = False


# ==========================================================
# Chat Response
# ==========================================================

class ChatResponse(BaseModel):

    session_id: str

    response: str

    model: str

    latency_ms: float

    tokens_used: int

    sources: List[str] = []


# ==========================================================
# Chat History
# ==========================================================

class ChatMessage(BaseModel):

    role: str

    content: str

    timestamp: str


class ChatHistory(BaseModel):

    session_id: str

    messages: List[ChatMessage]
