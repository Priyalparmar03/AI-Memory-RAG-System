from __future__ import annotations

import json
import logging
import uuid

from dataclasses import (
    dataclass,
    field,
)

from datetime import datetime

from enum import Enum

from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class TokenUsageError(Exception):
    """
    Token usage exception.
    """
    pass


# ==========================================================
# Provider
# ==========================================================

class Provider(str, Enum):

    OPENAI = "openai"

    ANTHROPIC = "anthropic"

    GOOGLE = "google"

    MISTRAL = "mistral"

    COHERE = "cohere"

    QWEN = "qwen"

    OLLAMA = "ollama"

    CUSTOM = "custom"


# ==========================================================
# Model Type
# ==========================================================

class ModelType(str, Enum):

    CHAT = "chat"

    EMBEDDING = "embedding"

    RERANKER = "reranker"

    OCR = "ocr"

    VISION = "vision"

    AUDIO = "audio"

    AGENT = "agent"


# ==========================================================
# Token Usage
# ==========================================================

@dataclass(slots=True)
class TokenUsage:
    """
    Production Token Usage Model.
    """

    provider: Provider

    model_name: str

    model_type: ModelType

    prompt_tokens: int = 0

    completion_tokens: int = 0

    embedding_tokens: int = 0

    cached_tokens: int = 0

    cost_per_1k_tokens: float = 0.0

    currency: str = "USD"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    id: str = field(
        default_factory=lambda: str(
            uuid.uuid4()
        )
    )

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    updated_at: datetime = field(
        default_factory=datetime.utcnow
    )

    # ======================================================
    # Initialization
    # ======================================================

    def __post_init__(
        self,
    ):

        self.validate()

    # ======================================================
    # Validation
    # ======================================================

    def validate(
        self,
    ) -> None:
        """
        Validate token counts.
        """

        values = [

            self.prompt_tokens,

            self.completion_tokens,

            self.embedding_tokens,

            self.cached_tokens,

        ]

        if any(

            value < 0

            for value

            in values

        ):

            raise TokenUsageError(

                "Token counts cannot "

                "be negative."

            )

        if self.cost_per_1k_tokens < 0:

            raise TokenUsageError(

                "Cost cannot "

                "be negative."

            )

    # ======================================================
    # Total Tokens
    # ======================================================

    @property
    def total_tokens(
        self,
    ) -> int:
        """
        Total tokens.
        """

        return (

            self.prompt_tokens

            +

            self.completion_tokens

            +

            self.embedding_tokens

            +

            self.cached_tokens

        )

    # ======================================================
    # Total Cost
    # ======================================================

    @property
    def total_cost(
        self,
    ) -> float:
        """
        Calculate total cost.
        """

        return round(

            (

                self.total_tokens

                / 1000

            )

            *

            self.cost_per_1k_tokens,

            6,

        )

    # ======================================================
    # Update Timestamp
    # ======================================================

    def touch(
        self,
    ) -> None:
        """
        Update timestamp.
        """

        self.updated_at = (

            datetime.utcnow()

        )

    # ======================================================
    # Serialization
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize object.
        """

        return {

            "id":

                self.id,

            "provider":

                self.provider.value,

            "model_name":

                self.model_name,

            "model_type":

                self.model_type.value,

            "prompt_tokens":

                self.prompt_tokens,

            "completion_tokens":

                self.completion_tokens,

            "embedding_tokens":

                self.embedding_tokens,

            "cached_tokens":

                self.cached_tokens,

            "total_tokens":

                self.total_tokens,

            "cost_per_1k_tokens":

                self.cost_per_1k_tokens,

            "total_cost":

                self.total_cost,

            "currency":

                self.currency,

            "metadata":

                self.metadata,

            "created_at":

                self.created_at.isoformat(),

            "updated_at":

                self.updated_at.isoformat(),

        }


# ==========================================================
# Default Pricing (Example Values)
# ==========================================================

DEFAULT_PRICING = {

    Provider.OPENAI: 0.005,

    Provider.ANTHROPIC: 0.008,

    Provider.GOOGLE: 0.002,

    Provider.MISTRAL: 0.003,

    Provider.COHERE: 0.001,

    Provider.QWEN: 0.001,

    Provider.OLLAMA: 0.0,

    Provider.CUSTOM: 0.0,

}

# ======================================================
# Add Prompt Tokens
# ======================================================

def add_prompt_tokens(
    self,
    tokens: int,
) -> None:
    """
    Add prompt tokens.
    """

    if tokens < 0:

        raise TokenUsageError(

            "Token count cannot "

            "be negative."

        )

    self.prompt_tokens += tokens

    self.touch()


# ======================================================
# Add Completion Tokens
# ======================================================

def add_completion_tokens(
    self,
    tokens: int,
) -> None:
    """
    Add completion tokens.
    """

    if tokens < 0:

        raise TokenUsageError(

            "Token count cannot "

            "be negative."

        )

    self.completion_tokens += tokens

    self.touch()


# ======================================================
# Add Embedding Tokens
# ======================================================

def add_embedding_tokens(
    self,
    tokens: int,
) -> None:
    """
    Add embedding tokens.
    """

    if tokens < 0:

        raise TokenUsageError(

            "Token count cannot "

            "be negative."

        )

    self.embedding_tokens += tokens

    self.touch()


# ======================================================
# Add Cached Tokens
# ======================================================

def add_cached_tokens(
    self,
    tokens: int,
) -> None:
    """
    Add cached tokens.
    """

    if tokens < 0:

        raise TokenUsageError(

            "Token count cannot "

            "be negative."

        )

    self.cached_tokens += tokens

    self.touch()


# ======================================================
# Update Provider Pricing
# ======================================================

def update_pricing(
    self,
    cost_per_1k_tokens: float,
) -> None:
    """
    Update pricing.
    """

    if cost_per_1k_tokens < 0:

        raise TokenUsageError(

            "Pricing cannot "

            "be negative."

        )

    self.cost_per_1k_tokens = (

        cost_per_1k_tokens

    )

    self.touch()


# ======================================================
# Apply Default Pricing
# ======================================================

def apply_default_pricing(
    self,
) -> None:
    """
    Apply default provider pricing.
    """

    self.cost_per_1k_tokens = (

        DEFAULT_PRICING.get(

            self.provider,

            0.0,

        )

    )

    self.touch()


# ======================================================
# Merge Usage
# ======================================================

def merge(
    self,
    other: "TokenUsage",
) -> None:
    """
    Merge another usage object.
    """

    if not isinstance(

        other,

        TokenUsage,

    ):

        raise TokenUsageError(

            "Expected TokenUsage."

        )

    self.prompt_tokens += (

        other.prompt_tokens

    )

    self.completion_tokens += (

        other.completion_tokens

    )

    self.embedding_tokens += (

        other.embedding_tokens

    )

    self.cached_tokens += (

        other.cached_tokens

    )

    self.metadata.update(

        other.metadata

    )

    self.touch()


# ======================================================
# Reset Usage
# ======================================================

def reset(
    self,
) -> None:
    """
    Reset token usage.
    """

    self.prompt_tokens = 0

    self.completion_tokens = 0

    self.embedding_tokens = 0

    self.cached_tokens = 0

    self.touch()


# ======================================================
# Statistics
# ======================================================

def statistics(
    self,
) -> Dict[str, Any]:
    """
    Usage statistics.
    """

    return {

        "provider":

            self.provider.value,

        "model":

            self.model_name,

        "model_type":

            self.model_type.value,

        "prompt_tokens":

            self.prompt_tokens,

        "completion_tokens":

            self.completion_tokens,

        "embedding_tokens":

            self.embedding_tokens,

        "cached_tokens":

            self.cached_tokens,

        "total_tokens":

            self.total_tokens,

        "cost_per_1k":

            self.cost_per_1k_tokens,

        "total_cost":

            self.total_cost,

        "currency":

            self.currency,

    }


# ======================================================
# Token Breakdown
# ======================================================

def token_breakdown(
    self,
) -> Dict[str, float]:
    """
    Percentage contribution
    of each token type.
    """

    total = max(

        self.total_tokens,

        1,

    )

    return {

        "prompt_percent":

            round(

                self.prompt_tokens

                / total

                * 100,

                2,

            ),

        "completion_percent":

            round(

                self.completion_tokens

                / total

                * 100,

                2,

            ),

        "embedding_percent":

            round(

                self.embedding_tokens

                / total

                * 100,

                2,

            ),

        "cached_percent":

            round(

                self.cached_tokens

                / total

                * 100,

                2,

            ),

    }


# ======================================================
# Is Empty
# ======================================================

@property
def is_empty(
    self,
) -> bool:
    """
    Check whether usage is empty.
    """

    return self.total_tokens == 0
