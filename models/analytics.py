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

from typing import (
    Any,
    Dict,
    List,
    Optional,
)


logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class AnalyticsError(Exception):
    """
    Analytics model exception.
    """
    pass


# ==========================================================
# Analytics Type
# ==========================================================

class AnalyticsType(str, Enum):

    SYSTEM = "system"

    USER = "user"

    SESSION = "session"

    DOCUMENT = "document"

    RAG = "rag"

    TOKEN = "token"

    PERFORMANCE = "performance"

    CUSTOM = "custom"


# ==========================================================
# Analytics Period
# ==========================================================

class AnalyticsPeriod(str, Enum):

    HOURLY = "hourly"

    DAILY = "daily"

    WEEKLY = "weekly"

    MONTHLY = "monthly"

    YEARLY = "yearly"

    CUSTOM = "custom"


# ==========================================================
# Analytics
# ==========================================================

@dataclass(slots=True)
class Analytics:
    """
    Production Analytics Model.
    """

    analytics_type: AnalyticsType = (

        AnalyticsType.SYSTEM

    )

    period: AnalyticsPeriod = (

        AnalyticsPeriod.DAILY

    )

    session_count: int = 0

    message_count: int = 0

    document_count: int = 0

    chunk_count: int = 0

    total_tokens: int = 0

    total_cost: float = 0.0

    average_latency_ms: float = 0.0

    retrieval_count: int = 0

    successful_queries: int = 0

    failed_queries: int = 0

    metadata: Dict[str, Any] = field(

        default_factory=dict

    )

    id: str = field(

        default_factory=lambda:

        str(uuid.uuid4())

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
        Validate analytics values.
        """

        numeric_values = [

            self.session_count,

            self.message_count,

            self.document_count,

            self.chunk_count,

            self.total_tokens,

            self.total_cost,

            self.average_latency_ms,

            self.retrieval_count,

            self.successful_queries,

            self.failed_queries,

        ]

        if any(

            value < 0

            for value

            in numeric_values

        ):

            raise AnalyticsError(

                "Analytics values "

                "cannot be negative."

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
    # Update Metadata
    # ======================================================

    def update_metadata(
        self,
        **kwargs,
    ) -> None:
        """
        Update metadata.
        """

        self.metadata.update(

            kwargs

        )

        self.touch()


    # ======================================================
    # Success Rate
    # ======================================================

    @property
    def success_rate(
        self,
    ) -> float:
        """
        Query success rate.
        """

        total = (

            self.successful_queries

            +

            self.failed_queries

        )

        if total == 0:

            return 0.0

        return round(

            (

                self.successful_queries

                /

                total

            )

            * 100,

            2,

        )


    # ======================================================
    # Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Basic analytics statistics.
        """

        return {

            "id":

                self.id,

            "analytics_type":

                self.analytics_type.value,

            "period":

                self.period.value,

            "sessions":

                self.session_count,

            "messages":

                self.message_count,

            "documents":

                self.document_count,

            "chunks":

                self.chunk_count,

            "tokens":

                self.total_tokens,

            "cost":

                self.total_cost,

            "latency_ms":

                self.average_latency_ms,

            "retrievals":

                self.retrieval_count,

            "success_rate":

                self.success_rate,

            "created_at":

                self.created_at.isoformat(),

            "updated_at":

                self.updated_at.isoformat(),

        }


    # ======================================================
    # Serialization
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize analytics.
        """

        return {

            "id":

                self.id,

            "analytics_type":

                self.analytics_type.value,

            "period":

                self.period.value,

            "session_count":

                self.session_count,

            "message_count":

                self.message_count,

            "document_count":

                self.document_count,

            "chunk_count":

                self.chunk_count,

            "total_tokens":

                self.total_tokens,

            "total_cost":

                self.total_cost,

            "average_latency_ms":

                self.average_latency_ms,

            "retrieval_count":

                self.retrieval_count,

            "successful_queries":

                self.successful_queries,

            "failed_queries":

                self.failed_queries,

            "metadata":

                self.metadata,

            "created_at":

                self.created_at.isoformat(),

            "updated_at":

                self.updated_at.isoformat(),

        }

  # ======================================================
# Update Session Count
# ======================================================

def update_sessions(
    self,
    count: int = 1,
) -> None:
    """
    Update session count.
    """

    if count < 0:

        raise AnalyticsError(

            "Count cannot be negative."

        )

    self.session_count += count

    self.touch()


# ======================================================
# Update Message Count
# ======================================================

def update_messages(
    self,
    count: int = 1,
) -> None:
    """
    Update message count.
    """

    if count < 0:

        raise AnalyticsError(

            "Count cannot be negative."

        )

    self.message_count += count

    self.touch()


# ======================================================
# Update Document Count
# ======================================================

def update_documents(
    self,
    count: int = 1,
) -> None:
    """
    Update document count.
    """

    if count < 0:

        raise AnalyticsError(

            "Count cannot be negative."

        )

    self.document_count += count

    self.touch()


# ======================================================
# Update Chunk Count
# ======================================================

def update_chunks(
    self,
    count: int = 1,
) -> None:
    """
    Update chunk count.
    """

    if count < 0:

        raise AnalyticsError(

            "Count cannot be negative."

        )

    self.chunk_count += count

    self.touch()


# ======================================================
# Update Token Usage
# ======================================================

def update_tokens(
    self,
    tokens: int,
    cost: float = 0.0,
) -> None:
    """
    Update token usage.
    """

    if tokens < 0 or cost < 0:

        raise AnalyticsError(

            "Tokens and cost "

            "cannot be negative."

        )

    self.total_tokens += tokens

    self.total_cost += cost

    self.touch()


# ======================================================
# Update Latency
# ======================================================

def update_latency(
    self,
    latency_ms: float,
) -> None:
    """
    Update average latency.
    """

    if latency_ms < 0:

        raise AnalyticsError(

            "Latency cannot "

            "be negative."

        )

    if self.average_latency_ms == 0:

        self.average_latency_ms = latency_ms

    else:

        self.average_latency_ms = round(

            (

                self.average_latency_ms

                + latency_ms

            )

            / 2,

            2,

        )

    self.touch()


# ======================================================
# Record Retrieval
# ======================================================

def record_retrieval(
    self,
) -> None:
    """
    Increment retrieval count.
    """

    self.retrieval_count += 1

    self.touch()


# ======================================================
# Record Successful Query
# ======================================================

def record_success(
    self,
) -> None:
    """
    Record successful query.
    """

    self.successful_queries += 1

    self.touch()


# ======================================================
# Record Failed Query
# ======================================================

def record_failure(
    self,
) -> None:
    """
    Record failed query.
    """

    self.failed_queries += 1

    self.touch()


# ======================================================
# Merge Analytics
# ======================================================

def merge(
    self,
    other: "Analytics",
) -> None:
    """
    Merge another analytics object.
    """

    if not isinstance(

        other,

        Analytics,

    ):

        raise AnalyticsError(

            "Expected Analytics."

        )

    self.session_count += (

        other.session_count

    )

    self.message_count += (

        other.message_count

    )

    self.document_count += (

        other.document_count

    )

    self.chunk_count += (

        other.chunk_count

    )

    self.total_tokens += (

        other.total_tokens

    )

    self.total_cost += (

        other.total_cost

    )

    self.retrieval_count += (

        other.retrieval_count

    )

    self.successful_queries += (

        other.successful_queries

    )

    self.failed_queries += (

        other.failed_queries

    )

    if (

        other.average_latency_ms > 0

    ):

        self.update_latency(

            other.average_latency_ms

        )

    self.metadata.update(

        other.metadata

    )

    self.touch()


# ======================================================
# Average Cost Per Token
# ======================================================

@property
def average_cost_per_token(
    self,
) -> float:
    """
    Average token cost.
    """

    if self.total_tokens == 0:

        return 0.0

    return round(

        self.total_cost

        /

        self.total_tokens,

        8,

    )


# ======================================================
# Average Messages Per Session
# ======================================================

@property
def average_messages_per_session(
    self,
) -> float:
    """
    Messages per session.
    """

    if self.session_count == 0:

        return 0.0

    return round(

        self.message_count

        /

        self.session_count,

        2,

    )


# ======================================================
# Average Chunks Per Document
# ======================================================

@property
def average_chunks_per_document(
    self,
) -> float:
    """
    Chunks per document.
    """

    if self.document_count == 0:

        return 0.0

    return round(

        self.chunk_count

        /

        self.document_count,

        2,

    )


# ======================================================
# Advanced Statistics
# ======================================================

def advanced_statistics(
    self,
) -> Dict[str, Any]:
    """
    Advanced analytics statistics.
    """

    return {

        **self.statistics(),

        "average_cost_per_token":

            self.average_cost_per_token,

        "average_messages_per_session":

            self.average_messages_per_session,

        "average_chunks_per_document":

            self.average_chunks_per_document,

        "metadata_entries":

            len(

                self.metadata

            ),

    }
