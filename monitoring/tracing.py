from __future__ import annotations
import json
import logging
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class TracingError(Exception):
    """
    Tracing system exception.
    """
    pass


# ==========================================================
# Trace Status
# ==========================================================

class TraceStatus(str, Enum):

    UNSET = "unset"

    OK = "ok"

    ERROR = "error"


# ==========================================================
# Span Type
# ==========================================================

class SpanType(str, Enum):

    LLM = "llm"

    RAG = "rag"

    MEMORY = "memory"

    RETRIEVAL = "retrieval"

    EMBEDDING = "embedding"

    RERANKING = "reranking"

    TOOL = "tool"

    DATABASE = "database"

    HTTP = "http"

    CUSTOM = "custom"


# ==========================================================
# Trace Context
# ==========================================================

@dataclass(slots=True)
class TraceContext:
    """
    Context used to propagate trace information.
    """

    trace_id: str

    span_id: Optional[str] = None

    parent_span_id: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize trace context.
        """

        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "metadata": self.metadata,
        }


# ==========================================================
# Span
# ==========================================================

@dataclass(slots=True)
class Span:
    """
    Individual operation span.
    """

    name: str

    span_type: SpanType = SpanType.CUSTOM

    trace_id: str = ""

    span_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    parent_span_id: Optional[str] = None

    status: TraceStatus = TraceStatus.UNSET

    start_time: datetime = field(
        default_factory=datetime.utcnow
    )

    end_time: Optional[datetime] = None

    attributes: Dict[str, Any] = field(
        default_factory=dict
    )

    events: List[Dict[str, Any]] = field(
        default_factory=list
    )

    error: Optional[str] = None

    def __post_init__(
        self,
    ) -> None:
        """
        Validate span.
        """

        if not self.name.strip():

            raise TracingError(
                "Span name cannot be empty."
            )

        if not self.trace_id:

            raise TracingError(
                "Trace ID cannot be empty."
            )

    # ======================================================
    # Set Attribute
    # ======================================================

    def set_attribute(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Add or update span attribute.
        """

        if not key.strip():

            raise TracingError(
                "Attribute key cannot be empty."
            )

        self.attributes[key] = value

    # ======================================================
    # Add Event
    # ======================================================

    def add_event(
        self,
        name: str,
        attributes: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        """
        Add event to span.
        """

        self.events.append(
            {
                "name": name,
                "timestamp": datetime.utcnow().isoformat(),
                "attributes": attributes or {},
            }
        )

    # ======================================================
    # Set Status
    # ======================================================

    def set_status(
        self,
        status: TraceStatus,
        error: Optional[str] = None,
    ) -> None:
        """
        Update span status.
        """

        self.status = status

        if error:

            self.error = error

    # ======================================================
    # End Span
    # ======================================================

    def end(
        self,
        status: Optional[TraceStatus] = None,
    ) -> None:
        """
        End span.
        """

        self.end_time = datetime.utcnow()

        if status is not None:

            self.status = status

    # ======================================================
    # Duration
    # ======================================================

    @property
    def duration_ms(
        self,
    ) -> float:
        """
        Span duration in milliseconds.
        """

        end = (

            self.end_time

            or

            datetime.utcnow()

        )

        return round(
            (
                end - self.start_time
            ).total_seconds() * 1000,
            3,
        )

    # ======================================================
    # Finished
    # ======================================================

    @property
    def is_finished(
        self,
    ) -> bool:
        """
        Whether span has ended.
        """

        return self.end_time is not None

    # ======================================================
    # Serialization
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize span.
        """

        return {
            "name": self.name,
            "span_type": self.span_type.value,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": (
                self.end_time.isoformat()
                if self.end_time
                else None
            ),
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
        }


# ==========================================================
# Trace
# ==========================================================

@dataclass(slots=True)
class Trace:
    """
    Complete distributed trace.
    """

    name: str

    trace_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    status: TraceStatus = TraceStatus.UNSET

    start_time: datetime = field(
        default_factory=datetime.utcnow
    )

    end_time: Optional[datetime] = None

    spans: List[Span] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:

        if not self.name.strip():

            raise TracingError(
                "Trace name cannot be empty."
            )

    # ======================================================
    # Add Span
    # ======================================================

    def add_span(
        self,
        span: Span,
    ) -> None:
        """
        Add span to trace.
        """

        if span.trace_id != self.trace_id:

            raise TracingError(
                "Span belongs to another trace."
            )

        self.spans.append(span)

    # ======================================================
    # End Trace
    # ======================================================

    def end(
        self,
        status: Optional[TraceStatus] = None,
    ) -> None:
        """
        End trace.
        """

        self.end_time = datetime.utcnow()

        if status is not None:

            self.status = status

    # ======================================================
    # Duration
    # ======================================================

    @property
    def duration_ms(
        self,
    ) -> float:
        """
        Trace duration in milliseconds.
        """

        end = (

            self.end_time

            or

            datetime.utcnow()

        )

        return round(
            (
                end - self.start_time
            ).total_seconds() * 1000,
            3,
        )

    # ======================================================
    # Serialization
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize trace.
        """

        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": (
                self.end_time.isoformat()
                if self.end_time
                else None
            ),
            "duration_ms": self.duration_ms,
            "spans": [
                span.to_dict()
                for span in self.spans
            ],
            "metadata": self.metadata,
        }


# ==========================================================
# Tracer
# ==========================================================

class Tracer:
    """
    Production tracing manager.
    """

    def __init__(
        self,
        service_name: str = "AI_MEMORY",
    ) -> None:

        if not service_name.strip():

            raise TracingError(
                "Service name cannot be empty."
            )

        self.service_name = service_name

        self.traces: Dict[
            str,
            Trace,
        ] = {}

        self.active_spans: Dict[
            str,
            Span,
        ] = {}

        self.lock = threading.RLock()

        self.created_at = datetime.utcnow()

        self.updated_at = datetime.utcnow()

    # ======================================================
    # Timestamp
    # ======================================================

    def touch(
        self,
    ) -> None:

        self.updated_at = datetime.utcnow()

    # ======================================================
    # Start Trace
    # ======================================================

    def start_trace(
        self,
        name: str,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Trace:
        """
        Start a new trace.
        """

        trace = Trace(
            name=name,
            metadata=metadata or {},
        )

        with self.lock:

            self.traces[
                trace.trace_id
            ] = trace

            self.touch()

        return trace

    # ======================================================
    # Start Span
    # ======================================================

    def start_span(
        self,
        trace: Trace,
        name: str,
        span_type: SpanType = SpanType.CUSTOM,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        """
        Start a span inside a trace.
        """

        if trace.trace_id not in self.traces:

            raise TracingError(
                "Trace is not registered."
            )

        span = Span(
            name=name,
            span_type=span_type,
            trace_id=trace.trace_id,
            parent_span_id=parent_span_id,
        )

        with self.lock:

            trace.add_span(span)

            self.active_spans[
                span.span_id
            ] = span

            self.touch()

        return span

    # ======================================================
    # Serialization
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize tracer state.
        """

        return {
            "service_name": self.service_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "traces": [
                trace.to_dict()
                for trace in self.traces.values()
            ],
            "active_spans": len(
                self.active_spans
            ),
        }

    # ======================================================
    # JSON
    # ======================================================

    def to_json(
        self,
        indent: int = 4,
    ) -> str:
        """
        Serialize tracer to JSON.
        """

        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
        )
