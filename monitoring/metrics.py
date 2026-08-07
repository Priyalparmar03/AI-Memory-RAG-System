from __future__ import annotations

import json
import logging
import threading
import uuid

from collections import defaultdict

from dataclasses import (
    dataclass,
    field,
)

from datetime import datetime

from enum import Enum

from statistics import (
    mean,
    median,
)

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

class MetricsError(Exception):
    """
    Metrics exception.
    """
    pass


# ==========================================================
# Metric Type
# ==========================================================

class MetricType(str, Enum):

    COUNTER = "counter"

    GAUGE = "gauge"

    HISTOGRAM = "histogram"

    SUMMARY = "summary"

    CUSTOM = "custom"


# ==========================================================
# Base Metric
# ==========================================================

@dataclass(slots=True)
class Metric:
    """
    Base metric.
    """

    name: str

    metric_type: MetricType

    description: str = ""

    value: float = 0.0

    labels: Dict[str, str] = field(

        default_factory=dict

    )

    metadata: Dict[str, Any] = field(

        default_factory=dict

    )

    created_at: datetime = field(

        default_factory=datetime.utcnow

    )

    updated_at: datetime = field(

        default_factory=datetime.utcnow

    )


    # ======================================================
    # Update Timestamp
    # ======================================================

    def touch(
        self,
    ) -> None:

        self.updated_at = (

            datetime.utcnow()

        )


    # ======================================================
    # Serialization
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {

            "name":

                self.name,

            "metric_type":

                self.metric_type.value,

            "description":

                self.description,

            "value":

                self.value,

            "labels":

                self.labels,

            "metadata":

                self.metadata,

            "created_at":

                self.created_at.isoformat(),

            "updated_at":

                self.updated_at.isoformat(),

        }


# ==========================================================
# Counter
# ==========================================================

@dataclass(slots=True)
class Counter(
    Metric
):
    """
    Counter metric.
    """

    metric_type: MetricType = (

        MetricType.COUNTER

    )

    value: float = 0.0


# ==========================================================
# Gauge
# ==========================================================

@dataclass(slots=True)
class Gauge(
    Metric
):
    """
    Gauge metric.
    """

    metric_type: MetricType = (

        MetricType.GAUGE

    )

    value: float = 0.0


# ==========================================================
# Histogram
# ==========================================================

@dataclass(slots=True)
class Histogram(
    Metric
):
    """
    Histogram metric.
    """

    metric_type: MetricType = (

        MetricType.HISTOGRAM

    )

    observations: List[float] = field(

        default_factory=list

    )

    value: float = 0.0


    # ======================================================
    # Serialization
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        data = super().to_dict()

        data.update(

            {

                "count":

                    len(

                        self.observations

                    ),

                "min":

                    min(

                        self.observations

                    )

                    if self.observations

                    else 0,

                "max":

                    max(

                        self.observations

                    )

                    if self.observations

                    else 0,

                "mean":

                    mean(

                        self.observations

                    )

                    if self.observations

                    else 0,

                "median":

                    median(

                        self.observations

                    )

                    if self.observations

                    else 0,

            }

        )

        return data


# ==========================================================
# Metrics Manager
# ==========================================================

class MetricsManager:
    """
    Production metrics manager.
    """

    def __init__(
        self,
    ):

        self.id = str(

            uuid.uuid4()

        )

        self.counters: Dict[
            str,

            Counter,

        ] = {}

        self.gauges: Dict[
            str,

            Gauge,

        ] = {}

        self.histograms: Dict[
            str,

            Histogram,

        ] = {}

        self.metadata: Dict[
            str,

            Any,

        ] = {}

        self.created_at = (

            datetime.utcnow()

        )

        self.updated_at = (

            datetime.utcnow()

        )

        self.lock = (

            threading.Lock()

        )


    # ======================================================
    # Timestamp
    # ======================================================

    def touch(
        self,
    ) -> None:

        self.updated_at = (

            datetime.utcnow()

        )


    # ======================================================
    # Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:

        return {

            "id":

                self.id,

            "counters":

                len(

                    self.counters

                ),

            "gauges":

                len(

                    self.gauges

                ),

            "histograms":

                len(

                    self.histograms

                ),

            "total_metrics":

                (

                    len(

                        self.counters

                    )

                    +

                    len(

                        self.gauges

                    )

                    +

                    len(

                        self.histograms

                    )

                ),

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

        return {

            "statistics":

                self.statistics(),

            "counters": {

                name:

                metric.to_dict()

                for name,

                metric

                in self.counters.items()

            },

            "gauges": {

                name:

                metric.to_dict()

                for name,

                metric

                in self.gauges.items()

            },

            "histograms": {

                name:

                metric.to_dict()

                for name,

                metric

                in self.histograms.items()

            },

            "metadata":

                self.metadata,

        }

# ======================================================
# Counter Operations
# ======================================================

def increment_counter(
    self,
    name: str,
    amount: float = 1.0,
    description: str = "",
) -> Counter:
    """
    Increment counter metric.
    """

    with self.lock:

        if name not in self.counters:

            self.counters[name] = Counter(

                name=name,

                description=description,

            )

        self.counters[name].value += amount

        self.counters[name].touch()

        self.touch()

        return self.counters[name]


# ======================================================
# Gauge Operations
# ======================================================

def set_gauge(
    self,
    name: str,
    value: float,
    description: str = "",
) -> Gauge:
    """
    Set gauge metric.
    """

    with self.lock:

        if name not in self.gauges:

            self.gauges[name] = Gauge(

                name=name,

                description=description,

            )

        self.gauges[name].value = value

        self.gauges[name].touch()

        self.touch()

        return self.gauges[name]


# ======================================================
# Histogram Operations
# ======================================================

def observe(
    self,
    name: str,
    value: float,
    description: str = "",
) -> Histogram:
    """
    Record histogram observation.
    """

    with self.lock:

        if name not in self.histograms:

            self.histograms[name] = Histogram(

                name=name,

                description=description,

            )

        histogram = self.histograms[name]

        histogram.observations.append(

            value

        )

        histogram.value = value

        histogram.touch()

        self.touch()

        return histogram


# ======================================================
# Token Metrics
# ======================================================

def record_tokens(
    self,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float = 0.0,
) -> None:
    """
    Record token usage.
    """

    total = (

        prompt_tokens

        +

        completion_tokens

    )

    self.increment_counter(

        "prompt_tokens",

        prompt_tokens,

    )

    self.increment_counter(

        "completion_tokens",

        completion_tokens,

    )

    self.increment_counter(

        "total_tokens",

        total,

    )

    self.increment_counter(

        "token_cost",

        cost,

    )


# ======================================================
# Latency Metrics
# ======================================================

def record_latency(
    self,
    operation: str,
    latency_ms: float,
) -> None:
    """
    Record operation latency.
    """

    self.observe(

        f"{operation}_latency_ms",

        latency_ms,

        description=

        "Operation latency",

    )


# ======================================================
# RAG Metrics
# ======================================================

def record_rag_query(
    self,
    retrieved_chunks: int,
    reranked_chunks: int = 0,
) -> None:
    """
    Record RAG query.
    """

    self.increment_counter(

        "rag_queries"

    )

    self.observe(

        "retrieved_chunks",

        retrieved_chunks,

    )

    self.observe(

        "reranked_chunks",

        reranked_chunks,

    )


# ======================================================
# LLM Metrics
# ======================================================

def record_llm_request(
    self,
    provider: str,
    model: str,
    latency_ms: float,
    tokens: int,
) -> None:
    """
    Record LLM request.
    """

    self.increment_counter(

        "llm_requests"

    )

    self.increment_counter(

        f"{provider}_requests"

    )

    self.increment_counter(

        f"{model}_requests"

    )

    self.record_latency(

        "llm",

        latency_ms,

    )

    self.increment_counter(

        "llm_tokens",

        tokens,

    )


# ======================================================
# System Metrics
# ======================================================

def update_system_metrics(
    self,
    cpu: float,
    memory: float,
    disk: float,
) -> None:
    """
    Update system gauges.
    """

    self.set_gauge(

        "cpu_usage",

        cpu,

    )

    self.set_gauge(

        "memory_usage",

        memory,

    )

    self.set_gauge(

        "disk_usage",

        disk,

    )


# ======================================================
# Get Counter
# ======================================================

def get_counter(
    self,
    name: str,
) -> Optional[Counter]:
    """
    Retrieve counter.
    """

    return self.counters.get(

        name

    )


# ======================================================
# Get Gauge
# ======================================================

def get_gauge(
    self,
    name: str,
) -> Optional[Gauge]:
    """
    Retrieve gauge.
    """

    return self.gauges.get(

        name

    )


# ======================================================
# Get Histogram
# ======================================================

def get_histogram(
    self,
    name: str,
) -> Optional[Histogram]:
    """
    Retrieve histogram.
    """

    return self.histograms.get(

        name

    )


# ======================================================
# Advanced Statistics
# ======================================================

def advanced_statistics(
    self,
) -> Dict[str, Any]:
    """
    Comprehensive metrics statistics.
    """

    counter_total = sum(

        metric.value

        for metric

        in self.counters.values()

    )

    gauge_average = (

        mean(

            [

                g.value

                for g

                in self.gauges.values()

            ]

        )

        if self.gauges

        else 0

    )

    histogram_samples = sum(

        len(

            h.observations

        )

        for h

        in self.histograms.values()

    )

    return {

        **self.statistics(),

        "counter_total":

            counter_total,

        "average_gauge":

            round(

                gauge_average,

                2,

            ),

        "histogram_samples":

            histogram_samples,

        "metadata_entries":

            len(

                self.metadata

            ),

    }
