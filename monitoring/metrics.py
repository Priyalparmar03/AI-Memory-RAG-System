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

# ======================================================
# JSON Serialization
# ======================================================

def to_json(
    self,
    indent: int = 4,
) -> str:
    """
    Serialize metrics to JSON.
    """

    return json.dumps(

        self.to_dict(),

        indent=indent,

        ensure_ascii=False,

    )


# ======================================================
# Summary
# ======================================================

def summary(
    self,
) -> Dict[str, Any]:
    """
    Metrics summary.
    """

    return {

        "id":

            self.id,

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

        "created_at":

            self.created_at.isoformat(),

        "updated_at":

            self.updated_at.isoformat(),

    }


# ======================================================
# Diagnostics
# ======================================================

def diagnostics(
    self,
) -> Dict[str, Any]:
    """
    Complete diagnostics.
    """

    return {

        "summary":

            self.summary(),

        "statistics":

            self.advanced_statistics(),

        "metadata":

            self.metadata,

    }


# ======================================================
# Prometheus Export
# ======================================================

def prometheus(
    self,
) -> str:
    """
    Export metrics in Prometheus format.
    """

    lines = []

    for metric in self.counters.values():

        lines.append(

            f"# TYPE {metric.name} counter"

        )

        lines.append(

            f"{metric.name} {metric.value}"

        )

    for metric in self.gauges.values():

        lines.append(

            f"# TYPE {metric.name} gauge"

        )

        lines.append(

            f"{metric.name} {metric.value}"

        )

    for metric in self.histograms.values():

        lines.append(

            f"# TYPE {metric.name} histogram"

        )

        lines.append(

            f"{metric.name}_count "

            f"{len(metric.observations)}"

        )

        lines.append(

            f"{metric.name}_sum "

            f"{sum(metric.observations)}"

        )

    return "\n".join(

        lines

    )


# ======================================================
# Export
# ======================================================

def export(
    self,
) -> Dict[str, Any]:
    """
    Export complete metrics.
    """

    return {

        "summary":

            self.summary(),

        "statistics":

            self.advanced_statistics(),

        "diagnostics":

            self.diagnostics(),

        "metrics":

            self.to_dict(),

    }


# ======================================================
# Search Metrics
# ======================================================

def search(
    self,
    keyword: str,
) -> Dict[str, Metric]:
    """
    Search metrics by name.
    """

    keyword = keyword.lower()

    results = {}

    for collection in (

        self.counters,

        self.gauges,

        self.histograms,

    ):

        for name, metric in collection.items():

            if keyword in name.lower():

                results[name] = metric

    return results


# ======================================================
# Filter Metrics
# ======================================================

def filter_type(
    self,
    metric_type: MetricType,
):
    """
    Return metrics by type.
    """

    if metric_type == MetricType.COUNTER:

        return self.counters

    if metric_type == MetricType.GAUGE:

        return self.gauges

    if metric_type == MetricType.HISTOGRAM:

        return self.histograms

    return {}


# ======================================================
# Reset Metrics
# ======================================================

def reset(
    self,
) -> None:
    """
    Clear all metrics.
    """

    self.counters.clear()

    self.gauges.clear()

    self.histograms.clear()

    self.metadata.clear()

    self.touch()


# ======================================================
# Remove Metric
# ======================================================

def remove_metric(
    self,
    name: str,
) -> bool:
    """
    Remove metric.
    """

    if name in self.counters:

        del self.counters[

            name

        ]

        self.touch()

        return True

    if name in self.gauges:

        del self.gauges[

            name

        ]

        self.touch()

        return True

    if name in self.histograms:

        del self.histograms[

            name

        ]

        self.touch()

        return True

    return False


# ======================================================
# Metric Exists
# ======================================================

def exists(
    self,
    name: str,
) -> bool:
    """
    Check whether metric exists.
    """

    return (

        name

        in

        self.counters

        or

        name

        in

        self.gauges

        or

        name

        in

        self.histograms

    )


# ======================================================
# Total Observations
# ======================================================

@property
def total_observations(
    self,
) -> int:
    """
    Total histogram observations.
    """

    return sum(

        len(

            histogram.observations

        )

        for histogram

        in self.histograms.values()

    )


# ======================================================
# Has Metrics
# ======================================================

@property
def has_metrics(
    self,
) -> bool:
    """
    Whether manager contains metrics.
    """

    return (

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

    ) > 0

# ======================================================
# Background Collection
# ======================================================

def start(
    self,
    interval: int = 30,
) -> None:
    """
    Start background metric collection.
    """

    if getattr(

        self,

        "_running",

        False,

    ):

        return

    self._running = True

    self._interval = interval

    self._thread = threading.Thread(

        target=self._collector,

        daemon=True,

        name="MetricsCollector",

    )

    self._thread.start()

    logger.info(

        "Metrics collection started."

    )


# ======================================================
# Stop Collection
# ======================================================

def stop(
    self,
) -> None:
    """
    Stop background collection.
    """

    self._running = False

    logger.info(

        "Metrics collection stopped."

    )


# ======================================================
# Background Collector
# ======================================================

def _collector(
    self,
) -> None:
    """
    Background collection loop.
    """

    import time

    while self._running:

        try:

            self.touch()

        except Exception as exc:

            logger.exception(

                "Metrics collector failed: %s",

                exc,

            )

        time.sleep(

            self._interval

        )


# ======================================================
# Cleanup
# ======================================================

def cleanup(
    self,
) -> None:
    """
    Cleanup metrics manager.
    """

    self.stop()

    self.reset()

    logger.info(

        "Metrics manager cleaned."

    )


# ======================================================
# Refresh
# ======================================================

def refresh(
    self,
) -> None:
    """
    Refresh metrics timestamp.
    """

    self.touch()

    logger.info(

        "Metrics refreshed."

    )


# ======================================================
# Context Manager
# ======================================================

def __enter__(
    self,
):
    """
    Context manager entry.
    """

    self.start()

    return self


def __exit__(
    self,
    exc_type,
    exc_value,
    traceback,
):
    """
    Context manager exit.
    """

    self.cleanup()


# ======================================================
# String Representation
# ======================================================

def __repr__(
    self,
):
    """
    Developer representation.
    """

    return (

        "MetricsManager("

        f"id='{self.id}', "

        f"metrics={len(self)}"

        ")"

    )


# ======================================================
# Human Readable
# ======================================================

def __str__(
    self,
):
    """
    Human-readable string.
    """

    return (

        f"MetricsManager "

        f"({len(self)} metrics)"

    )


# ======================================================
# Length
# ======================================================

def __len__(
    self,
):
    """
    Total registered metrics.
    """

    return (

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

    )


# ======================================================
# Iterator
# ======================================================

def __iter__(
    self,
):
    """
    Iterate over all metrics.
    """

    for collection in (

        self.counters,

        self.gauges,

        self.histograms,

    ):

        for metric in collection.values():

            yield metric


# ======================================================
# Get Item
# ======================================================

def __getitem__(
    self,
    name: str,
):
    """
    Dictionary-style access.
    """

    if name in self.counters:

        return self.counters[

            name

        ]

    if name in self.gauges:

        return self.gauges[

            name

        ]

    if name in self.histograms:

        return self.histograms[

            name

        ]

    raise KeyError(

        name

    )


# ======================================================
# Contains
# ======================================================

def __contains__(
    self,
    name: str,
):
    """
    Metric existence.
    """

    return self.exists(

        name

    )


# ======================================================
# Boolean
# ======================================================

def __bool__(
    self,
):
    """
    Whether metrics exist.
    """

    return self.has_metrics


# ======================================================
# Singleton
# ======================================================

_metrics_manager: Optional[
    MetricsManager
] = None


def get_metrics_manager(
) -> MetricsManager:
    """
    Singleton metrics manager.
    """

    global _metrics_manager

    if _metrics_manager is None:

        _metrics_manager = (

            MetricsManager()

        )

    return _metrics_manager


# ======================================================
# Reset Singleton
# ======================================================

def reset_metrics_manager(
) -> None:
    """
    Reset singleton instance.
    """

    global _metrics_manager

    if _metrics_manager:

        _metrics_manager.cleanup()

    _metrics_manager = None


# ======================================================
# Factory Methods
# ======================================================

@classmethod
def development(
    cls,
) -> "MetricsManager":
    """
    Development configuration.
    """

    return cls()


@classmethod
def production(
    cls,
) -> "MetricsManager":
    """
    Production configuration.
    """

    return cls()


@classmethod
def testing(
    cls,
) -> "MetricsManager":
    """
    Testing configuration.
    """

    return cls()


# ======================================================
# Convenience Properties
# ======================================================

@property
def is_running(
    self,
) -> bool:
    """
    Background collection status.
    """

    return getattr(

        self,

        "_running",

        False,

    )


@property
def uptime_seconds(
    self,
) -> float:
    """
    Metrics manager uptime.
    """

    return (

        datetime.utcnow()

        -

        self.created_at

    ).total_seconds()


@property
def metric_count(
    self,
) -> int:
    """
    Total metrics.
    """

    return len(

        self

    )


# ======================================================
# Module Instance
# ======================================================

metrics = (

    get_metrics_manager()

)
