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
