from __future__ import annotations

import logging
import statistics
import time

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class LatencyError(Exception):
    """Latency evaluation exception."""


# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class LatencyConfig:

    warmup_runs: int = 1

    benchmark_runs: int = 10

    monitor_cpu: bool = True

    monitor_memory: bool = True

    monitor_tokens: bool = True

    high_latency_threshold: float = 1000.0

    warning_latency_threshold: float = 500.0


# ==========================================================
# Pipeline Timing
# ==========================================================

@dataclass(slots=True)
class PipelineTiming:

    preprocessing: float = 0.0

    chunking: float = 0.0

    embedding: float = 0.0

    retrieval: float = 0.0

    vectorstore: float = 0.0

    reranking: float = 0.0

    generation: float = 0.0

    total: float = 0.0


# ==========================================================
# Result
# ==========================================================

@dataclass(slots=True)
class LatencyResult:

    total_latency: float

    throughput: float

    queries_per_second: float

    tokens_per_second: float

    cpu_usage: float

    memory_usage: float

    pipeline: PipelineTiming

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==========================================================
# Statistics
# ==========================================================

@dataclass(slots=True)
class LatencyStatistics:

    average_latency: float

    minimum_latency: float

    maximum_latency: float

    average_qps: float

    average_tokens_per_second: float


# ==========================================================
# Latency Evaluator
# ==========================================================

class LatencyEvaluator:

    def __init__(
        self,
        config: Optional[
            LatencyConfig
        ] = None,
    ):

        self.config = (

            config

            or

            LatencyConfig()

        )

        self.history: List[
            LatencyResult
        ] = []

        self._timers: Dict[
            str,
            float
        ] = {}

        logger.info(

            "LatencyEvaluator initialized."

        )

    # ======================================================
    # Start Timer
    # ======================================================

    def start_timer(
        self,
        name: str,
    ) -> None:
        """
        Start a named timer.
        """

        self._timers[name] = time.perf_counter()

    # ======================================================
    # Stop Timer
    # ======================================================

    def stop_timer(
        self,
        name: str,
    ) -> float:
        """
        Stop timer and return elapsed time (ms).
        """

        if name not in self._timers:

            raise LatencyError(

                f"Timer '{name}' "

                "was never started."

            )

        elapsed = (

            time.perf_counter()

            -

            self._timers.pop(name)

        ) * 1000

        return round(

            elapsed,

            3,

        )

    # ======================================================
    # Elapsed
    # ======================================================

    def elapsed(
        self,
        name: str,
    ) -> float:
        """
        Current elapsed time without stopping timer.
        """

        if name not in self._timers:

            return 0.0

        return round(

            (

                time.perf_counter()

                -

                self._timers[name]

            )

            * 1000,

            3,

        )

    # ======================================================
    # Timer Exists
    # ======================================================

    def timer_exists(
        self,
        name: str,
    ) -> bool:

        return name in self._timers

    # ======================================================
    # Active Timers
    # ======================================================

    def active_timers(
        self,
    ) -> List[str]:

        return list(

            self._timers.keys()

        )

    # ======================================================
    # Reset Timer
    # ======================================================

    def reset_timer(
        self,
        name: str,
    ) -> None:

        self._timers.pop(

            name,

            None,

        )

    # ======================================================
    # Reset All Timers
    # ======================================================

    def reset_all(
        self,
    ) -> None:

        self._timers.clear()

    # ======================================================
    # CPU Usage
    # ======================================================

    def cpu_usage(
        self,
    ) -> float:

        if not self.config.monitor_cpu:

            return 0.0

        return psutil.cpu_percent()

    # ======================================================
    # Memory Usage
    # ======================================================

    def memory_usage(
        self,
    ) -> float:

        if not self.config.monitor_memory:

            return 0.0

        process = psutil.Process()

        return round(

            process.memory_info().rss

            / (1024 ** 2),

            2,

        )

    # ======================================================
    # Warmup
    # ======================================================

    def warmup(
        self,
        function,
        *args,
        **kwargs,
    ) -> None:
        """
        Warm up a function before benchmarking.
        """

        for _ in range(

            self.config.warmup_runs

        ):

            function(

                *args,

                **kwargs,

            )
