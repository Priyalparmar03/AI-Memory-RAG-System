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

# ======================================================
# Measure Function
# ======================================================

def measure(
    self,
    function,
    *args,
    **kwargs,
) -> tuple[Any, float]:
    """
    Measure execution time of any function.
    Returns:
        (result, latency_ms)
    """

    start = time.perf_counter()

    result = function(
        *args,
        **kwargs,
    )

    latency = (

        time.perf_counter()

        - start

    ) * 1000

    return (

        result,

        round(latency, 3),

    )


# ======================================================
# Preprocessing Latency
# ======================================================

def preprocessing_latency(
    self,
    preprocessor,
    document: str,
) -> tuple[Any, float]:
    """
    Measure preprocessing latency.
    """

    return self.measure(

        preprocessor.preprocess,

        document,

    )


# ======================================================
# Chunking Latency
# ======================================================

def chunking_latency(
    self,
    chunker,
    document: str,
) -> tuple[Any, float]:
    """
    Measure chunking latency.
    """

    return self.measure(

        chunker.chunk,

        document,

    )


# ======================================================
# Embedding Latency
# ======================================================

def embedding_latency(
    self,
    embedder,
    texts: List[str],
) -> tuple[Any, float]:
    """
    Measure embedding generation latency.
    """

    return self.measure(

        embedder.embed,

        texts,

    )


# ======================================================
# Retrieval Latency
# ======================================================

def retrieval_latency(
    self,
    retriever,
    query: str,
    top_k: int = 5,
) -> tuple[Any, float]:
    """
    Measure retrieval latency.
    """

    return self.measure(

        retriever.retrieve,

        query,

        top_k,

    )


# ======================================================
# VectorStore Search Latency
# ======================================================

def vectorstore_latency(
    self,
    vectorstore,
    query_embedding,
    top_k: int = 5,
) -> tuple[Any, float]:
    """
    Measure vector database search latency.
    """

    return self.measure(

        vectorstore.search,

        query_embedding,

        top_k,

    )


# ======================================================
# Re-ranking Latency
# ======================================================

def reranking_latency(
    self,
    reranker,
    query: str,
    documents: List[str],
) -> tuple[Any, float]:
    """
    Measure reranker latency.
    """

    return self.measure(

        reranker.rerank,

        query,

        documents,

    )


# ======================================================
# LLM Generation Latency
# ======================================================

def generation_latency(
    self,
    llm,
    prompt: str,
) -> tuple[Any, float]:
    """
    Measure LLM response latency.
    """

    return self.measure(

        llm.generate,

        prompt,

    )


# ======================================================
# Total Pipeline Latency
# ======================================================

def total_latency(
    self,
    pipeline: PipelineTiming,
) -> float:
    """
    Calculate total pipeline latency.
    """

    pipeline.total = round(

        pipeline.preprocessing

        +

        pipeline.chunking

        +

        pipeline.embedding

        +

        pipeline.retrieval

        +

        pipeline.vectorstore

        +

        pipeline.reranking

        +

        pipeline.generation,

        3,

    )

    return pipeline.total


# ======================================================
# Create Pipeline Timing
# ======================================================

def create_pipeline(
    self,
    preprocessing: float = 0.0,
    chunking: float = 0.0,
    embedding: float = 0.0,
    retrieval: float = 0.0,
    vectorstore: float = 0.0,
    reranking: float = 0.0,
    generation: float = 0.0,
) -> PipelineTiming:
    """
    Create a PipelineTiming object.
    """

    pipeline = PipelineTiming(

        preprocessing=preprocessing,

        chunking=chunking,

        embedding=embedding,

        retrieval=retrieval,

        vectorstore=vectorstore,

        reranking=reranking,

        generation=generation,

    )

    pipeline.total = self.total_latency(
        pipeline
    )

    return pipeline


# ======================================================
# Stage Breakdown
# ======================================================

def pipeline_breakdown(
    self,
    pipeline: PipelineTiming,
) -> Dict[str, float]:
    """
    Percentage contribution of each stage.
    """

    total = max(

        pipeline.total,

        1e-9,

    )

    return {

        "preprocessing":

            round(

                pipeline.preprocessing

                / total * 100,

                2,

            ),

        "chunking":

            round(

                pipeline.chunking

                / total * 100,

                2,

            ),

        "embedding":

            round(

                pipeline.embedding

                / total * 100,

                2,

            ),

        "retrieval":

            round(

                pipeline.retrieval

                / total * 100,

                2,

            ),

        "vectorstore":

            round(

                pipeline.vectorstore

                / total * 100,

                2,

            ),

        "reranking":

            round(

                pipeline.reranking

                / total * 100,

                2,

            ),

        "generation":

            round(

                pipeline.generation

                / total * 100,

                2,

            ),

    }


# ======================================================
# Slowest Stage
# ======================================================

def slowest_stage(
    self,
    pipeline: PipelineTiming,
) -> tuple[str, float]:
    """
    Return the slowest pipeline stage.
    """

    stages = {

        "preprocessing":

            pipeline.preprocessing,

        "chunking":

            pipeline.chunking,

        "embedding":

            pipeline.embedding,

        "retrieval":

            pipeline.retrieval,

        "vectorstore":

            pipeline.vectorstore,

        "reranking":

            pipeline.reranking,

        "generation":

            pipeline.generation,

    }

    stage = max(

        stages,

        key=stages.get,

    )

    return (

        stage,

        stages[stage],

    )


# ======================================================
# Fastest Stage
# ======================================================

def fastest_stage(
    self,
    pipeline: PipelineTiming,
) -> tuple[str, float]:
    """
    Return the fastest pipeline stage.
    """

    stages = {

        "preprocessing":

            pipeline.preprocessing,

        "chunking":

            pipeline.chunking,

        "embedding":

            pipeline.embedding,

        "retrieval":

            pipeline.retrieval,

        "vectorstore":

            pipeline.vectorstore,

        "reranking":

            pipeline.reranking,

        "generation":

            pipeline.generation,

    }

    stage = min(

        stages,

        key=stages.get,

    )

    return (

        stage,

        stages[stage],

    )


# ======================================================
# Throughput
# ======================================================

def throughput(
    self,
    processed_items: int,
    total_time_ms: float,
) -> float:
    """
    Items processed per second.
    """

    if total_time_ms <= 0:

        return 0.0

    return round(

        processed_items

        /

        (total_time_ms / 1000),

        2,

    )


# ======================================================
# Queries Per Second
# ======================================================

def queries_per_second(
    self,
    queries: int,
    total_time_ms: float,
) -> float:
    """
    Queries processed per second.
    """

    return self.throughput(

        queries,

        total_time_ms,

    )


# ======================================================
# Tokens Per Second
# ======================================================

def tokens_per_second(
    self,
    total_tokens: int,
    total_time_ms: float,
) -> float:
    """
    Token generation speed.
    """

    if total_time_ms <= 0:

        return 0.0

    return round(

        total_tokens

        /

        (total_time_ms / 1000),

        2,

    )


# ======================================================
# Resource Usage
# ======================================================

def resource_usage(
    self,
) -> Dict[str, float]:
    """
    Current CPU and memory usage.
    """

    return {

        "cpu_percent":

            self.cpu_usage(),

        "memory_mb":

            self.memory_usage(),

    }


# ======================================================
# Evaluate Pipeline
# ======================================================

def evaluate(
    self,
    pipeline: PipelineTiming,
    processed_queries: int = 1,
    generated_tokens: int = 0,
) -> LatencyResult:
    """
    Evaluate pipeline latency.
    """

    total = self.total_latency(

        pipeline

    )

    result = LatencyResult(

        total_latency=total,

        throughput=self.throughput(

            processed_queries,

            total,

        ),

        queries_per_second=

            self.queries_per_second(

                processed_queries,

                total,

            ),

        tokens_per_second=

            self.tokens_per_second(

                generated_tokens,

                total,

            ),

        cpu_usage=

            self.cpu_usage(),

        memory_usage=

            self.memory_usage(),

        pipeline=pipeline,

        metadata={

            "breakdown":

                self.pipeline_breakdown(

                    pipeline

                ),

            "slowest_stage":

                self.slowest_stage(

                    pipeline

                ),

            "fastest_stage":

                self.fastest_stage(

                    pipeline

                ),

        },

    )

    self.history.append(

        result

    )

    return result


# ======================================================
# Batch Evaluate
# ======================================================

def batch_evaluate(
    self,
    pipelines: List[PipelineTiming],
    generated_tokens: Optional[List[int]] = None,
) -> List[LatencyResult]:
    """
    Evaluate multiple pipeline runs.
    """

    if generated_tokens is None:

        generated_tokens = [

            0

        ] * len(pipelines)

    if len(pipelines) != len(generated_tokens):

        raise LatencyError(

            "Pipeline and token count "

            "must have same length."

        )

    results = []

    for pipeline, tokens in zip(

        pipelines,

        generated_tokens,

    ):

        results.append(

            self.evaluate(

                pipeline,

                processed_queries=1,

                generated_tokens=tokens,

            )

        )

    return results


# ======================================================
# Average Latency
# ======================================================

def average_latency(
    self,
) -> float:
    """
    Average total latency.
    """

    if not self.history:

        return 0.0

    return round(

        statistics.mean(

            result.total_latency

            for result in self.history

        ),

        3,

    )


# ======================================================
# Best Result
# ======================================================

def best_result(
    self,
) -> Optional[LatencyResult]:
    """
    Lowest latency result.
    """

    if not self.history:

        return None

    return min(

        self.history,

        key=lambda r: r.total_latency,

    )


# ======================================================
# Worst Result
# ======================================================

def worst_result(
    self,
) -> Optional[LatencyResult]:
    """
    Highest latency result.
    """

    if not self.history:

        return None

    return max(

        self.history,

        key=lambda r: r.total_latency,

    )


# ======================================================
# High Latency Detection
# ======================================================

def high_latency(
    self,
    result: LatencyResult,
) -> bool:
    """
    Detect high latency.
    """

    return (

        result.total_latency

        >=

        self.config.high_latency_threshold

    )


# ======================================================
# Warning Latency Detection
# ======================================================

def warning_latency(
    self,
    result: LatencyResult,
) -> bool:
    """
    Detect warning latency.
    """

    return (

        result.total_latency

        >=

        self.config.warning_latency_threshold

    )


# ======================================================
# Performance Grade
# ======================================================

def performance_grade(
    self,
    result: LatencyResult,
) -> str:
    """
    Assign performance grade.
    """

    latency = result.total_latency

    if latency <= 100:

        return "A+"

    if latency <= 250:

        return "A"

    if latency <= 500:

        return "B"

    if latency <= 1000:

        return "C"

    if latency <= 2000:

        return "D"

    return "F"


# ======================================================
# Performance Summary
# ======================================================

def performance_summary(
    self,
    result: LatencyResult,
) -> Dict[str, Any]:
    """
    Human-readable performance report.
    """

    return {

        "latency_ms":

            result.total_latency,

        "throughput":

            result.throughput,

        "queries_per_second":

            result.queries_per_second,

        "tokens_per_second":

            result.tokens_per_second,

        "cpu_percent":

            result.cpu_usage,

        "memory_mb":

            result.memory_usage,

        "grade":

            self.performance_grade(

                result

            ),

        "slowest_stage":

            result.metadata[

                "slowest_stage"

            ],

        "fastest_stage":

            result.metadata[

                "fastest_stage"

            ],

    }

