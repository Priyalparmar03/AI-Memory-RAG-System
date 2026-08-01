from __future__ import annotations

import csv
import json
import logging
import statistics
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ==========================================================
# Exception
# ==========================================================

class BenchmarkError(Exception):
    """Benchmark exception."""


# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class BenchmarkConfig:

    iterations: int = 1

    top_k: int = 5

    save_reports: bool = True

    report_directory: str = "./benchmark_reports"

    export_json: bool = True

    export_csv: bool = True

# ==========================================================
# Dataset
# ==========================================================

@dataclass(slots=True)
class BenchmarkQuery:

    query: str

    expected_answer: str

    expected_documents: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkDataset:

    name: str

    queries: List[BenchmarkQuery]

# ==========================================================
# Result
# ==========================================================

@dataclass(slots=True)
class BenchmarkResult:

    query: str

    retrieval_score: float

    relevance_score: float

    faithfulness_score: float

    hallucination_score: float

    latency_ms: float

    overall_score: float

    metadata: Dict[str, Any] = field(default_factory=dict)

# ==========================================================
# Statistics
# ==========================================================

@dataclass(slots=True)
class BenchmarkStatistics:

    total_queries: int

    average_score: float

    average_latency: float

    minimum_score: float

    maximum_score: float

# ==========================================================
# Benchmark Runner
# ==========================================================

class BenchmarkRunner:

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
    ):

        self.config = config or BenchmarkConfig()

        self.results: List[BenchmarkResult] = []

        logger.info(

            "BenchmarkRunner initialized."

        )

      # ======================================================
    # JSON Dataset
    # ======================================================

    def load_json(
        self,
        path: str,
    ) -> BenchmarkDataset:

        with open(path, "r", encoding="utf8") as file:

            data = json.load(file)

        queries = [

            BenchmarkQuery(

                query=item["query"],

                expected_answer=item["expected_answer"],

                expected_documents=item.get(

                    "expected_documents",

                    [],

                ),

                metadata=item.get(

                    "metadata",

                    {},

                ),

            )

            for item in data["queries"]

        ]

        return BenchmarkDataset(

            name=data["name"],

            queries=queries,

        )

      # ======================================================
    # CSV Dataset
    # ======================================================

    def load_csv(
        self,
        path: str,
    ) -> BenchmarkDataset:

        queries = []

        with open(

            path,

            newline="",

            encoding="utf8",

        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                queries.append(

                    BenchmarkQuery(

                        query=row["query"],

                        expected_answer=row["expected_answer"],

                    )

                )

        return BenchmarkDataset(

            name=Path(path).stem,

            queries=queries,

        )

      # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(
        self,
        path: str,
    ) -> BenchmarkDataset:

        if path.endswith(".json"):

            return self.load_json(path)

        if path.endswith(".csv"):

            return self.load_csv(path)

        raise BenchmarkError(

            "Unsupported dataset."

        )

      # ======================================================
    # Validation
    # ======================================================

    def validate_dataset(
        self,
        dataset: BenchmarkDataset,
    ) -> bool:

        if not dataset.queries:

            return False

        for query in dataset.queries:

            if not query.query:

                return False

        return True


# ======================================================
# Benchmark Single Query
# ======================================================

def benchmark_query(
    self,
    query: BenchmarkQuery,
    retriever,
    generator=None,
    relevance_evaluator=None,
    faithfulness_evaluator=None,
    hallucination_evaluator=None,
) -> BenchmarkResult:
    """
    Benchmark a single query.
    """

    start = time.perf_counter()

    retrieved_documents = retriever.retrieve(
        query.query,
        top_k=self.config.top_k,
    )

    generated_answer = ""

    if generator is not None:

        generated_answer = generator.generate(
            query=query.query,
            context=retrieved_documents,
        )

    latency = (
        time.perf_counter() - start
    ) * 1000

    retrieval_score = 0.0

    relevance_score = 0.0

    faithfulness_score = 0.0

    hallucination_score = 0.0

    if relevance_evaluator:

        relevance_score = relevance_evaluator.evaluate(

            query=query.query,

            retrieved_documents=retrieved_documents,

            expected_documents=query.expected_documents,

        )

        retrieval_score = relevance_score

    if faithfulness_evaluator:

        faithfulness_score = faithfulness_evaluator.evaluate(

            answer=generated_answer,

            context=retrieved_documents,

        )

    if hallucination_evaluator:

        hallucination_score = hallucination_evaluator.evaluate(

            answer=generated_answer,

            context=retrieved_documents,

        )

    scores = [

        retrieval_score,

        relevance_score,

    ]

    if generator is not None:

        scores.extend(

            [

                faithfulness_score,

                1.0 - hallucination_score,

            ]

        )

    overall = round(

        sum(scores) / len(scores),

        4,

    )

    result = BenchmarkResult(

        query=query.query,

        retrieval_score=retrieval_score,

        relevance_score=relevance_score,

        faithfulness_score=faithfulness_score,

        hallucination_score=hallucination_score,

        latency_ms=round(

            latency,

            2,

        ),

        overall_score=overall,

        metadata={

            "generated_answer": generated_answer,

            "retrieved_documents": retrieved_documents,

        },

    )

    self.results.append(result)

    return result


# ======================================================
# Benchmark Batch
# ======================================================

def benchmark_batch(
    self,
    queries: List[BenchmarkQuery],
    retriever,
    generator=None,
    relevance_evaluator=None,
    faithfulness_evaluator=None,
    hallucination_evaluator=None,
) -> List[BenchmarkResult]:
    """
    Benchmark multiple queries.
    """

    results = []

    total = len(queries)

    for index, query in enumerate(queries, start=1):

        logger.info(

            "Benchmarking %d/%d",

            index,

            total,

        )

        results.append(

            self.benchmark_query(

                query=query,

                retriever=retriever,

                generator=generator,

                relevance_evaluator=relevance_evaluator,

                faithfulness_evaluator=faithfulness_evaluator,

                hallucination_evaluator=hallucination_evaluator,

            )

        )

    return results


# ======================================================
# Benchmark Dataset
# ======================================================

def benchmark_dataset(
    self,
    dataset: BenchmarkDataset,
    retriever,
    generator=None,
    relevance_evaluator=None,
    faithfulness_evaluator=None,
    hallucination_evaluator=None,
) -> BenchmarkStatistics:
    """
    Benchmark an entire dataset.
    """

    if not self.validate_dataset(dataset):

        raise BenchmarkError(

            "Invalid benchmark dataset."

        )

    self.results.clear()

    self.benchmark_batch(

        queries=dataset.queries,

        retriever=retriever,

        generator=generator,

        relevance_evaluator=relevance_evaluator,

        faithfulness_evaluator=faithfulness_evaluator,

        hallucination_evaluator=hallucination_evaluator,

    )

    return self.statistics()


# ======================================================
# Clear Results
# ======================================================

def clear_results(
    self,
) -> None:
    """
    Clear benchmark history.
    """

    self.results.clear()


# ======================================================
# Latest Result
# ======================================================

def latest_result(
    self,
) -> Optional[BenchmarkResult]:
    """
    Return the latest benchmark result.
    """

    if not self.results:

        return None

    return self.results[-1]


# ======================================================
# Statistics
# ======================================================

def statistics(
    self,
) -> BenchmarkStatistics:
    """
    Compute benchmark statistics.
    """

    if not self.results:

        return BenchmarkStatistics(

            total_queries=0,

            average_score=0.0,

            average_latency=0.0,

            minimum_score=0.0,

            maximum_score=0.0,

        )

    scores = [

        result.overall_score

        for result in self.results

    ]

    latencies = [

        result.latency_ms

        for result in self.results

    ]

    return BenchmarkStatistics(

        total_queries=len(self.results),

        average_score=round(

            statistics.mean(scores),

            4,

        ),

        average_latency=round(

            statistics.mean(latencies),

            2,

        ),

        minimum_score=min(scores),

        maximum_score=max(scores),

    )


# ======================================================
# Summary
# ======================================================

def benchmark_summary(
    self,
) -> Dict[str, Any]:
    """
    Human-readable benchmark summary.
    """

    stats = self.statistics()

    return {

        "queries": stats.total_queries,

        "average_score": stats.average_score,

        "average_latency_ms": stats.average_latency,

        "best_score": stats.maximum_score,

        "worst_score": stats.minimum_score,

    }


# ======================================================
# Success Rate
# ======================================================

def success_rate(
    self,
    threshold: float = 0.75,
) -> float:
    """
    Percentage of queries whose overall score
    meets or exceeds the threshold.
    """

    if not self.results:

        return 0.0

    successful = sum(

        1

        for result in self.results

        if result.overall_score >= threshold

    )

    return round(

        successful / len(self.results),

        4,

    )
