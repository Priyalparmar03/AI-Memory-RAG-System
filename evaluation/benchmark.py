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

# ======================================================
# Compare Models
# ======================================================

def compare_models(
    self,
    models: Dict[str, Any],
    dataset: BenchmarkDataset,
    retriever,
    relevance_evaluator=None,
    faithfulness_evaluator=None,
    hallucination_evaluator=None,
) -> Dict[str, BenchmarkStatistics]:
    """
    Compare multiple LLM models.
    """

    comparison = {}

    for model_name, generator in models.items():

        logger.info(
            "Benchmarking model: %s",
            model_name,
        )

        self.results.clear()

        stats = self.benchmark_dataset(

            dataset=dataset,

            retriever=retriever,

            generator=generator,

            relevance_evaluator=relevance_evaluator,

            faithfulness_evaluator=faithfulness_evaluator,

            hallucination_evaluator=hallucination_evaluator,

        )

        comparison[model_name] = stats

    return comparison


# ======================================================
# Compare Embedding Models
# ======================================================

def compare_embeddings(
    self,
    embedding_models: Dict[str, Any],
    dataset: BenchmarkDataset,
    retriever_factory,
    generator=None,
    relevance_evaluator=None,
    faithfulness_evaluator=None,
    hallucination_evaluator=None,
) -> Dict[str, BenchmarkStatistics]:
    """
    Compare embedding models.
    """

    comparison = {}

    for name, embedding_model in embedding_models.items():

        logger.info(
            "Embedding: %s",
            name,
        )

        retriever = retriever_factory(
            embedding_model
        )

        self.results.clear()

        stats = self.benchmark_dataset(

            dataset=dataset,

            retriever=retriever,

            generator=generator,

            relevance_evaluator=relevance_evaluator,

            faithfulness_evaluator=faithfulness_evaluator,

            hallucination_evaluator=hallucination_evaluator,

        )

        comparison[name] = stats

    return comparison


# ======================================================
# Compare Vector Stores
# ======================================================

def compare_vectorstores(
    self,
    vectorstores: Dict[str, Any],
    dataset: BenchmarkDataset,
    generator=None,
    relevance_evaluator=None,
    faithfulness_evaluator=None,
    hallucination_evaluator=None,
) -> Dict[str, BenchmarkStatistics]:
    """
    Compare vector databases.
    """

    comparison = {}

    for backend, retriever in vectorstores.items():

        logger.info(
            "VectorStore: %s",
            backend,
        )

        self.results.clear()

        stats = self.benchmark_dataset(

            dataset=dataset,

            retriever=retriever,

            generator=generator,

            relevance_evaluator=relevance_evaluator,

            faithfulness_evaluator=faithfulness_evaluator,

            hallucination_evaluator=hallucination_evaluator,

        )

        comparison[backend] = stats

    return comparison


# ======================================================
# Compare Retrievers
# ======================================================

def compare_retrievers(
    self,
    retrievers: Dict[str, Any],
    dataset: BenchmarkDataset,
    generator=None,
    relevance_evaluator=None,
    faithfulness_evaluator=None,
    hallucination_evaluator=None,
) -> Dict[str, BenchmarkStatistics]:
    """
    Compare retrieval algorithms.
    """

    comparison = {}

    for name, retriever in retrievers.items():

        logger.info(
            "Retriever: %s",
            name,
        )

        self.results.clear()

        stats = self.benchmark_dataset(

            dataset,

            retriever,

            generator,

            relevance_evaluator,

            faithfulness_evaluator,

            hallucination_evaluator,

        )

        comparison[name] = stats

    return comparison


# ======================================================
# Ranking
# ======================================================

def ranking(
    self,
    comparison: Dict[str, BenchmarkStatistics],
) -> List[tuple]:
    """
    Rank systems by average score.
    """

    ranked = sorted(

        comparison.items(),

        key=lambda x: x[1].average_score,

        reverse=True,

    )

    return ranked


# ======================================================
# Best System
# ======================================================

def best_system(
    self,
    comparison: Dict[str, BenchmarkStatistics],
) -> tuple:
    """
    Return best performing system.
    """

    ranking = self.ranking(
        comparison
    )

    return ranking[0]


# ======================================================
# Worst System
# ======================================================

def worst_system(
    self,
    comparison: Dict[str, BenchmarkStatistics],
) -> tuple:
    """
    Return worst performing system.
    """

    ranking = self.ranking(
        comparison
    )

    return ranking[-1]


# ======================================================
# Comparison Summary
# ======================================================

def comparison_summary(
    self,
    comparison: Dict[str, BenchmarkStatistics],
) -> List[Dict]:
    """
    Human-readable comparison.
    """

    summary = []

    for name, stats in comparison.items():

        summary.append(

            {

                "name": name,

                "queries": stats.total_queries,

                "average_score": stats.average_score,

                "average_latency": stats.average_latency,

            }

        )

    return summary

# ======================================================
# Generate Report
# ======================================================

def generate_report(
    self,
) -> Dict[str, Any]:
    """
    Generate benchmark report.
    """

    stats = self.statistics()

    return {

        "summary": {

            "queries": stats.total_queries,

            "average_score": stats.average_score,

            "average_latency": stats.average_latency,

            "best_score": stats.maximum_score,

            "worst_score": stats.minimum_score,

        },

        "results": [

            {

                "query": r.query,

                "retrieval_score": r.retrieval_score,

                "relevance_score": r.relevance_score,

                "faithfulness_score": r.faithfulness_score,

                "hallucination_score": r.hallucination_score,

                "latency_ms": r.latency_ms,

                "overall_score": r.overall_score,

            }

            for r in self.results

        ],

    }


# ======================================================
# Export JSON
# ======================================================

def export_json(
    self,
    path: str,
) -> None:
    """
    Export report as JSON.
    """

    with open(

        path,

        "w",

        encoding="utf8",

    ) as file:

        json.dump(

            self.generate_report(),

            file,

            indent=4,

        )

    logger.info(

        "JSON report exported: %s",

        path,

    )


# ======================================================
# Export CSV
# ======================================================

def export_csv(
    self,
    path: str,
) -> None:
    """
    Export benchmark results as CSV.
    """

    with open(

        path,

        "w",

        newline="",

        encoding="utf8",

    ) as file:

        writer = csv.writer(file)

        writer.writerow(

            [

                "query",

                "retrieval_score",

                "relevance_score",

                "faithfulness_score",

                "hallucination_score",

                "latency_ms",

                "overall_score",

            ]

        )

        for result in self.results:

            writer.writerow(

                [

                    result.query,

                    result.retrieval_score,

                    result.relevance_score,

                    result.faithfulness_score,

                    result.hallucination_score,

                    result.latency_ms,

                    result.overall_score,

                ]

            )

    logger.info(

        "CSV report exported: %s",

        path,

    )


# ======================================================
# Save Reports
# ======================================================

def save_reports(
    self,
) -> None:
    """
    Automatically save reports.
    """

    if not self.config.save_reports:

        return

    directory = Path(

        self.config.report_directory

    )

    directory.mkdir(

        parents=True,

        exist_ok=True,

    )

    if self.config.export_json:

        self.export_json(

            str(

                directory /

                "benchmark_report.json"

            )

        )

    if self.config.export_csv:

        self.export_csv(

            str(

                directory /

                "benchmark_report.csv"

            )

        )


# ======================================================
# Diagnostics
# ======================================================

def diagnostics(
    self,
) -> Dict[str, Any]:
    """
    Benchmark diagnostics.
    """

    return {

        "configuration": self.config,

        "statistics": self.statistics(),

        "summary": self.benchmark_summary(),

        "history_size": len(self.results),

    }


# ======================================================
# Runtime Statistics
# ======================================================

def runtime_statistics(
    self,
) -> Dict[str, Any]:
    """
    Runtime statistics.
    """

    if not self.results:

        return {

            "queries": 0,

            "throughput_qps": 0,

        }

    total_time = sum(

        r.latency_ms

        for r in self.results

    ) / 1000

    throughput = (

        len(self.results)

        / total_time

        if total_time > 0

        else 0

    )

    return {

        "queries": len(self.results),

        "total_runtime_seconds": round(

            total_time,

            3,

        ),

        "throughput_qps": round(

            throughput,

            2,

        ),

    }


# ======================================================
# Benchmark History
# ======================================================

def history(
    self,
) -> List[BenchmarkResult]:
    """
    Return benchmark history.
    """

    return list(

        self.results

    )


# ======================================================
# Cleanup
# ======================================================

def cleanup(
    self,
) -> None:
    """
    Cleanup benchmark runner.
    """

    self.results.clear()

    logger.info(

        "Benchmark runner cleaned."

    )


# ======================================================
# Context Manager
# ======================================================

def __enter__(self):

    return self


def __exit__(
    self,
    exc_type,
    exc_value,
    traceback,
):

    self.cleanup()


# ======================================================
# Python Protocols
# ======================================================

def __len__(
    self,
):

    return len(

        self.results

    )


def __iter__(
    self,
):

    return iter(

        self.results

    )


def __repr__(
    self,
):

    return (

        "BenchmarkRunner("

        f"queries={len(self.results)}, "

        f"average_score={self.statistics().average_score:.3f}"

        ")"

    )
