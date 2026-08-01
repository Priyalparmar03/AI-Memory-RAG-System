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


