from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class MetricError(Exception):
    """Metric calculation exception."""


# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class MetricConfig:

    lowercase: bool = True

    remove_punctuation: bool = True

    normalize_whitespace: bool = True

    epsilon: float = 1e-9


# ==========================================================
# Metric Result
# ==========================================================

@dataclass(slots=True)
class MetricResult:

    name: str

    score: float

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==========================================================
# Metric Calculator
# ==========================================================

class MetricCalculator:

    def __init__(
        self,
        config: Optional[
            MetricConfig
        ] = None,
    ):

        self.config = (

            config

            or

            MetricConfig()

        )

        self.history: List[
            MetricResult
        ] = []

        logger.info(

            "MetricCalculator initialized."

        )

    # ======================================================
    # Normalize
    # ======================================================

    def normalize(
        self,
        text: str,
    ) -> str:
        """
        Normalize text.
        """

        if self.config.lowercase:

            text = text.lower()

        if self.config.remove_punctuation:

            text = re.sub(

                r"[^\w\s]",

                "",

                text,

            )

        if self.config.normalize_whitespace:

            text = re.sub(

                r"\s+",

                " ",

                text,

            )

        return text.strip()

    # ======================================================
    # Tokenize
    # ======================================================

    def tokenize(
        self,
        text: str,
    ) -> List[str]:
        """
        Simple whitespace tokenizer.
        """

        text = self.normalize(text)

        if not text:

            return []

        return text.split()

    # ======================================================
    # Overlap Score
    # ======================================================

    def overlap_score(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Token overlap score.
        """

        tokens1 = set(

            self.tokenize(text1)

        )

        tokens2 = set(

            self.tokenize(text2)

        )

        if not tokens1:

            return 0.0

        overlap = len(

            tokens1 & tokens2

        )

        return round(

            overlap

            /

            len(tokens1),

            4,

        )

    # ======================================================
    # Jaccard Similarity
    # ======================================================

    def jaccard_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Jaccard similarity.
        """

        tokens1 = set(

            self.tokenize(text1)

        )

        tokens2 = set(

            self.tokenize(text2)

        )

        union = tokens1 | tokens2

        if not union:

            return 0.0

        intersection = (

            tokens1

            &

            tokens2

        )

        return round(

            len(intersection)

            /

            len(union),

            4,

        )

    # ======================================================
    # Cosine Similarity
    # ======================================================

    def cosine_similarity_score(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray,
    ) -> float:
        """
        Cosine similarity between vectors.
        """

        vector1 = np.asarray(

            vector1

        ).reshape(

            1,

            -1,

        )

        vector2 = np.asarray(

            vector2

        ).reshape(

            1,

            -1,

        )

        score = cosine_similarity(

            vector1,

            vector2,

        )[0][0]

        return round(

            float(score),

            4,

        )

    # ======================================================
    # Euclidean Distance
    # ======================================================

    def euclidean_distance(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray,
    ) -> float:
        """
        Euclidean distance.
        """

        vector1 = np.asarray(

            vector1

        )

        vector2 = np.asarray(

            vector2

        )

        distance = np.linalg.norm(

            vector1

            -

            vector2

        )

        return round(

            float(distance),

            4,

        )

    # ======================================================
    # Manhattan Distance
    # ======================================================

    def manhattan_distance(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray,
    ) -> float:
        """
        Manhattan distance.
        """

        vector1 = np.asarray(

            vector1

        )

        vector2 = np.asarray(

            vector2

        )

        distance = np.sum(

            np.abs(

                vector1

                -

                vector2

            )

        )

        return round(

            float(distance),

            4,

        )

    # ======================================================
    # Dot Product
    # ======================================================

    def dot_product(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray,
    ) -> float:
        """
        Dot product similarity.
        """

        return round(

            float(

                np.dot(

                    vector1,

                    vector2,

                )

            ),

            4,

        )

    # ======================================================
    # Save Metric
    # ======================================================

    def save_metric(
        self,
        name: str,
        score: float,
        **metadata,
    ) -> MetricResult:
        """
        Save metric result.
        """

        result = MetricResult(

            name=name,

            score=score,

            metadata=metadata,

        )

        self.history.append(

            result

        )

        return result
