from __future__ import annotations

import logging
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class RelevanceError(Exception):
    """Relevance evaluation exception."""


# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class RelevanceConfig:

    similarity_threshold: float = 0.75

    keyword_weight: float = 0.40

    semantic_weight: float = 0.60

    lowercase: bool = True

    remove_punctuation: bool = True

    normalize_whitespace: bool = True


# ==========================================================
# Result
# ==========================================================

@dataclass(slots=True)
class RelevanceResult:

    score: float

    keyword_score: float

    semantic_score: float

    document_count: int

    relevant_documents: int

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==========================================================
# Statistics
# ==========================================================

@dataclass(slots=True)
class RelevanceStatistics:

    average_score: float

    minimum_score: float

    maximum_score: float

    evaluations: int


# ==========================================================
# Relevance Evaluator
# ==========================================================

class RelevanceEvaluator:

    def __init__(
        self,
        config: Optional[
            RelevanceConfig
        ] = None,
    ):

        self.config = (

            config

            or

            RelevanceConfig()

        )

        self.history: List[
            RelevanceResult
        ] = []

        logger.info(

            "RelevanceEvaluator initialized."

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
        Tokenize text.
        """

        text = self.normalize(text)

        if not text:

            return []

        return text.split()

    # ======================================================
    # Query Preprocessing
    # ======================================================

    def preprocess_query(
        self,
        query: str,
    ) -> str:
        """
        Normalize query.
        """

        return self.normalize(query)

    # ======================================================
    # Keyword Overlap
    # ======================================================

    def keyword_overlap(
        self,
        query: str,
        document: str,
    ) -> float:
        """
        Keyword overlap score.
        """

        query_tokens = set(

            self.tokenize(query)

        )

        document_tokens = set(

            self.tokenize(document)

        )

        if not query_tokens:

            return 0.0

        overlap = len(

            query_tokens

            &

            document_tokens

        )

        return round(

            overlap

            /

            len(query_tokens),

            4,

        )

    # ======================================================
    # Jaccard Similarity
    # ======================================================

    def jaccard_similarity(
        self,
        query: str,
        document: str,
    ) -> float:
        """
        Jaccard similarity.
        """

        query_tokens = set(

            self.tokenize(query)

        )

        document_tokens = set(

            self.tokenize(document)

        )

        union = (

            query_tokens

            |

            document_tokens

        )

        if not union:

            return 0.0

        intersection = (

            query_tokens

            &

            document_tokens

        )

        return round(

            len(intersection)

            /

            len(union),

            4,

        )

    # ======================================================
    # Semantic Similarity
    # ======================================================

    def semantic_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Cosine similarity between embeddings.
        """

        embedding1 = np.asarray(

            embedding1

        ).reshape(

            1,

            -1,

        )

        embedding2 = np.asarray(

            embedding2

        ).reshape(

            1,

            -1,

        )

        score = cosine_similarity(

            embedding1,

            embedding2,

        )[0][0]

        return round(

            float(score),

            4,

        )

    # ======================================================
    # Combined Similarity
    # ======================================================

    def combined_similarity(
        self,
        keyword_score: float,
        semantic_score: float,
    ) -> float:
        """
        Weighted relevance score.
        """

        score = (

            self.config.keyword_weight

            * keyword_score

            +

            self.config.semantic_weight

            * semantic_score

        )

        return round(

            score,

            4,

        )

    # ======================================================
    # Query Summary
    # ======================================================

    def query_summary(
        self,
        query: str,
        documents: List[str],
    ) -> Dict[str, Any]:
        """
        Query statistics.
        """

        return {

            "query_tokens": len(

                self.tokenize(query)

            ),

            "documents": len(

                documents

            ),

            "average_document_tokens":

                round(

                    sum(

                        len(

                            self.tokenize(doc)

                        )

                        for doc in documents

                    )

                    /

                    max(

                        1,

                        len(documents),

                    ),

                    2,

                ),

        }
