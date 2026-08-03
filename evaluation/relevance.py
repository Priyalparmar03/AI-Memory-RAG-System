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

# ======================================================
# Document Relevance
# ======================================================

def document_relevance(
    self,
    query: str,
    document: str,
    query_embedding: Optional[np.ndarray] = None,
    document_embedding: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute document relevance score.
    """

    keyword_score = self.keyword_overlap(

        query,

        document,

    )

    semantic_score = 0.0

    if (

        query_embedding is not None

        and

        document_embedding is not None

    ):

        semantic_score = self.semantic_similarity(

            query_embedding,

            document_embedding,

        )

    score = self.combined_similarity(

        keyword_score,

        semantic_score,

    )

    return {

        "score": score,

        "keyword_score": keyword_score,

        "semantic_score": semantic_score,

    }


# ======================================================
# Chunk Relevance
# ======================================================

def chunk_relevance(
    self,
    query: str,
    chunks: List[str],
) -> List[Dict[str, Any]]:
    """
    Evaluate relevance of all chunks.
    """

    results = []

    for index, chunk in enumerate(

        chunks,

        start=1,

    ):

        relevance = self.document_relevance(

            query,

            chunk,

        )

        relevance["chunk_id"] = index

        relevance["text"] = chunk

        results.append(

            relevance

        )

    return results


# ======================================================
# Query Relevance
# ======================================================

def query_relevance(
    self,
    query: str,
    documents: List[str],
) -> float:
    """
    Average relevance across all documents.
    """

    if not documents:

        return 0.0

    scores = [

        self.document_relevance(

            query,

            document,

        )["score"]

        for document in documents

    ]

    return round(

        sum(scores)

        /

        len(scores),

        4,

    )


# ======================================================
# Evaluate Single Document
# ======================================================

def evaluate_document(
    self,
    query: str,
    document: str,
) -> RelevanceResult:
    """
    Evaluate one document.
    """

    relevance = self.document_relevance(

        query,

        document,

    )

    result = RelevanceResult(

        score=relevance["score"],

        keyword_score=

            relevance["keyword_score"],

        semantic_score=

            relevance["semantic_score"],

        document_count=1,

        relevant_documents=

            1

            if relevance["score"]

            >=

            self.config.similarity_threshold

            else 0,

        metadata={

            "document": document,

        },

    )

    self.history.append(

        result

    )

    return result


# ======================================================
# Evaluate Retrieved Documents
# ======================================================

def evaluate_documents(
    self,
    query: str,
    documents: List[str],
) -> List[RelevanceResult]:
    """
    Evaluate multiple retrieved documents.
    """

    results = []

    for document in documents:

        results.append(

            self.evaluate_document(

                query,

                document,

            )

        )

    return results


# ======================================================
# Retrieval Score
# ======================================================

def retrieval_score(
    self,
    query: str,
    retrieved_documents: List[str],
) -> float:
    """
    Overall retrieval relevance.
    """

    return self.query_relevance(

        query,

        retrieved_documents,

    )


# ======================================================
# Rank Documents
# ======================================================

def rank_documents(
    self,
    query: str,
    documents: List[str],
) -> List[Dict[str, Any]]:
    """
    Rank documents by relevance.
    """

    ranked = []

    for document in documents:

        relevance = self.document_relevance(

            query,

            document,

        )

        ranked.append(

            {

                "document": document,

                "score":

                    relevance["score"],

                "keyword_score":

                    relevance["keyword_score"],

                "semantic_score":

                    relevance["semantic_score"],

            }

        )

    ranked.sort(

        key=lambda x: x["score"],

        reverse=True,

    )

    return ranked


# ======================================================
# Top-K Relevant Documents
# ======================================================

def top_k_documents(
    self,
    query: str,
    documents: List[str],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return Top-K relevant documents.
    """

    return self.rank_documents(

        query,

        documents,

    )[:k]


# ======================================================
# Relevant Documents
# ======================================================

def relevant_documents(
    self,
    query: str,
    documents: List[str],
) -> List[str]:
    """
    Return only relevant documents.
    """

    relevant = []

    for document in documents:

        score = self.document_relevance(

            query,

            document,

        )["score"]

        if (

            score

            >=

            self.config.similarity_threshold

        ):

            relevant.append(

                document

            )

    return relevant


# ======================================================
# Irrelevant Documents
# ======================================================

def irrelevant_documents(
    self,
    query: str,
    documents: List[str],
) -> List[str]:
    """
    Return only irrelevant documents.
    """

    irrelevant = []

    for document in documents:

        score = self.document_relevance(

            query,

            document,

        )["score"]

        if (

            score

            <

            self.config.similarity_threshold

        ):

            irrelevant.append(

                document

            )

    return irrelevant

# ======================================================
# Evaluate
# ======================================================

def evaluate(
    self,
    query: str,
    retrieved_documents: List[str],
) -> RelevanceResult:
    """
    Complete relevance evaluation.
    """

    if not retrieved_documents:

        result = RelevanceResult(

            score=0.0,

            keyword_score=0.0,

            semantic_score=0.0,

            document_count=0,

            relevant_documents=0,

            metadata={

                "documents": [],

            },

        )

        self.history.append(result)

        return result

    keyword_scores = []

    semantic_scores = []

    overall_scores = []

    relevant_count = 0

    ranked_documents = []

    for document in retrieved_documents:

        relevance = self.document_relevance(

            query,

            document,

        )

        keyword_scores.append(

            relevance["keyword_score"]

        )

        semantic_scores.append(

            relevance["semantic_score"]

        )

        overall_scores.append(

            relevance["score"]

        )

        if (

            relevance["score"]

            >=

            self.config.similarity_threshold

        ):

            relevant_count += 1

        ranked_documents.append(

            {

                "document": document,

                "score":

                    relevance["score"],

            }

        )

    result = RelevanceResult(

        score=round(

            sum(overall_scores)

            /

            len(overall_scores),

            4,

        ),

        keyword_score=round(

            sum(keyword_scores)

            /

            len(keyword_scores),

            4,

        ),

        semantic_score=round(

            sum(semantic_scores)

            /

            len(semantic_scores),

            4,

        ),

        document_count=len(

            retrieved_documents

        ),

        relevant_documents=relevant_count,

        metadata={

            "ranked_documents":

                ranked_documents,

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
    queries: List[str],
    retrieved_batches: List[List[str]],
) -> List[RelevanceResult]:
    """
    Evaluate multiple queries.
    """

    if len(queries) != len(retrieved_batches):

        raise RelevanceError(

            "Queries and retrieved "

            "documents size mismatch."

        )

    results = []

    for query, documents in zip(

        queries,

        retrieved_batches,

    ):

        results.append(

            self.evaluate(

                query,

                documents,

            )

        )

    return results


# ======================================================
# Compare Retrievers
# ======================================================

def compare_retrievers(
    self,
    query: str,
    retrievers: Dict[str, List[str]],
) -> Dict[str, RelevanceResult]:
    """
    Compare multiple retrievers.
    """

    comparison = {}

    for name, documents in retrievers.items():

        comparison[name] = self.evaluate(

            query,

            documents,

        )

    return comparison


# ======================================================
# Average Score
# ======================================================

def average_score(
    self,
) -> float:
    """
    Average relevance score.
    """

    if not self.history:

        return 0.0

    return round(

        sum(

            result.score

            for result in self.history

        )

        /

        len(self.history),

        4,

    )


# ======================================================
# Best Result
# ======================================================

def best_result(
    self,
) -> Optional[RelevanceResult]:
    """
    Best relevance result.
    """

    if not self.history:

        return None

    return max(

        self.history,

        key=lambda x: x.score,

    )


# ======================================================
# Worst Result
# ======================================================

def worst_result(
    self,
) -> Optional[RelevanceResult]:
    """
    Worst relevance result.
    """

    if not self.history:

        return None

    return min(

        self.history,

        key=lambda x: x.score,

    )


# ======================================================
# Latest Result
# ======================================================

def latest_result(
    self,
) -> Optional[RelevanceResult]:
    """
    Return latest evaluation.
    """

    if not self.history:

        return None

    return self.history[-1]


# ======================================================
# Relevance Distribution
# ======================================================

def relevance_distribution(
    self,
) -> Dict[str, int]:
    """
    Distribution of relevance scores.
    """

    distribution = {

        "high": 0,

        "medium": 0,

        "low": 0,

    }

    for result in self.history:

        if result.score >= 0.80:

            distribution["high"] += 1

        elif result.score >= 0.50:

            distribution["medium"] += 1

        else:

            distribution["low"] += 1

    return distribution


# ======================================================
# History
# ======================================================

def evaluation_history(
    self,
) -> List[RelevanceResult]:
    """
    Return evaluation history.
    """

    return list(

        self.history

    )


# ======================================================
# Export Results
# ======================================================

def export_results(
    self,
) -> List[Dict[str, Any]]:
    """
    Export evaluation results.
    """

    return [

        {

            "score":

                result.score,

            "keyword_score":

                result.keyword_score,

            "semantic_score":

                result.semantic_score,

            "document_count":

                result.document_count,

            "relevant_documents":

                result.relevant_documents,

            "metadata":

                result.metadata,

        }

        for result in self.history

    ]
