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

# ======================================================
# Precision@K
# ======================================================

def precision_at_k(
    self,
    retrieved: List[Any],
    relevant: List[Any],
    k: int = 5,
) -> float:
    """
    Precision@K
    """

    if k <= 0:

        raise MetricError(

            "k must be positive."

        )

    retrieved = retrieved[:k]

    if not retrieved:

        return 0.0

    hits = sum(

        1

        for item in retrieved

        if item in relevant

    )

    score = hits / len(retrieved)

    self.save_metric(

        "precision@k",

        score,

        k=k,

    )

    return round(

        score,

        4,

    )


# ======================================================
# Recall@K
# ======================================================

def recall_at_k(
    self,
    retrieved: List[Any],
    relevant: List[Any],
    k: int = 5,
) -> float:
    """
    Recall@K
    """

    if not relevant:

        return 0.0

    retrieved = retrieved[:k]

    hits = sum(

        1

        for item in retrieved

        if item in relevant

    )

    score = hits / len(relevant)

    self.save_metric(

        "recall@k",

        score,

        k=k,

    )

    return round(

        score,

        4,

    )


# ======================================================
# F1 Score
# ======================================================

def f1_score(
    self,
    precision: float,
    recall: float,
) -> float:
    """
    F1 Score.
    """

    denominator = (

        precision

        +

        recall

    )

    if denominator == 0:

        return 0.0

    score = (

        2

        * precision

        * recall

    ) / denominator

    self.save_metric(

        "f1",

        score,

    )

    return round(

        score,

        4,

    )


# ======================================================
# Mean Reciprocal Rank (MRR)
# ======================================================

def mrr(
    self,
    retrieved: List[Any],
    relevant: List[Any],
) -> float:
    """
    Mean Reciprocal Rank for a
    single query.
    """

    for index, item in enumerate(

        retrieved,

        start=1,

    ):

        if item in relevant:

            score = 1.0 / index

            self.save_metric(

                "mrr",

                score,

            )

            return round(

                score,

                4,

            )

    return 0.0


# ======================================================
# Mean Reciprocal Rank Batch
# ======================================================

def mean_reciprocal_rank(
    self,
    retrieved_lists: List[List[Any]],
    relevant_lists: List[List[Any]],
) -> float:
    """
    Average MRR across queries.
    """

    if len(retrieved_lists) != len(relevant_lists):

        raise MetricError(

            "List size mismatch."

        )

    scores = [

        self.mrr(

            retrieved,

            relevant,

        )

        for retrieved, relevant

        in zip(

            retrieved_lists,

            relevant_lists,

        )

    ]

    if not scores:

        return 0.0

    return round(

        sum(scores)

        /

        len(scores),

        4,

    )


# ======================================================
# Discounted Cumulative Gain
# ======================================================

def dcg(
    self,
    relevance_scores: List[float],
    k: int = 5,
) -> float:
    """
    Discounted Cumulative Gain.
    """

    relevance_scores = relevance_scores[:k]

    score = 0.0

    for index, relevance in enumerate(

        relevance_scores,

        start=1,

    ):

        score += (

            relevance

            /

            math.log2(

                index + 1

            )

        )

    return score


# ======================================================
# Normalized DCG
# ======================================================

def ndcg(
    self,
    relevance_scores: List[float],
    k: int = 5,
) -> float:
    """
    Normalized DCG.
    """

    dcg = self.dcg(

        relevance_scores,

        k,

    )

    ideal = sorted(

        relevance_scores,

        reverse=True,

    )

    idcg = self.dcg(

        ideal,

        k,

    )

    if idcg == 0:

        return 0.0

    score = dcg / idcg

    self.save_metric(

        "ndcg",

        score,

        k=k,

    )

    return round(

        score,

        4,

    )


# ======================================================
# Hit Rate@K
# ======================================================

def hit_rate_at_k(
    self,
    retrieved: List[Any],
    relevant: List[Any],
    k: int = 5,
) -> float:
    """
    Hit Rate@K.
    """

    retrieved = retrieved[:k]

    score = any(

        item in relevant

        for item in retrieved

    )

    value = 1.0 if score else 0.0

    self.save_metric(

        "hit_rate",

        value,

        k=k,

    )

    return value


# ======================================================
# Average Precision
# ======================================================

def average_precision(
    self,
    retrieved: List[Any],
    relevant: List[Any],
) -> float:
    """
    Average Precision.
    """

    if not relevant:

        return 0.0

    hits = 0

    precision_sum = 0.0

    for index, item in enumerate(

        retrieved,

        start=1,

    ):

        if item in relevant:

            hits += 1

            precision_sum += (

                hits

                / index

            )

    if hits == 0:

        return 0.0

    score = (

        precision_sum

        /

        len(relevant)

    )

    self.save_metric(

        "average_precision",

        score,

    )

    return round(

        score,

        4,

    )


# ======================================================
# Mean Average Precision (MAP)
# ======================================================

def mean_average_precision(
    self,
    retrieved_lists: List[List[Any]],
    relevant_lists: List[List[Any]],
) -> float:
    """
    MAP over multiple queries.
    """

    if len(retrieved_lists) != len(relevant_lists):

        raise MetricError(

            "List size mismatch."

        )

    scores = [

        self.average_precision(

            retrieved,

            relevant,

        )

        for retrieved, relevant

        in zip(

            retrieved_lists,

            relevant_lists,

        )

    ]

    if not scores:

        return 0.0

    return round(

        sum(scores)

        /

        len(scores),

        4,

    )

# ======================================================
# BLEU Score
# ======================================================

def bleu_score(
    self,
    reference: str,
    candidate: str,
) -> float:
    """
    Compute BLEU score.
    """

    try:

        from sacrebleu.metrics import BLEU

        bleu = BLEU()

        score = bleu.sentence_score(

            candidate,

            [reference],

        ).score / 100.0

    except ImportError:

        raise MetricError(

            "Install 'sacrebleu' package."

        )

    self.save_metric(

        "bleu",

        score,

    )

    return round(

        score,

        4,

    )


# ======================================================
# ROUGE Score
# ======================================================

def rouge_score(
    self,
    reference: str,
    candidate: str,
) -> Dict[str, float]:
    """
    Compute ROUGE scores.
    """

    try:

        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(

            [

                "rouge1",

                "rouge2",

                "rougeL",

            ],

            use_stemmer=True,

        )

        scores = scorer.score(

            reference,

            candidate,

        )

    except ImportError:

        raise MetricError(

            "Install 'rouge-score' package."

        )

    result = {

        metric: round(

            value.fmeasure,

            4,

        )

        for metric, value

        in scores.items()

    }

    self.save_metric(

        "rouge",

        result["rougeL"],

    )

    return result


# ======================================================
# Exact Match
# ======================================================

def exact_match(
    self,
    reference: str,
    candidate: str,
) -> float:
    """
    Exact Match score.
    """

    score = (

        self.normalize(reference)

        ==

        self.normalize(candidate)

    )

    value = 1.0 if score else 0.0

    self.save_metric(

        "exact_match",

        value,

    )

    return value


# ======================================================
# Accuracy
# ======================================================

def accuracy(
    self,
    y_true: List[Any],
    y_pred: List[Any],
) -> float:
    """
    Classification accuracy.
    """

    if len(y_true) != len(y_pred):

        raise MetricError(

            "Length mismatch."

        )

    correct = sum(

        1

        for truth, pred

        in zip(

            y_true,

            y_pred,

        )

        if truth == pred

    )

    score = correct / len(y_true)

    self.save_metric(

        "accuracy",

        score,

    )

    return round(

        score,

        4,

    )


# ======================================================
# Precision
# ======================================================

def precision(
    self,
    tp: int,
    fp: int,
) -> float:
    """
    Precision.
    """

    denominator = tp + fp

    if denominator == 0:

        return 0.0

    score = tp / denominator

    self.save_metric(

        "precision",

        score,

    )

    return round(

        score,

        4,

    )


# ======================================================
# Recall
# ======================================================

def recall(
    self,
    tp: int,
    fn: int,
) -> float:
    """
    Recall.
    """

    denominator = tp + fn

    if denominator == 0:

        return 0.0

    score = tp / denominator

    self.save_metric(

        "recall",

        score,

    )

    return round(

        score,

        4,

    )


# ======================================================
# Specificity
# ======================================================

def specificity(
    self,
    tn: int,
    fp: int,
) -> float:
    """
    Specificity.
    """

    denominator = tn + fp

    if denominator == 0:

        return 0.0

    score = tn / denominator

    self.save_metric(

        "specificity",

        score,

    )

    return round(

        score,

        4,

    )


# ======================================================
# Embedding Similarity
# ======================================================

def embedding_similarity(
    self,
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> float:
    """
    Cosine similarity between embeddings.
    """

    score = self.cosine_similarity_score(

        embedding1,

        embedding2,

    )

    self.save_metric(

        "embedding_similarity",

        score,

    )

    return score


# ======================================================
# Mean Squared Error
# ======================================================

def mean_squared_error(
    self,
    y_true: List[float],
    y_pred: List[float],
) -> float:
    """
    Mean Squared Error.
    """

    if len(y_true) != len(y_pred):

        raise MetricError(

            "Length mismatch."

        )

    score = np.mean(

        (

            np.asarray(y_true)

            -

            np.asarray(y_pred)

        ) ** 2

    )

    self.save_metric(

        "mse",

        float(score),

    )

    return round(

        float(score),

        4,

    )


# ======================================================
# Root Mean Squared Error
# ======================================================

def root_mean_squared_error(
    self,
    y_true: List[float],
    y_pred: List[float],
) -> float:
    """
    Root Mean Squared Error.
    """

    mse = self.mean_squared_error(

        y_true,

        y_pred,

    )

    score = math.sqrt(mse)

    self.save_metric(

        "rmse",

        score,

    )

    return round(

        score,

        4,

    )


# ======================================================
# Mean Absolute Error
# ======================================================

def mean_absolute_error(
    self,
    y_true: List[float],
    y_pred: List[float],
) -> float:
    """
    Mean Absolute Error.
    """

    if len(y_true) != len(y_pred):

        raise MetricError(

            "Length mismatch."

        )

    score = np.mean(

        np.abs(

            np.asarray(y_true)

            -

            np.asarray(y_pred)

        )

    )

    self.save_metric(

        "mae",

        float(score),

    )

    return round(

        float(score),

        4,

    )
