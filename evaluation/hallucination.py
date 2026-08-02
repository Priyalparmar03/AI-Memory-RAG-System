from __future__ import annotations

import logging
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import nltk
import spacy

logger = logging.getLogger(__name__)

# ==========================================================
# Download Punkt (if needed)
# ==========================================================

try:

    nltk.data.find("tokenizers/punkt")

except LookupError:

    nltk.download("punkt")


# ==========================================================
# Load spaCy Model
# ==========================================================

try:

    NLP = spacy.load("en_core_web_sm")

except OSError:

    logger.warning(

        "spaCy model 'en_core_web_sm' not found."

    )

    NLP = None


# ==========================================================
# Exception
# ==========================================================

class HallucinationError(Exception):
    """Hallucination evaluation exception."""


# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class HallucinationConfig:

    similarity_threshold: float = 0.75

    ignore_case: bool = True

    normalize_whitespace: bool = True

    extract_entities: bool = True

    extract_numbers: bool = True

    extract_dates: bool = True


# ==========================================================
# Result
# ==========================================================

@dataclass(slots=True)
class HallucinationResult:

    score: float

    unsupported_claims: int

    entity_errors: int

    numeric_errors: int

    date_errors: int

    contradictions: int

    metadata: Dict[str, Any] = field(default_factory=dict)


# ==========================================================
# Hallucination Evaluator
# ==========================================================

class HallucinationEvaluator:

    def __init__(
        self,
        config: Optional[
            HallucinationConfig
        ] = None,
    ):

        self.config = (

            config

            or

            HallucinationConfig()

        )

        self.history: List[
            HallucinationResult
        ] = []

        logger.info(

            "HallucinationEvaluator initialized."

        )

    # ======================================================
    # Normalize Text
    # ======================================================

    def normalize(
        self,
        text: str,
    ) -> str:

        if self.config.ignore_case:

            text = text.lower()

        if self.config.normalize_whitespace:

            text = re.sub(

                r"\s+",

                " ",

                text,

            )

        return text.strip()

    # ======================================================
    # Extract Claims
    # ======================================================

    def extract_claims(
        self,
        answer: str,
    ) -> List[str]:
        """
        One sentence = one claim.
        """

        answer = self.normalize(answer)

        claims = nltk.sent_tokenize(answer)

        return [

            claim.strip()

            for claim in claims

            if claim.strip()

        ]

    # ======================================================
    # Extract Named Entities
    # ======================================================

    def extract_entities(
        self,
        text: str,
    ) -> List[str]:

        if NLP is None:

            return []

        document = NLP(text)

        return [

            entity.text

            for entity in document.ents

        ]

    # ======================================================
    # Extract Numbers
    # ======================================================

    def extract_numbers(
        self,
        text: str,
    ) -> List[str]:

        return re.findall(

            r"\b\d+(?:\.\d+)?\b",

            text,

        )

    # ======================================================
    # Extract Dates
    # ======================================================

    def extract_dates(
        self,
        text: str,
    ) -> List[str]:

        if NLP is None:

            return []

        document = NLP(text)

        return [

            entity.text

            for entity in document.ents

            if entity.label_ == "DATE"

        ]

    # ======================================================
    # Context Text
    # ======================================================

    def context_text(
        self,
        context: List[str],
    ) -> str:

        return self.normalize(

            " ".join(context)

        )

    # ======================================================
    # Context Claims
    # ======================================================

    def context_claims(
        self,
        context: List[str],
    ) -> List[str]:

        claims = []

        for chunk in context:

            claims.extend(

                nltk.sent_tokenize(

                    self.normalize(chunk)

                )

            )

        return claims

    # ======================================================
    # Summary
    # ======================================================

    def summary(
        self,
        answer: str,
        context: List[str],
    ) -> Dict[str, Any]:

        return {

            "claims": len(

                self.extract_claims(

                    answer

                )

            ),

            "entities": len(

                self.extract_entities(

                    answer

                )

            ),

            "numbers": len(

                self.extract_numbers(

                    answer

                )

            ),

            "dates": len(

                self.extract_dates(

                    answer

                )

            ),

            "context_sentences": len(

                self.context_claims(

                    context

                )

            ),

        }

# ======================================================
# Detect Unsupported Claims
# ======================================================

def detect_unsupported_claims(
    self,
    answer: str,
    context: List[str],
) -> List[Dict[str, Any]]:
    """
    Detect claims that are not supported
    by retrieved context.
    """

    context_text = self.context_text(context)

    unsupported = []

    for claim in self.extract_claims(answer):

        normalized = self.normalize(claim)

        supported = normalized in context_text

        if not supported:

            unsupported.append(

                {

                    "claim": claim,

                    "reason": "No supporting evidence",

                }

            )

    return unsupported


# ======================================================
# Detect Entity Hallucination
# ======================================================

def detect_entity_hallucination(
    self,
    answer: str,
    context: List[str],
) -> List[str]:
    """
    Detect entities present in the answer
    but missing from the retrieved context.
    """

    answer_entities = set(

        entity.lower()

        for entity in self.extract_entities(answer)

    )

    context_entities = set(

        entity.lower()

        for entity in self.extract_entities(

            self.context_text(context)

        )

    )

    return sorted(

        answer_entities - context_entities

    )


# ======================================================
# Detect Numeric Hallucination
# ======================================================

def detect_numeric_hallucination(
    self,
    answer: str,
    context: List[str],
) -> List[str]:
    """
    Detect numbers appearing only in the answer.
    """

    answer_numbers = set(

        self.extract_numbers(answer)

    )

    context_numbers = set(

        self.extract_numbers(

            self.context_text(context)

        )

    )

    return sorted(

        answer_numbers - context_numbers

    )


# ======================================================
# Detect Date Hallucination
# ======================================================

def detect_date_hallucination(
    self,
    answer: str,
    context: List[str],
) -> List[str]:
    """
    Detect dates appearing only in the answer.
    """

    answer_dates = set(

        date.lower()

        for date in self.extract_dates(answer)

    )

    context_dates = set(

        date.lower()

        for date in self.extract_dates(

            self.context_text(context)

        )

    )

    return sorted(

        answer_dates - context_dates

    )


# ======================================================
# Detect Contradictions
# ======================================================

def detect_contradictions(
    self,
    answer: str,
    context: List[str],
) -> List[Dict[str, Any]]:
    """
    Very simple contradiction detection.
    """

    contradictions = []

    context_text = self.context_text(context)

    negative_words = {

        "not",

        "never",

        "none",

        "no",

        "without",

        "cannot",

    }

    for claim in self.extract_claims(answer):

        normalized = self.normalize(claim)

        if normalized not in context_text:

            continue

        claim_negative = any(

            word in normalized.split()

            for word in negative_words

        )

        context_negative = any(

            word in context_text.split()

            for word in negative_words

        )

        if claim_negative != context_negative:

            contradictions.append(

                {

                    "claim": claim,

                    "reason": "Possible contradiction",

                }

            )

    return contradictions


# ======================================================
# Detect Missing Evidence
# ======================================================

def detect_missing_evidence(
    self,
    answer: str,
    context: List[str],
) -> List[str]:
    """
    Return claims without evidence.
    """

    unsupported = self.detect_unsupported_claims(

        answer,

        context,

    )

    return [

        item["claim"]

        for item in unsupported

    ]


# ======================================================
# Entity Coverage
# ======================================================

def entity_coverage(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Percentage of answer entities supported
    by the context.
    """

    answer_entities = set(

        entity.lower()

        for entity in self.extract_entities(answer)

    )

    if not answer_entities:

        return 1.0

    hallucinated = set(

        self.detect_entity_hallucination(

            answer,

            context,

        )

    )

    supported = (

        len(answer_entities)

        - len(hallucinated)

    )

    return round(

        supported

        /

        len(answer_entities),

        4,

    )


# ======================================================
# Numeric Coverage
# ======================================================

def numeric_coverage(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Percentage of answer numbers supported
    by the context.
    """

    answer_numbers = set(

        self.extract_numbers(answer)

    )

    if not answer_numbers:

        return 1.0

    hallucinated = set(

        self.detect_numeric_hallucination(

            answer,

            context,

        )

    )

    supported = (

        len(answer_numbers)

        - len(hallucinated)

    )

    return round(

        supported

        /

        len(answer_numbers),

        4,

    )


# ======================================================
# Date Coverage
# ======================================================

def date_coverage(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Percentage of answer dates supported
    by the context.
    """

    answer_dates = set(

        date.lower()

        for date in self.extract_dates(answer)

    )

    if not answer_dates:

        return 1.0

    hallucinated = set(

        self.detect_date_hallucination(

            answer,

            context,

        )

    )

    supported = (

        len(answer_dates)

        - len(hallucinated)

    )

    return round(

        supported

        /

        len(answer_dates),

        4,

    )


# ======================================================
# Detection Summary
# ======================================================

def detection_summary(
    self,
    answer: str,
    context: List[str],
) -> Dict[str, Any]:
    """
    Overall hallucination summary.
    """

    unsupported = self.detect_unsupported_claims(

        answer,

        context,

    )

    entities = self.detect_entity_hallucination(

        answer,

        context,

    )

    numbers = self.detect_numeric_hallucination(

        answer,

        context,

    )

    dates = self.detect_date_hallucination(

        answer,

        context,

    )

    contradictions = self.detect_contradictions(

        answer,

        context,

    )

    return {

        "unsupported_claims": len(unsupported),

        "entity_errors": len(entities),

        "numeric_errors": len(numbers),

        "date_errors": len(dates),

        "contradictions": len(contradictions),

        "entity_coverage": self.entity_coverage(

            answer,

            context,

        ),

        "numeric_coverage": self.numeric_coverage(

            answer,

            context,

        ),

        "date_coverage": self.date_coverage(

            answer,

            context,

        ),

    }

# ======================================================
# Hallucination Score
# ======================================================

def hallucination_score(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Overall hallucination score.
    Lower hallucination -> Higher score.
    """

    summary = self.detection_summary(
        answer,
        context,
    )

    total_errors = (

        summary["unsupported_claims"]

        +

        summary["entity_errors"]

        +

        summary["numeric_errors"]

        +

        summary["date_errors"]

        +

        summary["contradictions"]

    )

    total_claims = max(

        1,

        len(

            self.extract_claims(answer)

        ),

    )

    score = 1.0 - (

        total_errors

        /

        total_claims

    )

    return round(

        max(0.0, score),

        4,

    )


# ======================================================
# Factual Consistency
# ======================================================

def factual_consistency(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Estimate factual consistency.
    """

    summary = self.detection_summary(

        answer,

        context,

    )

    scores = [

        summary["entity_coverage"],

        summary["numeric_coverage"],

        summary["date_coverage"],

    ]

    return round(

        sum(scores)

        /

        len(scores),

        4,

    )


# ======================================================
# Evidence Coverage
# ======================================================

def evidence_coverage(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Percentage of claims backed by evidence.
    """

    unsupported = self.detect_unsupported_claims(

        answer,

        context,

    )

    claims = self.extract_claims(answer)

    if not claims:

        return 1.0

    supported = len(claims) - len(unsupported)

    return round(

        supported

        /

        len(claims),

        4,

    )


# ======================================================
# Confidence Score
# ======================================================

def confidence_score(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Confidence based on multiple metrics.
    """

    return round(

        (

            self.hallucination_score(

                answer,

                context,

            )

            +

            self.factual_consistency(

                answer,

                context,

            )

            +

            self.evidence_coverage(

                answer,

                context,

            )

        )

        /

        3,

        4,

    )


# ======================================================
# Evaluate
# ======================================================

def evaluate(
    self,
    answer: str,
    context: List[str],
) -> HallucinationResult:
    """
    Complete hallucination evaluation.
    """

    unsupported = self.detect_unsupported_claims(

        answer,

        context,

    )

    entity_errors = self.detect_entity_hallucination(

        answer,

        context,

    )

    numeric_errors = self.detect_numeric_hallucination(

        answer,

        context,

    )

    date_errors = self.detect_date_hallucination(

        answer,

        context,

    )

    contradictions = self.detect_contradictions(

        answer,

        context,

    )

    result = HallucinationResult(

        score=self.hallucination_score(

            answer,

            context,

        ),

        unsupported_claims=len(

            unsupported

        ),

        entity_errors=len(

            entity_errors

        ),

        numeric_errors=len(

            numeric_errors

        ),

        date_errors=len(

            date_errors

        ),

        contradictions=len(

            contradictions

        ),

        metadata={

            "unsupported": unsupported,

            "entity_errors": entity_errors,

            "numeric_errors": numeric_errors,

            "date_errors": date_errors,

            "contradictions": contradictions,

            "factual_consistency":

                self.factual_consistency(

                    answer,

                    context,

                ),

            "evidence_coverage":

                self.evidence_coverage(

                    answer,

                    context,

                ),

            "confidence":

                self.confidence_score(

                    answer,

                    context,

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
    answers: List[str],
    contexts: List[List[str]],
) -> List[HallucinationResult]:
    """
    Evaluate multiple answers.
    """

    if len(answers) != len(contexts):

        raise HallucinationError(

            "Answers and contexts "

            "must have equal length."

        )

    results = []

    for answer, context in zip(

        answers,

        contexts,

    ):

        results.append(

            self.evaluate(

                answer,

                context,

            )

        )

    return results


# ======================================================
# Average Score
# ======================================================

def average_score(
    self,
) -> float:
    """
    Average hallucination score.
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
) -> Optional[HallucinationResult]:
    """
    Best evaluation.
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
) -> Optional[HallucinationResult]:
    """
    Worst evaluation.
    """

    if not self.history:

        return None

    return min(

        self.history,

        key=lambda x: x.score,

    )

# ======================================================
# Statistics
# ======================================================

def statistics(
    self,
) -> Dict[str, Any]:
    """
    Overall hallucination statistics.
    """

    if not self.history:

        return {

            "evaluations": 0,

            "average_score": 0.0,

            "entity_errors": 0,

            "numeric_errors": 0,

            "date_errors": 0,

            "unsupported_claims": 0,

            "contradictions": 0,

        }

    scores = [

        result.score

        for result in self.history

    ]

    return {

        "evaluations": len(

            self.history

        ),

        "average_score": round(

            sum(scores)

            /

            len(scores),

            4,

        ),

        "entity_errors": sum(

            r.entity_errors

            for r in self.history

        ),

        "numeric_errors": sum(

            r.numeric_errors

            for r in self.history

        ),

        "date_errors": sum(

            r.date_errors

            for r in self.history

        ),

        "unsupported_claims": sum(

            r.unsupported_claims

            for r in self.history

        ),

        "contradictions": sum(

            r.contradictions

            for r in self.history

        ),

    }


# ======================================================
# Diagnostics
# ======================================================

def diagnostics(
    self,
) -> Dict[str, Any]:
    """
    Evaluator diagnostics.
    """

    return {

        "configuration": {

            "similarity_threshold":
                self.config.similarity_threshold,

            "ignore_case":
                self.config.ignore_case,

            "normalize_whitespace":
                self.config.normalize_whitespace,

            "extract_entities":
                self.config.extract_entities,

            "extract_numbers":
                self.config.extract_numbers,

            "extract_dates":
                self.config.extract_dates,

        },

        "statistics":

            self.statistics(),

        "history_size":

            len(self.history),

    }


# ======================================================
# Benchmark
# ======================================================

def benchmark(
    self,
    answers: List[str],
    contexts: List[List[str]],
) -> Dict[str, Any]:
    """
    Benchmark evaluator performance.
    """

    import time

    start = time.perf_counter()

    results = self.batch_evaluate(

        answers,

        contexts,

    )

    elapsed = (

        time.perf_counter()

        -

        start

    )

    return {

        "evaluations":

            len(results),

        "execution_time":

            round(

                elapsed,

                4,

            ),

        "evaluations_per_second":

            round(

                len(results)

                /

                max(

                    elapsed,

                    1e-9,

                ),

                2,

            ),

        "average_score":

            self.average_score(),

    }


# ======================================================
# Summary
# ======================================================

def summary(
    self,
) -> Dict[str, Any]:
    """
    Human-readable summary.
    """

    stats = self.statistics()

    return {

        "evaluations":

            stats["evaluations"],

        "average_score":

            stats["average_score"],

        "unsupported_claims":

            stats["unsupported_claims"],

        "entity_errors":

            stats["entity_errors"],

        "numeric_errors":

            stats["numeric_errors"],

        "date_errors":

            stats["date_errors"],

        "contradictions":

            stats["contradictions"],

    }


# ======================================================
# Latest Result
# ======================================================

def latest_result(
    self,
) -> Optional[HallucinationResult]:
    """
    Return latest evaluation.
    """

    if not self.history:

        return None

    return self.history[-1]


# ======================================================
# Clear History
# ======================================================

def clear_history(
    self,
) -> None:
    """
    Clear evaluation history.
    """

    self.history.clear()


# ======================================================
# Export Results
# ======================================================

def export_results(
    self,
) -> List[Dict[str, Any]]:
    """
    Export all evaluations.
    """

    return [

        {

            "score":

                result.score,

            "unsupported_claims":

                result.unsupported_claims,

            "entity_errors":

                result.entity_errors,

            "numeric_errors":

                result.numeric_errors,

            "date_errors":

                result.date_errors,

            "contradictions":

                result.contradictions,

            "metadata":

                result.metadata,

        }

        for result in self.history

    ]


# ======================================================
# Cleanup
# ======================================================

def cleanup(
    self,
) -> None:
    """
    Cleanup evaluator resources.
    """

    self.clear_history()

    logger.info(

        "HallucinationEvaluator cleaned."

    )


# ======================================================
# Context Manager
# ======================================================

def __enter__(
    self,
):

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

        self.history

    )


def __iter__(
    self,
):

    return iter(

        self.history

    )


def __repr__(
    self,
):

    stats = self.statistics()

    return (

        "HallucinationEvaluator("

        f"evaluations={stats['evaluations']}, "

        f"average_score={stats['average_score']}, "

        f"threshold={self.config.similarity_threshold}"

        ")"

    )
