from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import nltk

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ==========================================================
# Exception
# ==========================================================

class FaithfulnessError(Exception):
    """Faithfulness evaluation exception."""

# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class FaithfulnessConfig:

    similarity_threshold: float = 0.75

    minimum_claim_length: int = 5

    ignore_case: bool = True

    remove_duplicates: bool = True

    normalize_whitespace: bool = True


# ==========================================================
# Result
# ==========================================================

@dataclass(slots=True)
class FaithfulnessResult:

    score: float

    groundedness: float

    evidence_score: float

    supported_claims: int

    unsupported_claims: int

    total_claims: int

    metadata: Dict[str, Any] = field(default_factory=dict)

# ==========================================================
# Evaluator
# ==========================================================

class FaithfulnessEvaluator:

    def __init__(
        self,
        config: Optional[
            FaithfulnessConfig
        ] = None,
    ):

        self.config = (
            config
            or
            FaithfulnessConfig()
        )

        self.history: List[
            FaithfulnessResult
        ] = []

        logger.info(
            "FaithfulnessEvaluator initialized."
        )

      # ======================================================
    # Normalize
    # ======================================================

    def normalize_context(
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
    # Sentence Extraction
    # ======================================================

    def extract_sentences(
        self,
        text: str,
    ) -> List[str]:

        text = self.normalize_context(
            text
        )

        sentences = nltk.sent_tokenize(
            text
        )

        return [

            s.strip()

            for s in sentences

            if s.strip()

        ]

      # ======================================================
    # Claim Extraction
    # ======================================================

    def extract_claims(
        self,
        answer: str,
    ) -> List[str]:
        """
        Extract factual claims from answer.

        Current implementation:
        one sentence = one claim.
        """

        claims = self.extract_sentences(
            answer
        )

        claims = [

            claim

            for claim in claims

            if len(
                claim.split()
            )
            >= self.config.minimum_claim_length

        ]

        if self.config.remove_duplicates:

            claims = list(
                dict.fromkeys(claims)
            )

        return claims


    # ======================================================
    # Context Sentences
    # ======================================================

    def context_sentences(
        self,
        context: List[str],
    ) -> List[str]:

        sentences = []

        for chunk in context:

            sentences.extend(

                self.extract_sentences(
                    chunk
                )

            )

        return sentences

      # ======================================================
    # Summary
    # ======================================================

    def summary(
        self,
        answer: str,
        context: List[str],
    ) -> Dict[str, Any]:

        claims = self.extract_claims(
            answer
        )

        evidence = self.context_sentences(
            context
        )

        return {

            "claims": len(claims),

            "context_sentences": len(
                evidence
            ),

        }

  # ======================================================
# Verify Single Claim
# ======================================================

def verify_claim(
    self,
    claim: str,
    context: List[str],
) -> Dict[str, Any]:
    """
    Verify whether a claim is supported
    by the retrieved context.
    """

    claim = self.normalize_context(claim)

    evidence = self.context_sentences(context)

    best_sentence = ""
    best_overlap = 0.0

    claim_words = set(claim.split())

    for sentence in evidence:

        sentence = self.normalize_context(sentence)

        sentence_words = set(sentence.split())

        if not sentence_words:
            continue

        overlap = len(

            claim_words & sentence_words

        ) / max(

            1,

            len(claim_words),

        )

        if overlap > best_overlap:

            best_overlap = overlap

            best_sentence = sentence

    supported = (

        best_overlap

        >=

        self.config.similarity_threshold

    )

    return {

        "claim": claim,

        "supported": supported,

        "score": round(

            best_overlap,

            4,

        ),

        "evidence": best_sentence,

    }


# ======================================================
# Verify All Claims
# ======================================================

def verify_all_claims(
    self,
    answer: str,
    context: List[str],
) -> List[Dict[str, Any]]:
    """
    Verify every claim in the answer.
    """

    claims = self.extract_claims(answer)

    return [

        self.verify_claim(

            claim,

            context,

        )

        for claim in claims

    ]


# ======================================================
# Supported Claims
# ======================================================

def supported_claims(
    self,
    answer: str,
    context: List[str],
) -> List[Dict[str, Any]]:
    """
    Return only supported claims.
    """

    return [

        result

        for result in self.verify_all_claims(

            answer,

            context,

        )

        if result["supported"]

    ]


# ======================================================
# Unsupported Claims
# ======================================================

def unsupported_claims(
    self,
    answer: str,
    context: List[str],
) -> List[Dict[str, Any]]:
    """
    Return unsupported claims.
    """

    return [

        result

        for result in self.verify_all_claims(

            answer,

            context,

        )

        if not result["supported"]

    ]


# ======================================================
# Context Overlap
# ======================================================

def context_overlap(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Average overlap between answer claims
    and retrieved context.
    """

    results = self.verify_all_claims(

        answer,

        context,

    )

    if not results:

        return 0.0

    return round(

        sum(

            r["score"]

            for r in results

        )

        /

        len(results),

        4,

    )


# ======================================================
# Evidence Sentences
# ======================================================

def evidence_sentences(
    self,
    answer: str,
    context: List[str],
) -> List[str]:
    """
    Return supporting evidence sentences.
    """

    evidence = []

    for result in self.verify_all_claims(

        answer,

        context,

    ):

        if result["supported"]:

            evidence.append(

                result["evidence"]

            )

    return evidence


# ======================================================
# Citation Coverage
# ======================================================

def citation_coverage(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Percentage of claims supported by context.
    """

    verified = self.verify_all_claims(

        answer,

        context,

    )

    if not verified:

        return 0.0

    supported = sum(

        1

        for result in verified

        if result["supported"]

    )

    return round(

        supported / len(verified),

        4,

    )


# ======================================================
# Unsupported Ratio
# ======================================================

def unsupported_ratio(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Ratio of unsupported claims.
    """

    verified = self.verify_all_claims(

        answer,

        context,

    )

    if not verified:

        return 0.0

    unsupported = sum(

        1

        for result in verified

        if not result["supported"]

    )

    return round(

        unsupported / len(verified),

        4,

    )


# ======================================================
# Claim Coverage
# ======================================================

def claim_coverage(
    self,
    answer: str,
    context: List[str],
) -> Dict[str, int]:
    """
    Supported/unsupported claim counts.
    """

    verified = self.verify_all_claims(

        answer,

        context,

    )

    supported = sum(

        1

        for result in verified

        if result["supported"]

    )

    unsupported = len(verified) - supported

    return {

        "supported": supported,

        "unsupported": unsupported,

        "total": len(verified),

    }


# ======================================================
# Evidence Mapping
# ======================================================

def evidence_mapping(
    self,
    answer: str,
    context: List[str],
) -> Dict[str, str]:
    """
    Map every claim to its best evidence.
    """

    mapping = {}

    for result in self.verify_all_claims(

        answer,

        context,

    ):

        mapping[

            result["claim"]

        ] = result["evidence"]

    return mapping

# ======================================================
# Faithfulness Score
# ======================================================

def faithfulness_score(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Percentage of supported claims.
    """

    verified = self.verify_all_claims(
        answer,
        context,
    )

    if not verified:
        return 0.0

    supported = sum(
        1
        for claim in verified
        if claim["supported"]
    )

    return round(
        supported / len(verified),
        4,
    )


# ======================================================
# Groundedness Score
# ======================================================

def groundedness_score(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Average semantic overlap between
    answer claims and retrieved evidence.
    """

    verified = self.verify_all_claims(
        answer,
        context,
    )

    if not verified:
        return 0.0

    return round(
        sum(
            claim["score"]
            for claim in verified
        ) / len(verified),
        4,
    )


# ======================================================
# Evidence Score
# ======================================================

def evidence_score(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Measure evidence utilization.
    """

    evidence = self.evidence_sentences(
        answer,
        context,
    )

    context_sentences = self.context_sentences(
        context
    )

    if not context_sentences:
        return 0.0

    return round(
        len(set(evidence))
        /
        len(context_sentences),
        4,
    )


# ======================================================
# Completeness Score
# ======================================================

def completeness_score(
    self,
    answer: str,
    context: List[str],
) -> float:
    """
    Approximate completeness using
    lexical coverage.
    """

    answer_words = set(
        self.normalize_context(answer).split()
    )

    context_words = set(
        self.normalize_context(
            " ".join(context)
        ).split()
    )

    if not context_words:
        return 0.0

    return round(
        len(
            answer_words & context_words
        )
        /
        len(context_words),
        4,
    )


# ======================================================
# Overall Evaluation
# ======================================================

def evaluate(
    self,
    answer: str,
    context: List[str],
) -> FaithfulnessResult:
    """
    Complete faithfulness evaluation.
    """

    verified = self.verify_all_claims(
        answer,
        context,
    )

    supported = sum(
        1
        for claim in verified
        if claim["supported"]
    )

    unsupported = (
        len(verified)
        - supported
    )

    faithfulness = self.faithfulness_score(
        answer,
        context,
    )

    groundedness = self.groundedness_score(
        answer,
        context,
    )

    evidence = self.evidence_score(
        answer,
        context,
    )

    result = FaithfulnessResult(

        score=faithfulness,

        groundedness=groundedness,

        evidence_score=evidence,

        supported_claims=supported,

        unsupported_claims=unsupported,

        total_claims=len(verified),

        metadata={

            "claim_details": verified,

            "citation_coverage":
                self.citation_coverage(
                    answer,
                    context,
                ),

            "unsupported_ratio":
                self.unsupported_ratio(
                    answer,
                    context,
                ),

            "completeness":
                self.completeness_score(
                    answer,
                    context,
                ),

        },

    )

    self.history.append(result)

    return result


# ======================================================
# Batch Evaluation
# ======================================================

def batch_evaluate(
    self,
    answers: List[str],
    contexts: List[List[str]],
) -> List[FaithfulnessResult]:
    """
    Evaluate multiple answers.
    """

    if len(answers) != len(contexts):

        raise FaithfulnessError(

            "Answers and contexts must "

            "have the same length."

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
    Average faithfulness score.
    """

    if not self.history:

        return 0.0

    return round(

        sum(
            r.score
            for r in self.history
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
) -> Optional[FaithfulnessResult]:
    """
    Best evaluation.
    """

    if not self.history:

        return None

    return max(

        self.history,

        key=lambda r: r.score,

    )


# ======================================================
# Worst Result
# ======================================================

def worst_result(
    self,
) -> Optional[FaithfulnessResult]:
    """
    Worst evaluation.
    """

    if not self.history:

        return None

    return min(

        self.history,

        key=lambda r: r.score,

    )

# ======================================================
# Statistics
# ======================================================

def statistics(
    self,
) -> Dict[str, Any]:
    """
    Overall evaluation statistics.
    """

    if not self.history:

        return {

            "evaluations": 0,

            "average_score": 0.0,

            "average_groundedness": 0.0,

            "average_evidence_score": 0.0,

        }

    scores = [

        r.score

        for r in self.history

    ]

    groundedness = [

        r.groundedness

        for r in self.history

    ]

    evidence = [

        r.evidence_score

        for r in self.history

    ]

    supported = sum(

        r.supported_claims

        for r in self.history

    )

    unsupported = sum(

        r.unsupported_claims

        for r in self.history

    )

    return {

        "evaluations": len(self.history),

        "average_score": round(

            sum(scores) / len(scores),

            4,

        ),

        "average_groundedness": round(

            sum(groundedness) / len(groundedness),

            4,

        ),

        "average_evidence_score": round(

            sum(evidence) / len(evidence),

            4,

        ),

        "supported_claims": supported,

        "unsupported_claims": unsupported,

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

            "minimum_claim_length":
                self.config.minimum_claim_length,

            "ignore_case":
                self.config.ignore_case,

            "remove_duplicates":
                self.config.remove_duplicates,

        },

        "statistics": self.statistics(),

        "history_size": len(

            self.history

        ),

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
    Benchmark evaluator.
    """

    import time

    start = time.perf_counter()

    results = self.batch_evaluate(

        answers,

        contexts,

    )

    elapsed = (

        time.perf_counter()

        - start

    )

    return {

        "evaluations": len(results),

        "seconds": round(

            elapsed,

            4,

        ),

        "evaluations_per_second": round(

            len(results)

            / max(

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

        "average_groundedness":

            stats["average_groundedness"],

        "average_evidence_score":

            stats["average_evidence_score"],

        "supported_claims":

            stats["supported_claims"],

        "unsupported_claims":

            stats["unsupported_claims"],

    }


# ======================================================
# Latest Result
# ======================================================

def latest_result(
    self,
) -> Optional[FaithfulnessResult]:
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
# Cleanup
# ======================================================

def cleanup(
    self,
) -> None:
    """
    Cleanup evaluator.
    """

    self.clear_history()

    logger.info(

        "FaithfulnessEvaluator cleaned."

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

        "FaithfulnessEvaluator("

        f"evaluations={stats['evaluations']}, "

        f"average_score={stats['average_score']}, "

        f"threshold={self.config.similarity_threshold}"

        ")"

    )
