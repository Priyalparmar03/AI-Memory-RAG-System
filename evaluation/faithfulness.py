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
