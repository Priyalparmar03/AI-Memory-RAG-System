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
