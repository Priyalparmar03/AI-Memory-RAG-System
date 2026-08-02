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
