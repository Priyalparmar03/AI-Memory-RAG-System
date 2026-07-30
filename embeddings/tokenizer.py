from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class TokenizerError(Exception):
    """Tokenizer exception."""


# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class TokenizerConfig:

    model_name: str = "BAAI/bge-small-en-v1.5"

    backend: str = "huggingface"

    max_tokens: int = 512

    padding: bool = False

    truncation: bool = False

    add_special_tokens: bool = True

    return_attention_mask: bool = True

    return_token_type_ids: bool = False


# ==========================================================
# Statistics
# ==========================================================

@dataclass(slots=True)
class TokenizerStatistics:

    model_name: str

    backend: str

    vocabulary_size: int

    max_length: int


# ==========================================================
# Tokenizer
# ==========================================================

class TextTokenizer:

    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
    ):

        self.config = config or TokenizerConfig()

        self.tokenizer = self._load_tokenizer()

        logger.info(
            "Tokenizer loaded: %s",
            self.config.model_name,
        )

    # ======================================================
    # Initialization
    # ======================================================

    def _load_tokenizer(self):

        backend = self.config.backend.lower()

        if backend == "huggingface":

            return AutoTokenizer.from_pretrained(
                self.config.model_name
            )

        elif backend == "whitespace":

            return None

        raise TokenizerError(
            f"Unsupported backend: {backend}"
        )

    # ======================================================
    # Encode
    # ======================================================

    def encode(
        self,
        text: str,
    ) -> List[int]:

        if self.config.backend == "whitespace":

            return list(range(len(text.split())))

        return self.tokenizer.encode(

            text,

            add_special_tokens=self.config.add_special_tokens,

            truncation=self.config.truncation,

            max_length=self.config.max_tokens,

        )

    # ======================================================
    # Decode
    # ======================================================

    def decode(
        self,
        token_ids: List[int],
    ) -> str:

        if self.config.backend == "whitespace":

            return " ".join(

                map(str, token_ids)

            )

        return self.tokenizer.decode(

            token_ids,

            skip_special_tokens=True,

        )

    # ======================================================
    # Tokenize
    # ======================================================

    def tokenize(
        self,
        text: str,
    ) -> List[str]:

        if self.config.backend == "whitespace":

            return text.split()

        return self.tokenizer.tokenize(text)

    # ======================================================
    # Detokenize
    # ======================================================

    def detokenize(
        self,
        tokens: List[str],
    ) -> str:

        if self.config.backend == "whitespace":

            return " ".join(tokens)

        return self.tokenizer.convert_tokens_to_string(
            tokens
        )

    # ======================================================
    # Count Tokens
    # ======================================================

    def count_tokens(
        self,
        text: str,
    ) -> int:

        return len(

            self.encode(text)

        )


    # ======================================================
    # Vocabulary Size
    # ======================================================

    def vocabulary_size(
        self,
    ) -> int:

        if self.config.backend == "whitespace":

            return 0

        return self.tokenizer.vocab_size

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(
        self,
    ) -> TokenizerStatistics:

        return TokenizerStatistics(

            model_name=self.config.model_name,

            backend=self.config.backend,

            vocabulary_size=self.vocabulary_size(),

            max_length=self.config.max_tokens,

        )

