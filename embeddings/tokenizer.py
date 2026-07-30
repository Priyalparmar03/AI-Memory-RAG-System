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
