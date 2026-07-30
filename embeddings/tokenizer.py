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

# ==========================================================
# Batch Encode
# ==========================================================

def batch_encode(
    self,
    texts: List[str],
) -> List[List[int]]:
    """
    Encode multiple texts.
    """

    if not texts:
        return []

    if self.config.backend == "whitespace":

        return [

            list(range(len(text.split())))

            for text in texts

        ]

    encoded = self.tokenizer(

        texts,

        add_special_tokens=self.config.add_special_tokens,

        padding=self.config.padding,

        truncation=self.config.truncation,

        max_length=self.config.max_tokens,

        return_attention_mask=self.config.return_attention_mask,

        return_token_type_ids=self.config.return_token_type_ids,

    )

    return encoded["input_ids"]


# ==========================================================
# Batch Decode
# ==========================================================

def batch_decode(
    self,
    batch_ids: List[List[int]],
) -> List[str]:
    """
    Decode multiple token sequences.
    """

    if not batch_ids:
        return []

    if self.config.backend == "whitespace":

        return [

            " ".join(map(str, ids))

            for ids in batch_ids

        ]

    return self.tokenizer.batch_decode(

        batch_ids,

        skip_special_tokens=True,

    )


# ==========================================================
# Batch Tokenize
# ==========================================================

def tokenize_batch(
    self,
    texts: List[str],
) -> List[List[str]]:
    """
    Tokenize multiple texts.
    """

    return [

        self.tokenize(text)

        for text in texts

    ]


# ==========================================================
# Batch Token Count
# ==========================================================

def count_batch(
    self,
    texts: List[str],
) -> List[int]:
    """
    Count tokens for each text.
    """

    return [

        self.count_tokens(text)

        for text in texts

    ]


# ==========================================================
# Estimate Tokens
# ==========================================================

def estimate_tokens(
    self,
    text: str,
) -> int:
    """
    Fast token estimation without tokenizer.
    Useful for quick chunk planning.
    """

    words = len(text.split())

    return max(

        1,

        int(words * 1.3),

    )


# ==========================================================
# Estimate Batch Tokens
# ==========================================================

def estimate_batch_tokens(
    self,
    texts: List[str],
) -> List[int]:

    return [

        self.estimate_tokens(text)

        for text in texts

    ]


# ==========================================================
# Remaining Context
# ==========================================================

def remaining_tokens(
    self,
    text: str,
) -> int:
    """
    Remaining available context window.
    """

    used = self.count_tokens(text)

    return max(

        0,

        self.config.max_tokens - used,

    )


# ==========================================================
# Fits Context
# ==========================================================

def fits_context(
    self,
    text: str,
) -> bool:
    """
    Check whether text fits into model context.
    """

    return (

        self.count_tokens(text)

        <=

        self.config.max_tokens

    )


# ==========================================================
# Maximum Context
# ==========================================================

def max_context(
    self,
) -> int:
    """
    Maximum context length.
    """

    return self.config.max_tokens


# ==========================================================
# Special Tokens
# ==========================================================

def special_tokens(
    self,
) -> dict:
    """
    Return tokenizer special tokens.
    """

    if self.config.backend == "whitespace":

        return {}

    return {

        "bos": self.tokenizer.bos_token,

        "eos": self.tokenizer.eos_token,

        "cls": self.tokenizer.cls_token,

        "sep": self.tokenizer.sep_token,

        "pad": self.tokenizer.pad_token,

        "mask": self.tokenizer.mask_token,

        "unk": self.tokenizer.unk_token,

    }


# ==========================================================
# Vocabulary
# ==========================================================

def vocabulary(
    self,
) -> dict:
    """
    Return tokenizer vocabulary.
    """

    if self.config.backend == "whitespace":

        return {}

    return self.tokenizer.get_vocab()


# ==========================================================
# Vocabulary Lookup
# ==========================================================

def token_to_id(
    self,
    token: str,
) -> int | None:

    if self.config.backend == "whitespace":

        return None

    return self.tokenizer.convert_tokens_to_ids(
        token
    )


def id_to_token(
    self,
    token_id: int,
) -> str:

    if self.config.backend == "whitespace":

        return str(token_id)

    return self.tokenizer.convert_ids_to_tokens(
        token_id
    )


# ==========================================================
# Batch Statistics
# ==========================================================

def batch_statistics(
    self,
    texts: List[str],
) -> dict:
    """
    Statistics for a batch of texts.
    """

    counts = self.count_batch(texts)

    if not counts:

        return {

            "documents": 0,

            "total_tokens": 0,

            "average_tokens": 0,

            "min_tokens": 0,

            "max_tokens": 0,

        }

    return {

        "documents": len(texts),

        "total_tokens": sum(counts),

        "average_tokens": round(

            sum(counts) / len(counts),

            2,

        ),

        "min_tokens": min(counts),

        "max_tokens": max(counts),

    }
