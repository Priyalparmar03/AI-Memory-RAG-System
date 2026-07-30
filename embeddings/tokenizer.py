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

# ==========================================================
# Truncate
# ==========================================================

def truncate(
    self,
    text: str,
    max_tokens: int | None = None,
) -> str:
    """
    Truncate text to maximum token limit.
    """

    limit = max_tokens or self.config.max_tokens

    if self.config.backend == "whitespace":

        words = text.split()

        return " ".join(words[:limit])

    token_ids = self.encode(text)

    token_ids = token_ids[:limit]

    return self.decode(token_ids)


# ==========================================================
# Batch Truncate
# ==========================================================

def truncate_batch(
    self,
    texts: List[str],
    max_tokens: int | None = None,
) -> List[str]:

    return [

        self.truncate(

            text,

            max_tokens,

        )

        for text in texts

    ]


# ==========================================================
# Sliding Window
# ==========================================================

def sliding_window(
    self,
    text: str,
    window_size: int = 512,
    stride: int = 256,
) -> List[str]:
    """
    Create overlapping windows.
    """

    token_ids = self.encode(text)

    windows = []

    start = 0

    while start < len(token_ids):

        chunk = token_ids[

            start:

            start + window_size

        ]

        if not chunk:

            break

        windows.append(

            self.decode(chunk)

        )

        start += stride

    return windows


# ==========================================================
# Batch Sliding Window
# ==========================================================

def sliding_window_batch(
    self,
    texts: List[str],
    window_size: int = 512,
    stride: int = 256,
) -> List[List[str]]:

    return [

        self.sliding_window(

            text,

            window_size,

            stride,

        )

        for text in texts

    ]


# ==========================================================
# Overlap Chunks
# ==========================================================

def overlap_chunks(
    self,
    chunks: List[str],
    overlap: int = 50,
) -> List[str]:
    """
    Add overlap between chunks.
    """

    if len(chunks) <= 1:

        return chunks

    output = [

        chunks[0]

    ]

    for i in range(

        1,

        len(chunks),

    ):

        previous = self.encode(

            chunks[i - 1]

        )

        current = self.encode(

            chunks[i]

        )

        overlap_tokens = previous[-overlap:]

        merged = overlap_tokens + current

        output.append(

            self.decode(

                merged

            )

        )

    return output


# ==========================================================
# Context Validation
# ==========================================================

def validate_context(
    self,
    text: str,
) -> dict:
    """
    Validate context length.
    """

    used = self.count_tokens(text)

    remaining = max(

        0,

        self.config.max_tokens - used,

    )

    return {

        "valid": used <= self.config.max_tokens,

        "used_tokens": used,

        "remaining_tokens": remaining,

        "max_tokens": self.config.max_tokens,

    }


# ==========================================================
# Check Limits
# ==========================================================

def check_limits(
    self,
    texts: List[str],
) -> List[dict]:

    return [

        self.validate_context(

            text

        )

        for text in texts

    ]


# ==========================================================
# Split By Token Count
# ==========================================================

def split_by_tokens(
    self,
    text: str,
    max_tokens: int,
) -> List[str]:
    """
    Split text into token-sized chunks.
    """

    token_ids = self.encode(text)

    chunks = []

    for i in range(

        0,

        len(token_ids),

        max_tokens,

    ):

        chunk = token_ids[

            i:i + max_tokens

        ]

        chunks.append(

            self.decode(chunk)

        )

    return chunks


# ==========================================================
# Split With Overlap
# ==========================================================

def split_with_overlap(
    self,
    text: str,
    max_tokens: int,
    overlap: int,
) -> List[str]:
    """
    Token-aware splitting with overlap.
    """

    token_ids = self.encode(text)

    chunks = []

    step = max_tokens - overlap

    for i in range(

        0,

        len(token_ids),

        step,

    ):

        chunk = token_ids[

            i:i + max_tokens

        ]

        if chunk:

            chunks.append(

                self.decode(chunk)

            )

    return chunks


# ==========================================================
# Long Document
# ==========================================================

def prepare_document(
    self,
    text: str,
    max_tokens: int | None = None,
    overlap: int = 50,
) -> List[str]:
    """
    Prepare long document for embedding.
    """

    max_tokens = (

        max_tokens

        or

        self.config.max_tokens

    )

    if self.count_tokens(text) <= max_tokens:

        return [text]

    return self.split_with_overlap(

        text,

        max_tokens,

        overlap,

    )
