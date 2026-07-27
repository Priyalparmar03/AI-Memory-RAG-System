"""
embeddings/chunker.py
=====================

Production Semantic Chunker

Features
--------
- Fixed-size chunking
- Sentence chunking
- Paragraph chunking
- Sliding window
- Overlap support
- Metadata generation
- Chunk validation
- Statistics
- Logging

Author: Priyal Parmar
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ==========================================================
# Exceptions
# ==========================================================

class ChunkingError(Exception):
    """Raised when chunking fails."""


# ==========================================================
# Chunk Object
# ==========================================================

@dataclass
class Chunk:

    chunk_id: str

    text: str

    index: int

    start_char: int

    end_char: int

    token_count: int

    word_count: int

    metadata: Dict


# ==========================================================
# Chunker
# ==========================================================

class SemanticChunker:
    """
    Production semantic chunker.

    Parameters
    ----------
    chunk_size : int

    overlap : int
    """

    def __init__(

        self,

        chunk_size: int = 500,

        overlap: int = 50,

    ):

        if overlap >= chunk_size:

            raise ChunkingError(

                "overlap must be smaller than chunk_size."

            )

        self.chunk_size = chunk_size

        self.overlap = overlap

        logger.info(

            "SemanticChunker initialized "

            "(chunk_size=%d overlap=%d)",

            chunk_size,

            overlap,

        )

    # ======================================================
    # Validation
    # ======================================================

    @staticmethod
    def validate_text(

        text: str,

    ) -> str:

        if text is None:

            raise ChunkingError(

                "Text cannot be None."

            )

        if not isinstance(

            text,

            str,

        ):

            raise ChunkingError(

                "Input must be string."

            )

        text = text.strip()

        if not text:

            raise ChunkingError(

                "Empty text."

            )

        return text

    # ======================================================
    # Utilities
    # ======================================================

    @staticmethod
    def estimate_tokens(

        text: str,

    ) -> int:
        """
        Approximate token count.
        """

        return max(

            1,

            len(text.split()),

        )

    @staticmethod
    def word_count(

        text: str,

    ) -> int:

        return len(

            text.split()

        )

    @staticmethod
    def sentence_count(

        text: str,

    ) -> int:

        sentences = re.split(

            r"[.!?]+",

            text,

        )

        return len(

            [

                s

                for s in sentences

                if s.strip()

            ]

        )

    # ======================================================
    # Metadata
    # ======================================================

    def create_metadata(

        self,

        text: str,

        index: int,

        start: int,

        end: int,

        extra: Optional[Dict] = None,

    ) -> Dict:

        metadata = {

            "chunk_id": str(uuid.uuid4()),

            "chunk_index": index,

            "characters": len(text),

            "words": self.word_count(text),

            "tokens": self.estimate_tokens(text),

            "start": start,

            "end": end,

        }

        if extra:

            metadata.update(extra)

        return metadata

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(

        self,

        chunks: List[Chunk],

    ) -> Dict:

        if not chunks:

            return {}

        token_counts = [

            c.token_count

            for c in chunks

        ]

        word_counts = [

            c.word_count

            for c in chunks

        ]

        return {

            "chunks": len(chunks),

            "avg_tokens": sum(token_counts)

            / len(token_counts),

            "avg_words": sum(word_counts)

            / len(word_counts),

            "max_tokens": max(token_counts),

            "min_tokens": min(token_counts),

        }

    # ======================================================
    # Fixed Character Chunking
    # ======================================================

    def fixed_chunks(

        self,

        text: str,

    ) -> List[Chunk]:
        """
        Fixed-size character chunking.
        """

        text = self.validate_text(text)

        chunks: List[Chunk] = []

        start = 0

        index = 0

        while start < len(text):

            end = min(

                start + self.chunk_size,

                len(text),

            )

            chunk_text = text[start:end].strip()

            metadata = self.create_metadata(

                chunk_text,

                index,

                start,

                end,

            )

            chunks.append(

                Chunk(

                    chunk_id=metadata["chunk_id"],

                    text=chunk_text,

                    index=index,

                    start_char=start,

                    end_char=end,

                    token_count=metadata["tokens"],

                    word_count=metadata["words"],

                    metadata=metadata,

                )

            )

            index += 1

            start = end - self.overlap

        logger.info(

            "Generated %d fixed chunks.",

            len(chunks),

        )

        return chunks
