from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class SplitterError(Exception):
    """Raised when text splitting fails."""


# ==========================================================
# Configuration
# ==========================================================

@dataclass
class SplitConfig:
    """
    Configuration for the TextSplitter.
    """

    chunk_size: int = 1000
    overlap: int = 100
    separator: str = "\n\n"
    keep_separator: bool = False
    strip_whitespace: bool = True
    remove_empty: bool = True


# ==========================================================
# Text Splitter
# ==========================================================

class TextSplitter:
    """
    Production-grade text splitter.
    """

    def __init__(
        self,
        config: Optional[SplitConfig] = None,
    ):

        self.config = config or SplitConfig()

        logger.info(
            "TextSplitter initialized "
            "(chunk_size=%d overlap=%d)",
            self.config.chunk_size,
            self.config.overlap,
        )

    # ======================================================
    # Validation
    # ======================================================

    @staticmethod
    def validate_text(text: str) -> None:

        if not isinstance(text, str):

            raise SplitterError(
                "Input must be a string."
            )

        if not text.strip():

            raise SplitterError(
                "Input text is empty."
            )

    # ======================================================
    # Statistics
    # ======================================================

    @staticmethod
    def statistics(text: str) -> Dict:

        return {

            "characters": len(text),

            "words": len(text.split()),

            "lines": len(text.splitlines()),

            "paragraphs": len(
                re.split(r"\n\s*\n", text)
            ),

        }

    # ======================================================
    # Character Splitter
    # ======================================================

    def split_characters(
        self,
        text: str,
        size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Split text by characters.
        """

        self.validate_text(text)

        size = size or self.config.chunk_size
        overlap = overlap if overlap is not None else self.config.overlap

        if overlap >= size:

            raise SplitterError(
                "Overlap must be smaller than chunk size."
            )

        chunks = []

        step = size - overlap

        for start in range(0, len(text), step):

            chunk = text[start:start + size]

            if self.config.strip_whitespace:

                chunk = chunk.strip()

            if self.config.remove_empty and not chunk:

                continue

            chunks.append(chunk)

        return chunks

    # ======================================================
    # Word Splitter
    # ======================================================

    def split_words(
        self,
        text: str,
        words_per_chunk: int = 200,
        overlap_words: int = 20,
    ) -> List[str]:
        """
        Split text into word groups.
        """

        self.validate_text(text)

        words = text.split()

        chunks = []

        step = words_per_chunk - overlap_words

        if step <= 0:

            raise SplitterError(
                "Invalid overlap."
            )

        for i in range(0, len(words), step):

            part = words[i:i + words_per_chunk]

            if not part:

                continue

            chunks.append(" ".join(part))

        return chunks

    # ======================================================
    # Custom Separator Split
    # ======================================================

    def split_by_separator(
        self,
        text: str,
        separator: Optional[str] = None,
    ) -> List[str]:
        """
        Split using a custom separator.
        """

        self.validate_text(text)

        separator = separator or self.config.separator

        if self.config.keep_separator:

            parts = re.split(
                f"({re.escape(separator)})",
                text,
            )

            merged = []

            for i in range(0, len(parts), 2):

                piece = parts[i]

                if i + 1 < len(parts):

                    piece += parts[i + 1]

                if piece.strip():

                    merged.append(piece)

            return merged

        parts = text.split(separator)

        if self.config.strip_whitespace:

            parts = [

                p.strip()

                for p in parts

            ]

        if self.config.remove_empty:

            parts = [

                p

                for p in parts

                if p

            ]

        return parts

    # ======================================================
    # Metadata
    # ======================================================

    def metadata(
        self,
        chunks: List[str],
    ) -> List[Dict]:

        metadata = []

        for idx, chunk in enumerate(chunks):

            metadata.append(

                {

                    "index": idx,

                    "characters": len(chunk),

                    "words": len(chunk.split()),

                }

            )

        return metadata
