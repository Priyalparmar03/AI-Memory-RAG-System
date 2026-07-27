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

    # ======================================================
    # Sentence Chunking
    # ======================================================

    def sentence_chunks(
        self,
        text: str,
    ) -> List[Chunk]:
        """
        Split text into sentence-based chunks while respecting
        the configured chunk size.
        """

        text = self.validate_text(text)

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text,
        )

        chunks = []

        current = ""

        start_char = 0

        index = 0

        cursor = 0

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:

                continue

            if len(current) + len(sentence) + 1 <= self.chunk_size:

                if current:

                    current += " "

                current += sentence

            else:

                end_char = start_char + len(current)

                metadata = self.create_metadata(

                    current,

                    index,

                    start_char,

                    end_char,

                )

                chunks.append(

                    Chunk(

                        chunk_id=metadata["chunk_id"],

                        text=current,

                        index=index,

                        start_char=start_char,

                        end_char=end_char,

                        token_count=metadata["tokens"],

                        word_count=metadata["words"],

                        metadata=metadata,

                    )

                )

                index += 1

                start_char = cursor

                current = sentence

            cursor += len(sentence) + 1

        if current:

            end_char = start_char + len(current)

            metadata = self.create_metadata(

                current,

                index,

                start_char,

                end_char,

            )

            chunks.append(

                Chunk(

                    chunk_id=metadata["chunk_id"],

                    text=current,

                    index=index,

                    start_char=start_char,

                    end_char=end_char,

                    token_count=metadata["tokens"],

                    word_count=metadata["words"],

                    metadata=metadata,

                )

            )

        logger.info(

            "Generated %d sentence chunks.",

            len(chunks),

        )

        return chunks

    # ======================================================
    # Paragraph Chunking
    # ======================================================

    def paragraph_chunks(
        self,
        text: str,
    ) -> List[Chunk]:
        """
        Chunk text paragraph by paragraph.
        """

        text = self.validate_text(text)

        paragraphs = [

            p.strip()

            for p in text.split("\n\n")

            if p.strip()

        ]

        chunks = []

        cursor = 0

        for index, paragraph in enumerate(paragraphs):

            metadata = self.create_metadata(

                paragraph,

                index,

                cursor,

                cursor + len(paragraph),

            )

            chunks.append(

                Chunk(

                    chunk_id=metadata["chunk_id"],

                    text=paragraph,

                    index=index,

                    start_char=cursor,

                    end_char=cursor + len(paragraph),

                    token_count=metadata["tokens"],

                    word_count=metadata["words"],

                    metadata=metadata,

                )

            )

            cursor += len(paragraph) + 2

        logger.info(

            "Generated %d paragraph chunks.",

            len(chunks),

        )

        return chunks

    # ======================================================
    # Sliding Window
    # ======================================================

    def sliding_window(
        self,
        text: str,
    ) -> List[Chunk]:
        """
        Sliding window chunking with overlap.
        """

        return self.fixed_chunks(text)

    # ======================================================
    # Recursive Chunking
    # ======================================================

    def recursive_chunks(
        self,
        text: str,
    ) -> List[Chunk]:
        """
        Recursive strategy.

        Paragraphs
            ↓
        Sentences
            ↓
        Fixed chunks
        """

        text = self.validate_text(text)

        if len(text) <= self.chunk_size:

            return self.fixed_chunks(text)

        paragraphs = self.paragraph_chunks(text)

        results = []

        for paragraph in paragraphs:

            if len(paragraph.text) <= self.chunk_size:

                results.append(paragraph)

            else:

                results.extend(

                    self.sentence_chunks(

                        paragraph.text

                    )

                )

        final_chunks = []

        for chunk in results:

            if len(chunk.text) <= self.chunk_size:

                final_chunks.append(chunk)

            else:

                final_chunks.extend(

                    self.fixed_chunks(

                        chunk.text

                    )

                )

        return final_chunks

    # ======================================================
    # Merge Small Chunks
    # ======================================================

    def merge_small_chunks(
        self,
        chunks: List[Chunk],
        minimum_size: int = 120,
    ) -> List[Chunk]:
        """
        Merge chunks that are smaller than the minimum size.
        """

        if not chunks:

            return []

        merged = []

        buffer = None

        for chunk in chunks:

            if buffer is None:

                buffer = chunk

                continue

            if len(buffer.text) < minimum_size:

                buffer.text += "\n" + chunk.text

                buffer.end_char = chunk.end_char

                buffer.word_count = self.word_count(buffer.text)

                buffer.token_count = self.estimate_tokens(buffer.text)

            else:

                merged.append(buffer)

                buffer = chunk

        if buffer:

            merged.append(buffer)

        logger.info(

            "Merged into %d chunks.",

            len(merged),

        )

        return merged

    # ======================================================
    # Split by Headings
    # ======================================================

    def heading_chunks(
        self,
        text: str,
    ) -> List[Chunk]:
        """
        Split markdown or documentation using headings.
        """

        text = self.validate_text(text)

        sections = re.split(

            r'(?=^#{1,6}\s)',

            text,

            flags=re.MULTILINE,

        )

        sections = [

            s.strip()

            for s in sections

            if s.strip()

        ]

        if not sections:

            return self.fixed_chunks(text)

        chunks = []

        cursor = 0

        for index, section in enumerate(sections):

            metadata = self.create_metadata(

                section,

                index,

                cursor,

                cursor + len(section),

            )

            chunks.append(

                Chunk(

                    chunk_id=metadata["chunk_id"],

                    text=section,

                    index=index,

                    start_char=cursor,

                    end_char=cursor + len(section),

                    token_count=metadata["tokens"],

                    word_count=metadata["words"],

                    metadata=metadata,

                )

            )

            cursor += len(section)

        logger.info(

            "Generated %d heading chunks.",

            len(chunks),

        )

        return chunks
