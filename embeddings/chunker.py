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

    # ======================================================
    # Token-aware Chunking
    # ======================================================

    def token_chunks(
        self,
        text: str,
        max_tokens: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Split text based on estimated token count.
        """

        text = self.validate_text(text)

        max_tokens = max_tokens or self.chunk_size

        words = text.split()

        chunks = []

        current = []

        index = 0

        cursor = 0

        for word in words:

            current.append(word)

            if len(current) >= max_tokens:

                chunk_text = " ".join(current)

                metadata = self.create_metadata(

                    chunk_text,

                    index,

                    cursor,

                    cursor + len(chunk_text),

                )

                chunks.append(

                    Chunk(

                        chunk_id=metadata["chunk_id"],

                        text=chunk_text,

                        index=index,

                        start_char=cursor,

                        end_char=cursor + len(chunk_text),

                        token_count=metadata["tokens"],

                        word_count=metadata["words"],

                        metadata=metadata,

                    )

                )

                cursor += len(chunk_text) + 1

                current = []

                index += 1

        if current:

            chunk_text = " ".join(current)

            metadata = self.create_metadata(

                chunk_text,

                index,

                cursor,

                cursor + len(chunk_text),

            )

            chunks.append(

                Chunk(

                    chunk_id=metadata["chunk_id"],

                    text=chunk_text,

                    index=index,

                    start_char=cursor,

                    end_char=cursor + len(chunk_text),

                    token_count=metadata["tokens"],

                    word_count=metadata["words"],

                    metadata=metadata,

                )

            )

        logger.info("Generated %d token chunks.", len(chunks))

        return chunks

    # ======================================================
    # Code Chunking
    # ======================================================

    def code_chunks(
        self,
        code: str,
    ) -> List[Chunk]:
        """
        Split source code by functions/classes.
        """

        code = self.validate_text(code)

        pattern = r"(?=^\s*(?:def|class)\s+)"

        blocks = re.split(

            pattern,

            code,

            flags=re.MULTILINE,

        )

        blocks = [

            b.strip()

            for b in blocks

            if b.strip()

        ]

        if not blocks:

            return self.fixed_chunks(code)

        chunks = []

        cursor = 0

        for index, block in enumerate(blocks):

            metadata = self.create_metadata(

                block,

                index,

                cursor,

                cursor + len(block),

            )

            chunks.append(

                Chunk(

                    chunk_id=metadata["chunk_id"],

                    text=block,

                    index=index,

                    start_char=cursor,

                    end_char=cursor + len(block),

                    token_count=metadata["tokens"],

                    word_count=metadata["words"],

                    metadata=metadata,

                )

            )

            cursor += len(block)

        logger.info("Generated %d code chunks.", len(chunks))

        return chunks

    # ======================================================
    # Filter Empty Chunks
    # ======================================================

    def remove_empty(
        self,
        chunks: List[Chunk],
    ) -> List[Chunk]:

        return [

            chunk

            for chunk in chunks

            if chunk.text.strip()

        ]

    # ======================================================
    # Remove Duplicate Chunks
    # ======================================================

    def remove_duplicates(
        self,
        chunks: List[Chunk],
    ) -> List[Chunk]:

        unique = {}

        for chunk in chunks:

            unique.setdefault(

                chunk.text,

                chunk,

            )

        return list(unique.values())

    # ======================================================
    # Sort Chunks
    # ======================================================

    @staticmethod
    def sort_chunks(
        chunks: List[Chunk],
    ) -> List[Chunk]:

        return sorted(

            chunks,

            key=lambda chunk: chunk.index,

        )

    # ======================================================
    # Chunk Length Filter
    # ======================================================

    def filter_by_length(
        self,
        chunks: List[Chunk],
        minimum: int = 30,
    ) -> List[Chunk]:

        return [

            chunk

            for chunk in chunks

            if len(chunk.text) >= minimum

        ]

    # ======================================================
    # Export Metadata
    # ======================================================

    @staticmethod
    def export_metadata(
        chunks: List[Chunk],
    ) -> List[dict]:

        return [

            chunk.metadata

            for chunk in chunks

        ]

    # ======================================================
    # Text Only
    # ======================================================

    @staticmethod
    def texts(
        chunks: List[Chunk],
    ) -> List[str]:

        return [

            chunk.text

            for chunk in chunks

        ]

    # ======================================================
    # Chunk Dictionary
    # ======================================================

    @staticmethod
    def as_dict(
        chunk: Chunk,
    ) -> dict:

        return {

            "id": chunk.chunk_id,

            "text": chunk.text,

            "index": chunk.index,

            "start": chunk.start_char,

            "end": chunk.end_char,

            "tokens": chunk.token_count,

            "words": chunk.word_count,

            "metadata": chunk.metadata,

        }

    # ======================================================
    # All Chunks as Dictionaries
    # ======================================================

    def dictionaries(
        self,
        chunks: List[Chunk],
    ) -> List[dict]:

        return [

            self.as_dict(chunk)

            for chunk in chunks

        ]

    # ======================================================
    # Chunk Validation
    # ======================================================

    def validate_chunks(
        self,
        chunks: List[Chunk],
    ) -> bool:
        """
        Validate generated chunks.
        """

        if not chunks:
            return False

        for chunk in chunks:

            if not chunk.text.strip():
                return False

            if chunk.start_char > chunk.end_char:
                return False

            if chunk.word_count <= 0:
                return False

            if chunk.token_count <= 0:
                return False

        return True

    # ======================================================
    # Chunk File
    # ======================================================

    def chunk_file(
        self,
        path: str,
        strategy: str = "recursive",
        encoding: str = "utf-8",
    ) -> List[Chunk]:
        """
        Read a text file and chunk it.
        """

        with open(
            path,
            "r",
            encoding=encoding,
        ) as f:

            text = f.read()

        return self.chunk_text(
            text=text,
            strategy=strategy,
        )

    # ======================================================
    # Generic Chunk API
    # ======================================================

    def chunk_text(
        self,
        text: str,
        strategy: str = "recursive",
    ) -> List[Chunk]:
        """
        Generic chunking interface.
        """

        strategy = strategy.lower()

        strategies = {

            "fixed": self.fixed_chunks,

            "sentence": self.sentence_chunks,

            "paragraph": self.paragraph_chunks,

            "recursive": self.recursive_chunks,

            "token": self.token_chunks,

            "heading": self.heading_chunks,

            "code": self.code_chunks,

        }

        if strategy not in strategies:

            raise ChunkingError(

                f"Unknown strategy '{strategy}'."

            )

        return strategies[strategy](text)

    # ======================================================
    # Batch Chunking
    # ======================================================

    def batch_chunk(
        self,
        texts: List[str],
        strategy: str = "recursive",
    ) -> List[List[Chunk]]:
        """
        Chunk multiple documents.
        """

        results = []

        for text in texts:

            results.append(

                self.chunk_text(

                    text,

                    strategy,

                )

            )

        return results

    # ======================================================
    # Health
    # ======================================================

    def health(self) -> dict:
        """
        Health information.
        """

        return {

            "status": "healthy",

            "chunk_size": self.chunk_size,

            "overlap": self.overlap,

        }

    # ======================================================
    # Diagnostics
    # ======================================================

    def diagnostics(self) -> dict:
        """
        Diagnostics information.
        """

        return {

            "health": self.health(),

            "configuration": {

                "chunk_size": self.chunk_size,

                "overlap": self.overlap,

            },

        }

    # ======================================================
    # Benchmark
    # ======================================================

    def benchmark(
        self,
        text: str,
        strategy: str = "recursive",
    ) -> dict:
        """
        Measure chunking performance.
        """

        import time

        start = time.perf_counter()

        chunks = self.chunk_text(

            text=text,

            strategy=strategy,

        )

        elapsed = time.perf_counter() - start

        return {

            "strategy": strategy,

            "chunks": len(chunks),

            "seconds": round(

                elapsed,

                5,

            ),

            "characters": len(text),

            "characters_per_second": round(

                len(text) / max(elapsed, 1e-9),

                2,

            ),

        }

    # ======================================================
    # Configuration
    # ======================================================

    def configure(
        self,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> None:
        """
        Update chunker configuration.
        """

        if chunk_size is not None:

            self.chunk_size = chunk_size

        if overlap is not None:

            if overlap >= self.chunk_size:

                raise ChunkingError(

                    "Overlap must be smaller than chunk size."

                )

            self.overlap = overlap

        logger.info(

            "Chunker reconfigured "

            "(chunk_size=%d overlap=%d)",

            self.chunk_size,

            self.overlap,

        )

    # ======================================================
    # Reset
    # ======================================================

    def reset(self) -> None:
        """
        Reset to default configuration.
        """

        self.chunk_size = 500

        self.overlap = 50

        logger.info("Chunker reset to defaults.")

    # ======================================================
    # Close
    # ======================================================

    def close(self) -> None:
        """
        Cleanup resources.
        """

        logger.info("Chunker closed.")

    # ======================================================
    # Context Manager
    # ======================================================

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()

    # ======================================================
    # String Representation
    # ======================================================

    def __repr__(self) -> str:

        return (

            "SemanticChunker("

            f"chunk_size={self.chunk_size}, "

            f"overlap={self.overlap}"

            ")"

        )
