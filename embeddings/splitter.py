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

    # ======================================================
    # Sentence Splitter
    # ======================================================

    def split_sentences(
        self,
        text: str,
    ) -> List[str]:
        """
        Split text into sentences.
        """

        self.validate_text(text)

        sentences = re.split(

            r'(?<=[.!?])\s+',

            text.strip(),

        )

        sentences = [

            s.strip()

            for s in sentences

            if s.strip()

        ]

        return sentences

    # ======================================================
    # Paragraph Splitter
    # ======================================================

    def split_paragraphs(
        self,
        text: str,
    ) -> List[str]:
        """
        Split text into paragraphs.
        """

        self.validate_text(text)

        paragraphs = re.split(

            r"\n\s*\n",

            text,

        )

        paragraphs = [

            p.strip()

            for p in paragraphs

            if p.strip()

        ]

        return paragraphs

    # ======================================================
    # Recursive Splitter
    # ======================================================

    def recursive_split(
        self,
        text: str,
        chunk_size: Optional[int] = None,
    ) -> List[str]:
        """
        Hierarchical splitting strategy.

        Paragraph
            ↓
        Sentence
            ↓
        Character
        """

        chunk_size = chunk_size or self.config.chunk_size

        paragraphs = self.split_paragraphs(text)

        chunks = []

        for paragraph in paragraphs:

            if len(paragraph) <= chunk_size:

                chunks.append(paragraph)

                continue

            sentences = self.split_sentences(paragraph)

            current = ""

            for sentence in sentences:

                if len(current) + len(sentence) < chunk_size:

                    current += " " + sentence

                else:

                    if current.strip():

                        chunks.append(current.strip())

                    current = sentence

            if current.strip():

                chunks.append(current.strip())

        return chunks

    # ======================================================
    # Sliding Window
    # ======================================================

    def sliding_window(
        self,
        text: str,
        window_size: int = 1000,
        overlap: int = 100,
    ) -> List[str]:
        """
        Character-based sliding window.
        """

        self.validate_text(text)

        if overlap >= window_size:

            raise SplitterError(

                "Overlap must be smaller than window size."

            )

        windows = []

        step = window_size - overlap

        for i in range(

            0,

            len(text),

            step,

        ):

            part = text[i:i + window_size]

            if part.strip():

                windows.append(part.strip())

        return windows

    # ======================================================
    # Merge Small Parts
    # ======================================================

    def merge_small(
        self,
        chunks: List[str],
        minimum_size: int = 100,
    ) -> List[str]:
        """
        Merge very small chunks together.
        """

        merged = []

        buffer = ""

        for chunk in chunks:

            if len(buffer) + len(chunk) < minimum_size:

                buffer += "\n" + chunk

            else:

                if buffer:

                    merged.append(buffer.strip())

                buffer = chunk

        if buffer:

            merged.append(buffer.strip())

        return merged

    # ======================================================
    # Token Estimation
    # ======================================================

    @staticmethod
    def estimate_tokens(
        text: str,
    ) -> int:
        """
        Rough token estimation.

        ~4 characters per token.
        """

        return max(

            1,

            len(text) // 4,

        )

    # ======================================================
    # Token-aware Split
    # ======================================================

    def token_split(
        self,
        text: str,
        max_tokens: int = 512,
    ) -> List[str]:
        """
        Approximate token-based splitting.
        """

        self.validate_text(text)

        pieces = self.split_sentences(text)

        chunks = []

        current = ""

        current_tokens = 0

        for sentence in pieces:

            sentence_tokens = self.estimate_tokens(

                sentence

            )

            if (

                current_tokens + sentence_tokens

                <= max_tokens

            ):

                current += " " + sentence

                current_tokens += sentence_tokens

            else:

                if current.strip():

                    chunks.append(current.strip())

                current = sentence

                current_tokens = sentence_tokens

        if current.strip():

            chunks.append(current.strip())

        return chunks

    # ======================================================
    # Validate Chunks
    # ======================================================

    @staticmethod
    def validate_chunks(
        chunks: List[str],
    ) -> bool:
        """
        Validate chunk list.
        """

        if not isinstance(chunks, list):

            return False

        for chunk in chunks:

            if not isinstance(chunk, str):

                return False

            if not chunk.strip():

                return False

        return True

    # ======================================================
    # Remove Duplicates
    # ======================================================

    @staticmethod
    def remove_duplicates(
        chunks: List[str],
    ) -> List[str]:
        """
        Remove duplicate chunks while preserving order.
        """

        seen = set()

        unique = []

        for chunk in chunks:

            if chunk not in seen:

                unique.append(chunk)

                seen.add(chunk)

        return unique

    # ======================================================
    # Markdown Splitter
    # ======================================================

    def split_markdown(
        self,
        text: str,
    ) -> List[str]:
        """
        Split Markdown by headings.
        """

        self.validate_text(text)

        sections = re.split(
            r"(?=^#{1,6}\s)",
            text,
            flags=re.MULTILINE,
        )

        return [
            s.strip()
            for s in sections
            if s.strip()
        ]

    # ======================================================
    # HTML Splitter
    # ======================================================

    def split_html(
        self,
        html: str,
    ) -> List[str]:
        """
        Split HTML into logical text blocks.
        """

        self.validate_text(html)

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            blocks = []

            for tag in soup.find_all(
                [
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "p",
                    "li",
                    "pre",
                    "code",
                    "blockquote",
                ]
            ):

                text = tag.get_text(
                    " ",
                    strip=True,
                )

                if text:

                    blocks.append(text)

            return blocks

        except ImportError:

            logger.warning(
                "BeautifulSoup not installed."
            )

            clean = re.sub(
                r"<[^>]+>",
                " ",
                html,
            )

            return self.split_paragraphs(clean)

    # ======================================================
    # JSON Splitter
    # ======================================================

    def split_json(
        self,
        text: str,
    ) -> List[str]:
        """
        Split JSON into top-level objects.
        """

        import json

        self.validate_text(text)

        obj = json.loads(text)

        if isinstance(obj, dict):

            return [

                f"{k}: {v}"

                for k, v in obj.items()

            ]

        if isinstance(obj, list):

            return [

                json.dumps(
                    item,
                    indent=2,
                )

                for item in obj

            ]

        return [str(obj)]

    # ======================================================
    # XML Splitter
    # ======================================================

    def split_xml(
        self,
        text: str,
    ) -> List[str]:
        """
        Split XML into child elements.
        """

        import xml.etree.ElementTree as ET

        self.validate_text(text)

        root = ET.fromstring(text)

        chunks = []

        for child in root:

            chunks.append(

                ET.tostring(

                    child,

                    encoding="unicode",

                )

            )

        return chunks

    # ======================================================
    # CSV Splitter
    # ======================================================

    def split_csv(
        self,
        text: str,
        rows_per_chunk: int = 100,
    ) -> List[str]:
        """
        Split CSV by row groups.
        """

        self.validate_text(text)

        lines = text.splitlines()

        if not lines:

            return []

        header = lines[0]

        rows = lines[1:]

        chunks = []

        for i in range(
            0,
            len(rows),
            rows_per_chunk,
        ):

            part = rows[
                i:i + rows_per_chunk
            ]

            chunks.append(

                "\n".join(
                    [header] + part
                )

            )

        return chunks

    # ======================================================
    # YAML Splitter
    # ======================================================

    def split_yaml(
        self,
        text: str,
    ) -> List[str]:
        """
        Split YAML documents.
        """

        self.validate_text(text)

        docs = re.split(
            r"^---\s*$",
            text,
            flags=re.MULTILINE,
        )

        return [

            d.strip()

            for d in docs

            if d.strip()

        ]

    # ======================================================
    # SQL Splitter
    # ======================================================

    def split_sql(
        self,
        text: str,
    ) -> List[str]:
        """
        Split SQL script into statements.
        """

        self.validate_text(text)

        statements = re.split(
            r";\s*\n",
            text,
        )

        return [

            s.strip()

            for s in statements

            if s.strip()

        ]

    # ======================================================
    # Log Splitter
    # ======================================================

    def split_logs(
        self,
        text: str,
    ) -> List[str]:
        """
        Split log file into entries.
        """

        self.validate_text(text)

        entries = re.split(
            r"\n(?=\d{4}-\d{2}-\d{2})",
            text,
        )

        return [

            e.strip()

            for e in entries

            if e.strip()

        ]

    # ======================================================
    # Python Code Splitter
    # ======================================================

    def split_python(
        self,
        code: str,
    ) -> List[str]:
        """
        Split Python source by class/function definitions.
        """

        self.validate_text(code)

        pattern = (
            r"(?=^class\s)"
            r"|(?=^def\s)"
            r"|(?=^async\s+def\s)"
        )

        blocks = re.split(
            pattern,
            code,
            flags=re.MULTILINE,
        )

        return [

            b.strip()

            for b in blocks

            if b.strip()

        ]

    # ======================================================
    # Generic Code Splitter
    # ======================================================

    def split_code(
        self,
        code: str,
    ) -> List[str]:
        """
        Generic source-code splitter.
        """

        self.validate_text(code)

        return self.split_characters(
            code,
            size=1500,
            overlap=150,
        )
