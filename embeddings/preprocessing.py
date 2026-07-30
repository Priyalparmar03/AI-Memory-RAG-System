from __future__ import annotations

import html
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class PreprocessingError(Exception):
    """Raised when preprocessing fails."""


# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class PreprocessingConfig:

    lowercase: bool = False

    normalize_unicode: bool = True

    remove_html: bool = True

    remove_markdown: bool = True

    normalize_whitespace: bool = True

    remove_empty_lines: bool = True


# ==========================================================
# Text Preprocessor
# ==========================================================

class TextPreprocessor:

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
    ) -> None:

        self.config = config or PreprocessingConfig()

    # ------------------------------------------------------
    # Unicode
    # ------------------------------------------------------

    def normalize_unicode(
        self,
        text: str,
    ) -> str:

        return unicodedata.normalize(
            "NFKC",
            text,
        )

    # ------------------------------------------------------
    # HTML Entity Decode
    # ------------------------------------------------------

    def decode_html_entities(
        self,
        text: str,
    ) -> str:

        return html.unescape(text)

    # ------------------------------------------------------
    # Remove HTML
    # ------------------------------------------------------

    def remove_html(
        self,
        text: str,
    ) -> str:

        return re.sub(
            r"<[^>]+>",
            " ",
            text,
        )

    # ------------------------------------------------------
    # Remove Markdown
    # ------------------------------------------------------

    def remove_markdown(
        self,
        text: str,
    ) -> str:

        patterns = [

            r"```.*?```",          # fenced code

            r"`.*?`",              # inline code

            r"!\[.*?\]\(.*?\)",    # image

            r"\[([^\]]+)\]\(.*?\)",# links

            r"^#{1,6}\s*",         # headings

            r"[*_]{1,3}",          # emphasis

            r"^>\s*",              # quote

        ]

        for pattern in patterns:

            text = re.sub(

                pattern,

                " ",

                text,

                flags=re.MULTILINE | re.DOTALL,

            )

        return text

    # ------------------------------------------------------
    # Whitespace
    # ------------------------------------------------------

    def normalize_whitespace(
        self,
        text: str,
    ) -> str:

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        return re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

    # ------------------------------------------------------
    # Remove Empty Lines
    # ------------------------------------------------------

    def remove_empty_lines(
        self,
        text: str,
    ) -> str:

        return "\n".join(

            line

            for line in text.splitlines()

            if line.strip()

        )

    # ------------------------------------------------------
    # Lowercase
    # ------------------------------------------------------

    def lowercase(
        self,
        text: str,
    ) -> str:

        return text.lower()

    # ------------------------------------------------------
    # Strip
    # ------------------------------------------------------

    def strip(
        self,
        text: str,
    ) -> str:

        return text.strip()

    # ------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------

    def preprocess(
        self,
        text: str,
    ) -> str:

        if not isinstance(text, str):

            raise PreprocessingError(
                "Input must be a string."
            )

        if self.config.normalize_unicode:

            text = self.normalize_unicode(text)

        text = self.decode_html_entities(text)

        if self.config.remove_html:

            text = self.remove_html(text)

        if self.config.remove_markdown:

            text = self.remove_markdown(text)

        if self.config.normalize_whitespace:

            text = self.normalize_whitespace(text)

        if self.config.remove_empty_lines:

            text = self.remove_empty_lines(text)

        if self.config.lowercase:

            text = self.lowercase(text)

        return self.strip(text)
