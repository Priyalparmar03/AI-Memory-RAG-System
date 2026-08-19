from __future__ import annotations
import html
import logging
import re
import unicodedata

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class CleaningError(Exception):
    """
    Text cleaning exception.
    """
    pass


# ==========================================================
# Cleaning Configuration
# ==========================================================

@dataclass(slots=True)
class CleaningConfig:
    """
    Configuration for text cleaning.
    """

    normalize_unicode: bool = True

    normalize_whitespace: bool = True

    remove_extra_spaces: bool = True

    remove_empty_lines: bool = True

    strip_lines: bool = True

    preserve_paragraphs: bool = True

    decode_html_entities: bool = True

    remove_control_characters: bool = True

    collapse_newlines: bool = True

    max_consecutive_newlines: int = 2

    min_text_length: int = 1

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ==========================================================
# Text Cleaner
# ==========================================================

class TextCleaner:
    """
    Production text cleaning engine.
    """

    def __init__(
        self,
        config: Optional[CleaningConfig] = None,
    ) -> None:

        self.config = (
            config
            or CleaningConfig()
        )

        self.rules: List[Dict[str, Any]] = []

        self.statistics: Dict[str, int] = {
            "documents_cleaned": 0,
            "characters_before": 0,
            "characters_after": 0,
            "lines_removed": 0,
            "spaces_removed": 0,
            "control_characters_removed": 0,
        }

    # ======================================================
    # Validate Input
    # ======================================================

    def validate(
        self,
        text: str,
    ) -> None:
        """
        Validate text input.
        """

        if not isinstance(
            text,
            str,
        ):
            raise CleaningError(
                "Text must be a string."
            )

        if (
            text
            and
            len(text)
            < self.config.min_text_length
        ):
            raise CleaningError(
                "Text is shorter than "
                "the configured minimum length."
            )

    # ======================================================
    # Unicode Normalization
    # ======================================================

    def normalize_unicode(
        self,
        text: str,
    ) -> str:
        """
        Normalize Unicode characters.
        """

        if not self.config.normalize_unicode:
            return text

        return unicodedata.normalize(
            "NFKC",
            text,
        )

    # ======================================================
    # Remove Control Characters
    # ======================================================

    def remove_control_characters(
        self,
        text: str,
    ) -> str:
        """
        Remove unwanted control characters
        while preserving newlines and tabs.
        """

        if not self.config.remove_control_characters:
            return text

        result = []

        for character in text:

            if character in (
                "\n",
                "\r",
                "\t",
            ):
                result.append(character)
                continue

            category = (
                unicodedata.category(
                    character
                )
            )

            if category == "Cc":

                self.statistics[
                    "control_characters_removed"
                ] += 1

                continue

            result.append(character)

        return "".join(result)

    # ======================================================
    # Decode HTML Entities
    # ======================================================

    def decode_html_entities(
        self,
        text: str,
    ) -> str:
        """
        Decode HTML entities.
        """

        if not self.config.decode_html_entities:
            return text

        return html.unescape(text)

    # ======================================================
    # Normalize Line Endings
    # ======================================================

    def normalize_line_endings(
        self,
        text: str,
    ) -> str:
        """
        Normalize Windows and old Mac line endings.
        """

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        return text

    # ======================================================
    # Strip Lines
    # ======================================================

    def strip_lines(
        self,
        text: str,
    ) -> str:
        """
        Remove unnecessary whitespace
        around individual lines.
        """

        if not self.config.strip_lines:
            return text

        lines = text.split("\n")

        return "\n".join(
            line.strip()
            for line in lines
        )

    # ======================================================
    # Normalize Whitespace
    # ======================================================

    def normalize_whitespace(
        self,
        text: str,
    ) -> str:
        """
        Normalize spaces and tabs.
        """

        if not self.config.normalize_whitespace:
            return text

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        return text

    # ======================================================
    # Remove Empty Lines
    # ======================================================

    def remove_empty_lines(
        self,
        text: str,
    ) -> str:
        """
        Remove unnecessary empty lines.
        """

        if not self.config.remove_empty_lines:
            return text

        lines = text.split("\n")

        cleaned_lines = []

        removed = 0

        for line in lines:

            if line.strip():

                cleaned_lines.append(
                    line
                )

            else:

                removed += 1

        self.statistics[
            "lines_removed"
        ] += removed

        return "\n".join(
            cleaned_lines
        )

    # ======================================================
    # Collapse Newlines
    # ======================================================

    def collapse_newlines(
        self,
        text: str,
    ) -> str:
        """
        Limit consecutive newlines.
        """

        if not self.config.collapse_newlines:
            return text

        maximum = max(
            1,
            self.config.max_consecutive_newlines,
        )

        pattern = (
            r"\n{"
            f"{maximum + 1},"
            r"}"
        )

        replacement = (
            "\n" * maximum
        )

        return re.sub(
            pattern,
            replacement,
            text,
        )

    # ======================================================
    # Clean Basic Text
    # ======================================================

    def clean_basic(
        self,
        text: str,
    ) -> str:
        """
        Apply the basic cleaning pipeline.
        """

        self.validate(text)

        original_length = len(text)

        text = self.normalize_unicode(
            text
        )

        text = self.decode_html_entities(
            text
        )

        text = self.remove_control_characters(
            text
        )

        text = self.normalize_line_endings(
            text
        )

        text = self.normalize_whitespace(
            text
        )

        text = self.strip_lines(
            text
        )

        text = self.collapse_newlines(
            text
        )

        if self.config.remove_empty_lines:

            text = self.remove_empty_lines(
                text
            )

        if self.config.remove_extra_spaces:

            text = re.sub(
                r" {2,}",
                " ",
                text,
            )

        text = text.strip()

        self.statistics[
            "documents_cleaned"
        ] += 1

        self.statistics[
            "characters_before"
        ] += original_length

        self.statistics[
            "characters_after"
        ] += len(text)

        self.statistics[
            "spaces_removed"
        ] += max(
            0,
            original_length - len(text),
        )

        return text
