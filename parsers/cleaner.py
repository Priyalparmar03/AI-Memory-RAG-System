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

# ======================================================
# Remove HTML Tags
# ======================================================

def remove_html_tags(
    self,
    text: str,
) -> str:
    """
    Remove HTML/XML tags from text.
    """

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    return text


# ======================================================
# Remove URLs
# ======================================================

def remove_urls(
    self,
    text: str,
    replacement: str = "",
) -> str:
    """
    Remove URLs from text.
    """

    pattern = (
        r"https?://"
        r"(?:www\.)?"
        r"[^\s<]+"
    )

    return re.sub(
        pattern,
        replacement,
        text,
        flags=re.IGNORECASE,
    )


# ======================================================
# Remove Email Addresses
# ======================================================

def remove_emails(
    self,
    text: str,
    replacement: str = "",
) -> str:
    """
    Remove email addresses.
    """

    pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+"
        r"\.[A-Za-z]{2,}\b"
    )

    return re.sub(
        pattern,
        replacement,
        text,
    )


# ======================================================
# Normalize Quotes
# ======================================================

def normalize_quotes(
    self,
    text: str,
) -> str:
    """
    Normalize common Unicode quotation marks.
    """

    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new,
        )

    return text


# ======================================================
# Normalize Dashes
# ======================================================

def normalize_dashes(
    self,
    text: str,
) -> str:
    """
    Normalize different dash characters.
    """

    replacements = {
        "–": "-",
        "—": "-",
        "―": "-",
        "‐": "-",
        "-": "-",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new,
        )

    return text


# ======================================================
# Remove Repeated Spaces
# ======================================================

def remove_repeated_spaces(
    self,
    text: str,
) -> str:
    """
    Remove repeated spaces.
    """

    return re.sub(
        r"[ ]{2,}",
        " ",
        text,
    )


# ======================================================
# Remove Duplicate Lines
# ======================================================

def remove_duplicate_lines(
    self,
    text: str,
) -> str:
    """
    Remove duplicate lines while
    preserving their original order.
    """

    lines = text.splitlines()

    seen = set()

    result = []

    removed = 0

    for line in lines:

        normalized = line.strip()

        if not normalized:

            result.append(line)

            continue

        key = normalized.casefold()

        if key in seen:

            removed += 1

            continue

        seen.add(key)

        result.append(line)

    self.statistics[
        "lines_removed"
    ] += removed

    return "\n".join(result)


# ======================================================
# Detect Repeated Lines
# ======================================================

def find_repeated_lines(
    self,
    text: str,
    minimum_occurrences: int = 2,
) -> List[str]:
    """
    Find lines appearing multiple times.
    """

    counts: Dict[str, int] = {}

    original_lines: Dict[str, str] = {}

    for line in text.splitlines():

        normalized = line.strip()

        if not normalized:

            continue

        key = normalized.casefold()

        counts[key] = (
            counts.get(key, 0) + 1
        )

        original_lines[key] = normalized

    return [
        original_lines[key]
        for key, count in counts.items()
        if count >= minimum_occurrences
    ]


# ======================================================
# Remove Repeated Headers and Footers
# ======================================================

def remove_repeated_headers_footers(
    self,
    text: str,
    minimum_occurrences: int = 3,
) -> str:
    """
    Remove lines that repeatedly occur
    throughout a document.

    Useful for PDF-extracted page headers
    and footers.
    """

    repeated = set(
        line.casefold()
        for line in self.find_repeated_lines(
            text,
            minimum_occurrences,
        )
    )

    if not repeated:

        return text

    lines = text.splitlines()

    result = []

    removed = 0

    for line in lines:

        if line.strip().casefold() in repeated:

            removed += 1

            continue

        result.append(line)

    self.statistics[
        "lines_removed"
    ] += removed

    return "\n".join(result)


# ======================================================
# OCR Noise Cleanup
# ======================================================

def clean_ocr_noise(
    self,
    text: str,
) -> str:
    """
    Remove common OCR extraction noise.
    """

    replacements = {
        "\u00ad": "",
        "\ufeff": "",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new,
        )

    # Remove spaces accidentally inserted
    # between individual characters.
    text = re.sub(
        r"(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)",
        " ",
        text,
    )

    return text


# ======================================================
# Remove Page Numbers
# ======================================================

def remove_page_numbers(
    self,
    text: str,
) -> str:
    """
    Remove lines containing simple page numbers.
    """

    lines = text.splitlines()

    result = []

    removed = 0

    for line in lines:

        stripped = line.strip()

        if re.fullmatch(
            r"(?:page\s+)?\d+",
            stripped,
            flags=re.IGNORECASE,
        ):

            removed += 1

            continue

        result.append(line)

    self.statistics[
        "lines_removed"
    ] += removed

    return "\n".join(result)


# ======================================================
# Remove Excessive Punctuation
# ======================================================

def normalize_punctuation(
    self,
    text: str,
) -> str:
    """
    Normalize excessive punctuation.
    """

    text = re.sub(
        r"!{2,}",
        "!",
        text,
    )

    text = re.sub(
        r"\?{2,}",
        "?",
        text,
    )

    text = re.sub(
        r"\.{4,}",
        "...",
        text,
    )

    return text


# ======================================================
# Apply Custom Regex Rule
# ======================================================

def add_rule(
    self,
    pattern: str,
    replacement: str = "",
    flags: int = 0,
    name: str = "",
) -> None:
    """
    Add a custom regex cleaning rule.
    """

    try:

        re.compile(
            pattern,
            flags,
        )

    except re.error as exc:

        raise CleaningError(
            f"Invalid regex rule: {exc}"
        ) from exc

    self.rules.append(
        {
            "pattern": pattern,
            "replacement": replacement,
            "flags": flags,
            "name": name or pattern,
        }
    )


# ======================================================
# Apply Custom Rules
# ======================================================

def apply_rules(
    self,
    text: str,
) -> str:
    """
    Apply registered custom cleaning rules.
    """

    for rule in self.rules:

        text = re.sub(
            rule["pattern"],
            rule["replacement"],
            text,
            flags=rule["flags"],
        )

    return text


# ======================================================
# Advanced Cleaning Pipeline
# ======================================================

def clean_advanced(
    self,
    text: str,
    remove_links: bool = False,
    remove_email_addresses: bool = False,
    remove_duplicates: bool = False,
    remove_page_numbers: bool = False,
    remove_headers_footers: bool = False,
    clean_ocr: bool = True,
) -> str:
    """
    Apply advanced document cleaning.
    """

    text = self.clean_basic(
        text
    )

    text = self.remove_html_tags(
        text
    )

    text = self.normalize_quotes(
        text
    )

    text = self.normalize_dashes(
        text
    )

    if remove_links:

        text = self.remove_urls(
            text
        )

    if remove_email_addresses:

        text = self.remove_emails(
            text
        )

    if clean_ocr:

        text = self.clean_ocr_noise(
            text
        )

    text = self.normalize_punctuation(
        text
    )

    if remove_page_numbers:

        text = self.remove_page_numbers(
            text
        )

    if remove_headers_footers:

        text = self.remove_repeated_headers_footers(
            text
        )

    if remove_duplicates:

        text = self.remove_duplicate_lines(
            text
        )

    text = self.apply_rules(
        text
    )

    text = self.normalize_whitespace(
        text
    )

    text = self.collapse_newlines(
        text
    )

    return text.strip()
