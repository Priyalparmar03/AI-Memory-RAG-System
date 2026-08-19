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

# ======================================================
# Normalize Paragraphs
# ======================================================

def normalize_paragraphs(
    self,
    text: str,
) -> str:
    """
    Normalize paragraph structure while
    preserving paragraph boundaries.
    """

    paragraphs = re.split(
        r"\n\s*\n+",
        text,
    )

    cleaned = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:

            continue

        paragraph = re.sub(
            r"\s+",
            " ",
            paragraph,
        )

        cleaned.append(
            paragraph
        )

    separator = (
        "\n\n"
        if self.config.preserve_paragraphs
        else "\n"
    )

    return separator.join(
        cleaned
    )


# ======================================================
# Normalize Sentences
# ======================================================

def normalize_sentences(
    self,
    text: str,
) -> str:
    """
    Normalize whitespace around sentences.
    """

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text,
    )

    text = re.sub(
        r"([.!?])([A-Za-z])",
        r"\1 \2",
        text,
    )

    text = re.sub(
        r" {2,}",
        " ",
        text,
    )

    return text.strip()


# ======================================================
# Split Paragraphs
# ======================================================

def split_paragraphs(
    self,
    text: str,
) -> List[str]:
    """
    Split cleaned text into paragraphs.
    """

    return [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n+",
            text,
        )
        if paragraph.strip()
    ]


# ======================================================
# Split Sentences
# ======================================================

def split_sentences(
    self,
    text: str,
) -> List[str]:
    """
    Basic sentence segmentation.
    """

    text = self.normalize_sentences(
        text
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ======================================================
# Remove Blank Characters
# ======================================================

def remove_blank_characters(
    self,
    text: str,
) -> str:
    """
    Remove zero-width and invisible characters.
    """

    characters = (
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
        "\u2060",
    )

    for character in characters:

        text = text.replace(
            character,
            "",
        )

    return text


# ======================================================
# Replace Text
# ======================================================

def replace_text(
    self,
    text: str,
    replacements: Dict[str, str],
) -> str:
    """
    Apply literal text replacements.
    """

    for old, new in replacements.items():

        text = text.replace(
            old,
            new,
        )

    return text


# ======================================================
# Batch Cleaning
# ======================================================

def clean_batch(
    self,
    texts: List[str],
    advanced: bool = True,
    **kwargs,
) -> List[str]:
    """
    Clean multiple text documents.
    """

    cleaned = []

    for text in texts:

        if advanced:

            result = self.clean_advanced(
                text,
                **kwargs,
            )

        else:

            result = self.clean_basic(
                text
            )

        cleaned.append(
            result
        )

    return cleaned


# ======================================================
# Cleaning Report
# ======================================================

def cleaning_report(
    self,
) -> Dict[str, Any]:
    """
    Return cleaning statistics.
    """

    before = self.statistics[
        "characters_before"
    ]

    after = self.statistics[
        "characters_after"
    ]

    removed = max(
        0,
        before - after,
    )

    reduction = (
        (removed / before) * 100
        if before
        else 0.0
    )

    return {
        "documents_cleaned":
            self.statistics[
                "documents_cleaned"
            ],

        "characters_before":
            before,

        "characters_after":
            after,

        "characters_removed":
            removed,

        "reduction_percent":
            round(
                reduction,
                2,
            ),

        "lines_removed":
            self.statistics[
                "lines_removed"
            ],

        "spaces_removed":
            self.statistics[
                "spaces_removed"
            ],

        "control_characters_removed":
            self.statistics[
                "control_characters_removed"
            ],

        "custom_rules":
            len(self.rules),
    }


# ======================================================
# Text Statistics
# ======================================================

def text_statistics(
    self,
    text: str,
) -> Dict[str, Any]:
    """
    Calculate text statistics.
    """

    words = re.findall(
        r"\b\w+\b",
        text,
        flags=re.UNICODE,
    )

    sentences = self.split_sentences(
        text
    )

    paragraphs = self.split_paragraphs(
        text
    )

    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    return {
        "characters":
            len(text),

        "characters_without_spaces":
            len(
                re.sub(
                    r"\s+",
                    "",
                    text,
                )
            ),

        "words":
            len(words),

        "sentences":
            len(sentences),

        "paragraphs":
            len(paragraphs),

        "lines":
            len(lines),

        "average_word_length":
            round(
                sum(
                    len(word)
                    for word in words
                ) / len(words),
                2,
            )
            if words
            else 0.0,

        "average_sentence_length":
            round(
                len(words)
                / len(sentences),
                2,
            )
            if sentences
            else 0.0,
    }


# ======================================================
# Validate Cleaned Text
# ======================================================

def validate_cleaned_text(
    self,
    text: str,
) -> Dict[str, Any]:
    """
    Validate the quality of cleaned text.
    """

    stripped = text.strip()

    return {
        "valid":
            bool(stripped),

        "empty":
            not bool(stripped),

        "character_count":
            len(stripped),

        "word_count":
            len(
                re.findall(
                    r"\b\w+\b",
                    stripped,
                    flags=re.UNICODE,
                )
            ),

        "has_html":
            bool(
                re.search(
                    r"<[^>]+>",
                    stripped,
                )
            ),

        "has_urls":
            bool(
                re.search(
                    r"https?://\S+",
                    stripped,
                )
            ),

        "has_excessive_spaces":
            bool(
                re.search(
                    r"[ \t]{2,}",
                    stripped,
                )
            ),

        "has_excessive_newlines":
            bool(
                re.search(
                    r"\n{3,}",
                    stripped,
                )
            ),
    }


# ======================================================
# Clean Document
# ======================================================

def clean_document(
    self,
    text: str,
    replacements: Optional[
        Dict[str, str]
    ] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Clean a complete document and return
    the text together with diagnostics.
    """

    original = text

    cleaned = self.clean_advanced(
        text,
        **kwargs,
    )

    cleaned = self.remove_blank_characters(
        cleaned
    )

    if replacements:

        cleaned = self.replace_text(
            cleaned,
            replacements,
        )

    cleaned = self.normalize_paragraphs(
        cleaned
    )

    cleaned = self.normalize_sentences(
        cleaned
    )

    return {
        "text":
            cleaned,

        "original_length":
            len(original),

        "cleaned_length":
            len(cleaned),

        "statistics":
            self.text_statistics(
                cleaned
            ),

        "validation":
            self.validate_cleaned_text(
                cleaned
            ),

        "report":
            self.cleaning_report(),
    }


# ======================================================
# Serialization
# ======================================================

def to_dict(
    self,
) -> Dict[str, Any]:
    """
    Serialize cleaner configuration and state.
    """

    return {
        "config": {
            "normalize_unicode":
                self.config.normalize_unicode,

            "normalize_whitespace":
                self.config.normalize_whitespace,

            "remove_extra_spaces":
                self.config.remove_extra_spaces,

            "remove_empty_lines":
                self.config.remove_empty_lines,

            "strip_lines":
                self.config.strip_lines,

            "preserve_paragraphs":
                self.config.preserve_paragraphs,

            "decode_html_entities":
                self.config.decode_html_entities,

            "remove_control_characters":
                self.config.remove_control_characters,

            "collapse_newlines":
                self.config.collapse_newlines,

            "max_consecutive_newlines":
                self.config.max_consecutive_newlines,

            "min_text_length":
                self.config.min_text_length,

            "metadata":
                self.config.metadata,
        },

        "rules":
            self.rules,

        "statistics":
            self.statistics,

        "report":
            self.cleaning_report(),
    }


# ======================================================
# JSON Serialization
# ======================================================

def to_json(
    self,
    indent: int = 4,
) -> str:
    """
    Serialize cleaner state to JSON.
    """

    return json.dumps(
        self.to_dict(),
        indent=indent,
        ensure_ascii=False,
        default=str,
    )
