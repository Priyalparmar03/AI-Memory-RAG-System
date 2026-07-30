from __future__ import annotations
import string
import unicodedata
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import html
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List
from unidecode import unidecode
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
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self._stopwords = {
            lang: set(stopwords.words(lang))
            for lang in stopwords.fileids()
        }
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


    # ------------------------------------------------------
    # URL Normalization
    # ------------------------------------------------------

    def normalize_urls(
        self,
        text: str,
        replacement: str = "<URL>",
    ) -> str:

        pattern = r"https?://\S+|www\.\S+"

        return re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------
    # Email Normalization
    # ------------------------------------------------------

    def normalize_emails(
        self,
        text: str,
        replacement: str = "<EMAIL>",
    ) -> str:

        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

        return re.sub(
            pattern,
            replacement,
            text,
        )

    # ------------------------------------------------------
    # Phone Number Normalization
    # ------------------------------------------------------

    def normalize_phone_numbers(
        self,
        text: str,
        replacement: str = "<PHONE>",
    ) -> str:

        pattern = (
            r"\+?\d{1,3}[-.\s]?"
            r"\(?\d{2,4}\)?[-.\s]?"
            r"\d{3,4}[-.\s]?\d{3,4}"
        )

        return re.sub(
            pattern,
            replacement,
            text,
        )

    # ------------------------------------------------------
    # Number Normalization
    # ------------------------------------------------------

    def normalize_numbers(
        self,
        text: str,
        replacement: str = "<NUMBER>",
    ) -> str:

        return re.sub(
            r"\b\d+(\.\d+)?\b",
            replacement,
            text,
        )

    # ------------------------------------------------------
    # Quote Normalization
    # ------------------------------------------------------

    def normalize_quotes(
        self,
        text: str,
    ) -> str:

        replacements = {

            "\u2018": "'",

            "\u2019": "'",

            "\u201C": '"',

            "\u201D": '"',

        }

        for old, new in replacements.items():

            text = text.replace(old, new)

        return text

    # ------------------------------------------------------
    # Dash Normalization
    # ------------------------------------------------------

    def normalize_dashes(
        self,
        text: str,
    ) -> str:

        replacements = {

            "\u2013": "-",

            "\u2014": "-",

            "\u2212": "-",

        }

        for old, new in replacements.items():

            text = text.replace(old, new)

        return text

    # ------------------------------------------------------
    # Remove Zero Width Characters
    # ------------------------------------------------------

    def remove_zero_width(
        self,
        text: str,
    ) -> str:

        return re.sub(

            r"[\u200B-\u200D\uFEFF]",

            "",

            text,

        )

    # ------------------------------------------------------
    # Remove Extra Punctuation
    # ------------------------------------------------------

    def normalize_punctuation(
        self,
        text: str,
    ) -> str:

        text = re.sub(

            r"[!?]{2,}",

            "!",

            text,

        )

        text = re.sub(

            r"\.{3,}",

            "...",

            text,

        )

        text = re.sub(

            r",{2,}",

            ",",

            text,

        )

        return text

    # ------------------------------------------------------
    # Symbol Normalization
    # ------------------------------------------------------

    def normalize_symbols(
        self,
        text: str,
    ) -> str:

        replacements = {

            "•": "-",

            "●": "-",

            "▪": "-",

            "■": "-",

            "→": "->",

            "⇒": "=>",

        }

        for old, new in replacements.items():

            text = text.replace(old, new)

        return text

    # ------------------------------------------------------
    # Remove Page Numbers
    # ------------------------------------------------------

    def remove_page_numbers(
        self,
        text: str,
    ) -> str:
        """
        Remove standalone page numbers.
        """

        return re.sub(
            r"(?m)^\s*(page\s*)?\d+\s*$",
            "",
            text,
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------
    # Remove Headers & Footers
    # ------------------------------------------------------

    def remove_headers_footers(
        self,
        text: str,
    ) -> str:
        """
        Remove repeated headers and footers.
        """

        lines = text.splitlines()

        if len(lines) < 6:
            return text

        counts = {}

        for line in lines:
            line = line.strip()

            if not line:
                continue

            counts[line] = counts.get(line, 0) + 1

        repeated = {

            line

            for line, count in counts.items()

            if count >= 3

        }

        cleaned = [

            line

            for line in lines

            if line.strip() not in repeated

        ]

        return "\n".join(cleaned)

    # ------------------------------------------------------
    # Fix Hyphenation
    # ------------------------------------------------------

    def fix_hyphenation(
        self,
        text: str,
    ) -> str:
        """
        Join words broken across lines.
        """

        return re.sub(

            r"(\w)-\n(\w)",

            r"\1\2",

            text,

        )

    # ------------------------------------------------------
    # Merge Broken Paragraphs
    # ------------------------------------------------------

    def merge_paragraphs(
        self,
        text: str,
    ) -> str:
        """
        Merge wrapped lines into paragraphs.
        """

        text = re.sub(

            r"(?<!\n)\n(?!\n)",

            " ",

            text,

        )

        return text

    # ------------------------------------------------------
    # Remove Watermarks
    # ------------------------------------------------------

    def remove_watermarks(
        self,
        text: str,
    ) -> str:
        """
        Remove common watermark patterns.
        """

        patterns = [

            r"CONFIDENTIAL",

            r"DRAFT",

            r"COPY",

            r"SAMPLE",

            r"DO NOT DISTRIBUTE",

        ]

        for pattern in patterns:

            text = re.sub(

                pattern,

                "",

                text,

                flags=re.IGNORECASE,

            )

        return text

    # ------------------------------------------------------
    # Remove Table Artifacts
    # ------------------------------------------------------

    def remove_table_artifacts(
        self,
        text: str,
    ) -> str:
        """
        Remove repeated table separators.
        """

        text = re.sub(

            r"[-=]{3,}",

            " ",

            text,

        )

        text = re.sub(

            r"\|{2,}",

            "|",

            text,

        )

        return text

    # ------------------------------------------------------
    # Remove Duplicate Lines
    # ------------------------------------------------------

    def remove_duplicate_lines(
        self,
        text: str,
    ) -> str:
        """
        Remove duplicated OCR lines.
        """

        seen = set()

        output = []

        for line in text.splitlines():

            clean = line.strip()

            if not clean:

                continue

            if clean in seen:

                continue

            seen.add(clean)

            output.append(line)

        return "\n".join(output)

    # ------------------------------------------------------
    # Detect Language
    # ------------------------------------------------------

    def detect_language(
        self,
        text: str,
    ) -> str:

        try:

            return detect(text)

        except LangDetectException:

            return "unknown"

    # ------------------------------------------------------
    # Stopword Removal
    # ------------------------------------------------------

    def remove_stopwords(
        self,
        text: str,
        language: str = "english",
    ) -> str:

        words = text.split()

        stops = self._stopwords.get(
            language,
            set(),
        )

        return " ".join(

            word

            for word in words

            if word.lower() not in stops

        )

    # ------------------------------------------------------
    # Lemmatize
    # ------------------------------------------------------

    def lemmatize(
        self,
        text: str,
    ) -> str:

        return " ".join(

            self.lemmatizer.lemmatize(word)

            for word in text.split()

        )

    # ------------------------------------------------------
    # Stem
    # ------------------------------------------------------

    def stem(
        self,
        text: str,
    ) -> str:

        return " ".join(

            self.stemmer.stem(word)

            for word in text.split()

        )

    # ------------------------------------------------------
    # Transliterate
    # ------------------------------------------------------

    def transliterate(
        self,
        text: str,
    ) -> str:

        return unidecode(text)

    # ------------------------------------------------------
    # Remove Accents
    # ------------------------------------------------------

    def remove_accents(
        self,
        text: str,
    ) -> str:

        text = unicodedata.normalize(
            "NFKD",
            text,
        )

        return "".join(

            c

            for c in text

            if not unicodedata.combining(c)

        )

    # ------------------------------------------------------
    # Remove Emojis
    # ------------------------------------------------------

    def remove_emojis(
        self,
        text: str,
    ) -> str:

        emoji_pattern = re.compile(

            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",

            flags=re.UNICODE,

        )

        return emoji_pattern.sub(
            "",
            text,
        )

    # ------------------------------------------------------
    # Replace Emojis
    # ------------------------------------------------------

    def replace_emojis(
        self,
        text: str,
        replacement: str = "<EMOJI>",
    ) -> str:

        emoji_pattern = re.compile(

            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",

            flags=re.UNICODE,

        )

        return emoji_pattern.sub(
            replacement,
            text,
        )

