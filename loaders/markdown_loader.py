from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import markdown
from bs4 import BeautifulSoup

from .base_loader import BaseLoader, LoaderError

logger = logging.getLogger(__name__)


# ==========================================================
# Markdown Loader
# ==========================================================

class MarkdownLoader(BaseLoader):
    """
    Production Markdown Loader.

    Supports:

    - .md
    - .markdown

    Converts Markdown to plain text while
    preserving metadata.
    """

    SUPPORTED_EXTENSIONS = [

        ".md",

        ".markdown",

    ]

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ):

        super().__init__(file_path)

        self.validate_extension(

            self.SUPPORTED_EXTENSIONS

        )

        self.encoding = encoding

        logger.info(

            f"MarkdownLoader initialized: "

            f"{self.file_name}"

        )

    # ======================================================
    # Read Markdown File
    # ======================================================

    def read_markdown(
        self,
    ) -> str:
        """
        Read markdown document.
        """

        try:

            with open(

                self.file_path,

                "r",

                encoding=self.encoding,

            ) as file:

                return file.read()

        except UnicodeDecodeError:

            raise LoaderError(

                "Unsupported encoding."

            )

        except Exception as error:

            raise LoaderError(

                str(error)

            )

    # ======================================================
    # Markdown → Plain Text
    # ======================================================

    def extract_text(
        self,
    ) -> str:
        """
        Convert markdown to plain text.
        """

        markdown_text = self.read_markdown()

        html = markdown.markdown(

            markdown_text,

            extensions=[

                "tables",

                "fenced_code",

            ],

        )

        soup = BeautifulSoup(

            html,

            "html.parser",

        )

        return soup.get_text(

            separator="\n"

        )

    # ======================================================
    # Markdown HTML
    # ======================================================

    def html(
        self,
    ) -> str:
        """
        Convert markdown to HTML.
        """

        return markdown.markdown(

            self.read_markdown(),

            extensions=[

                "tables",

                "fenced_code",

            ],

        )

    # ======================================================
    # Metadata
    # ======================================================

    def markdown_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Basic markdown metadata.
        """

        text = self.read_markdown()

        return {

            "characters":

                len(text),

            "words":

                len(

                    text.split()

                ),

            "lines":

                len(

                    text.splitlines()

                ),

            "extension":

                self.extension,

            "file_size":

                self.file_size,

        }

    # ======================================================
    # Load
    # ======================================================

    def load(
        self,
    ) -> Dict[str, Any]:
        """
        Load markdown document.
        """

        text = self.extract_text()

        metadata = self.markdown_metadata()

        return self.build_result(

            text=text,

            pages=[],

            tables=[],

            images=[],

            extra_metadata=metadata,

        )

    # ======================================================
    # Preview
    # ======================================================

    def preview(
        self,
        characters: int = 1000,
    ) -> str:
        """
        Preview markdown.
        """

        return self.extract_text()[:characters]

    # ======================================================
    # Empty Check
    # ======================================================

    def is_empty(
        self,
    ) -> bool:
        """
        Check whether markdown is empty.
        """

        return (

            len(

                self.read_markdown().strip()

            )

            == 0

        )

    # ======================================================
    # String Representation
    # ======================================================

    def __repr__(
        self,
    ):

        return (

            "MarkdownLoader("

            f"file='{self.file_name}'"

            ")"

        )

# ======================================================
# Extract Headings
# ======================================================

def extract_headings(
    self,
) -> List[Dict[str, Any]]:
    """
    Extract markdown headings.
    """

    text = self.read_markdown()

    headings = []

    pattern = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    for match in pattern.finditer(text):

        level = len(match.group(1))

        title = match.group(2).strip()

        headings.append(

            {

                "level": level,

                "title": title,

            }

        )

    return headings


# ======================================================
# Extract Code Blocks
# ======================================================

def extract_code_blocks(
    self,
) -> List[Dict[str, Any]]:
    """
    Extract fenced code blocks.
    """

    text = self.read_markdown()

    pattern = re.compile(

        r"```(\w+)?\n(.*?)```",

        re.DOTALL,

    )

    code_blocks = []

    for index, match in enumerate(

        pattern.finditer(text),

        start=1,

    ):

        language = match.group(1) or "text"

        code = match.group(2).rstrip()

        code_blocks.append(

            {

                "id": index,

                "language": language,

                "code": code,

            }

        )

    return code_blocks


# ======================================================
# Extract Lists
# ======================================================

def extract_lists(
    self,
) -> List[str]:
    """
    Extract markdown list items.
    """

    text = self.read_markdown()

    pattern = re.compile(

        r"^\s*[-*+]\s+(.*)$",

        re.MULTILINE,

    )

    return [

        item.strip()

        for item in pattern.findall(text)

    ]


# ======================================================
# Extract Tables
# ======================================================

def extract_tables(
    self,
) -> List[str]:
    """
    Extract markdown tables.
    """

    text = self.read_markdown()

    lines = text.splitlines()

    tables = []

    current_table = []

    for line in lines:

        if "|" in line:

            current_table.append(line)

        else:

            if current_table:

                tables.append(

                    "\n".join(

                        current_table

                    )

                )

                current_table = []

    if current_table:

        tables.append(

            "\n".join(

                current_table

            )

        )

    return tables


# ======================================================
# Extract Links
# ======================================================

def extract_links(
    self,
) -> List[Dict[str, str]]:
    """
    Extract markdown hyperlinks.
    """

    text = self.read_markdown()

    pattern = re.compile(

        r"\[(.*?)\]\((.*?)\)"

    )

    links = []

    for title, url in pattern.findall(text):

        links.append(

            {

                "text": title,

                "url": url,

            }

        )

    return links


# ======================================================
# Extract Images
# ======================================================

def extract_images(
    self,
) -> List[Dict[str, str]]:
    """
    Extract markdown images.
    """

    text = self.read_markdown()

    pattern = re.compile(

        r"!\[(.*?)\]\((.*?)\)"

    )

    images = []

    for alt, path in pattern.findall(text):

        images.append(

            {

                "alt": alt,

                "path": path,

            }

        )

    return images


# ======================================================
# Heading Count
# ======================================================

def heading_count(
    self,
) -> int:
    """
    Number of headings.
    """

    return len(

        self.extract_headings()

    )


# ======================================================
# Code Block Count
# ======================================================

def code_block_count(
    self,
) -> int:
    """
    Number of code blocks.
    """

    return len(

        self.extract_code_blocks()

    )


# ======================================================
# Link Count
# ======================================================

def link_count(
    self,
) -> int:
    """
    Number of hyperlinks.
    """

    return len(

        self.extract_links()

    )


# ======================================================
# Image Count
# ======================================================

def image_count(
    self,
) -> int:
    """
    Number of images.
    """

    return len(

        self.extract_images()

    )


# ======================================================
# Table Count
# ======================================================

def table_count(
    self,
) -> int:
    """
    Number of markdown tables.
    """

    return len(

        self.extract_tables()

    )
