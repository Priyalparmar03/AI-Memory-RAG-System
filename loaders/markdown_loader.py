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

# ======================================================
# Extract Frontmatter
# ======================================================

def extract_frontmatter(
    self,
) -> Dict[str, Any]:
    """
    Extract YAML frontmatter.
    """

    try:

        import yaml

    except ImportError:

        raise LoaderError(

            "Install PyYAML."

        )

    text = self.read_markdown()

    pattern = re.compile(

        r"^---\n(.*?)\n---",

        re.DOTALL,

    )

    match = pattern.search(text)

    if not match:

        return {}

    try:

        return yaml.safe_load(

            match.group(1)

        ) or {}

    except Exception:

        return {}


# ======================================================
# Extract Sections
# ======================================================

def extract_sections(
    self,
) -> List[Dict[str, Any]]:
    """
    Split markdown into sections
    based on headings.
    """

    markdown_text = self.read_markdown()

    pattern = re.compile(

        r"^(#{1,6})\s+(.*)$",

        re.MULTILINE,

    )

    matches = list(

        pattern.finditer(

            markdown_text

        )

    )

    sections = []

    if not matches:

        return [

            {

                "heading": None,

                "level": 0,

                "content": markdown_text,

            }

        ]

    for index, match in enumerate(

        matches

    ):

        start = match.end()

        end = (

            matches[index + 1].start()

            if index + 1 < len(matches)

            else len(markdown_text)

        )

        sections.append(

            {

                "heading":

                    match.group(2).strip(),

                "level":

                    len(

                        match.group(1)

                    ),

                "content":

                    markdown_text[

                        start:end

                    ].strip(),

            }

        )

    return sections


# ======================================================
# Search
# ======================================================

def search(
    self,
    keyword: str,
) -> List[Dict[str, Any]]:
    """
    Search markdown.
    """

    keyword = keyword.lower()

    sections = self.extract_sections()

    results = []

    for section in sections:

        content = section[

            "content"

        ]

        if keyword in content.lower():

            results.append(

                {

                    "heading":

                        section[

                            "heading"

                        ],

                    "level":

                        section[

                            "level"

                        ],

                    "preview":

                        content[:300],

                }

            )

    return results


# ======================================================
# Statistics
# ======================================================

def statistics(
    self,
) -> Dict[str, Any]:
    """
    Markdown statistics.
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

        "headings":

            self.heading_count(),

        "sections":

            len(

                self.extract_sections()

            ),

        "tables":

            self.table_count(),

        "links":

            self.link_count(),

        "images":

            self.image_count(),

        "code_blocks":

            self.code_block_count(),

        "lists":

            len(

                self.extract_lists()

            ),

    }


# ======================================================
# Section Titles
# ======================================================

def section_titles(
    self,
) -> List[str]:
    """
    Return all section titles.
    """

    return [

        section["heading"]

        for section

        in self.extract_sections()

        if section["heading"]

    ]


# ======================================================
# Preview Section
# ======================================================

def preview_section(
    self,
    heading: str,
    characters: int = 500,
) -> str:
    """
    Preview one section.
    """

    for section in self.extract_sections():

        if (

            section["heading"]

            and

            section["heading"].lower()

            ==

            heading.lower()

        ):

            return section[

                "content"

            ][:characters]

    return ""


# ======================================================
# Document Summary
# ======================================================

def summary(
    self,
) -> Dict[str, Any]:
    """
    Markdown summary.
    """

    stats = self.statistics()

    return {

        "file":

            self.file_name,

        "sections":

            stats["sections"],

        "headings":

            stats["headings"],

        "tables":

            stats["tables"],

        "images":

            stats["images"],

        "links":

            stats["links"],

        "code_blocks":

            stats["code_blocks"],

        "lists":

            stats["lists"],

        "frontmatter":

            bool(

                self.extract_frontmatter()

            ),

    }


# ======================================================
# Export Structure
# ======================================================

def export_structure(
    self,
) -> Dict[str, Any]:
    """
    Export complete markdown structure.
    """

    return {

        "metadata":

            self.markdown_metadata(),

        "frontmatter":

            self.extract_frontmatter(),

        "headings":

            self.extract_headings(),

        "sections":

            self.extract_sections(),

        "tables":

            self.extract_tables(),

        "links":

            self.extract_links(),

        "images":

            self.extract_images(),

        "code_blocks":

            self.extract_code_blocks(),

    }
