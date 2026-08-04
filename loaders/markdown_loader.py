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
