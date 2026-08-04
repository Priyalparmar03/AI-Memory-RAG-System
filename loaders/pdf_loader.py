from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from .base_loader import BaseLoader, LoaderError

logger = logging.getLogger(__name__)


# ==========================================================
# PDF Loader
# ==========================================================

class PDFLoader(BaseLoader):
    """
    Production PDF Loader.

    Features
    --------
    - Text extraction
    - Page extraction
    - Metadata
    - Document statistics
    - OCR Ready
    - Table Extraction Ready
    - Image Extraction Ready
    """

    SUPPORTED_EXTENSIONS = [
        ".pdf",
    ]

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(
        self,
        file_path: str,
    ):

        super().__init__(file_path)

        self.validate_extension(
            self.SUPPORTED_EXTENSIONS
        )

        self.document: Optional[
            fitz.Document
        ] = None

        logger.info(

            f"PDFLoader initialized for "

            f"{self.file_name}"

        )

    # ======================================================
    # Open PDF
    # ======================================================

    def open_pdf(
        self,
    ) -> fitz.Document:
        """
        Open PDF document.
        """

        if self.document is None:

            try:

                self.document = fitz.open(

                    self.file_path

                )

            except Exception as error:

                raise LoaderError(

                    f"Unable to open PDF: "

                    f"{error}"

                )

        return self.document

    # ======================================================
    # Close PDF
    # ======================================================

    def close_pdf(
        self,
    ) -> None:
        """
        Close opened PDF.
        """

        if self.document:

            self.document.close()

            self.document = None

    # ======================================================
    # Page Count
    # ======================================================

    @property
    def page_count(
        self,
    ) -> int:
        """
        Number of pages.
        """

        document = self.open_pdf()

        return len(document)

    # ======================================================
    # Document Metadata
    # ======================================================

    def document_info(
        self,
    ) -> Dict[str, Any]:
        """
        Extract PDF metadata.
        """

        document = self.open_pdf()

        info = document.metadata

        return {

            "title":

                info.get("title"),

            "author":

                info.get("author"),

            "subject":

                info.get("subject"),

            "keywords":

                info.get("keywords"),

            "creator":

                info.get("creator"),

            "producer":

                info.get("producer"),

            "creation_date":

                info.get("creationDate"),

            "modification_date":

                info.get("modDate"),

            "page_count":

                len(document),

            "encrypted":

                document.is_encrypted,

        }

    # ======================================================
    # Basic Text Extraction
    # ======================================================

    def extract_text(
        self,
    ) -> str:
        """
        Extract complete document text.
        """

        document = self.open_pdf()

        text = []

        for page in document:

            text.append(

                page.get_text()

            )

        return "\n".join(text)

    # ======================================================
    # Load
    # ======================================================

    def load(
        self,
    ) -> Dict[str, Any]:
        """
        Load complete PDF.
        """

        text = self.extract_text()

        metadata = self.document_info()

        return self.build_result(

            text=text,

            pages=[],

            tables=[],

            images=[],

            extra_metadata=metadata,

        )

    # ======================================================
    # File Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Basic PDF statistics.
        """

        text = self.extract_text()

        return {

            "pages":

                self.page_count,

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

        }

    # ======================================================
    # Preview
    # ======================================================

    def preview(
        self,
        characters: int = 1000,
    ) -> str:
        """
        Preview PDF text.
        """

        return self.extract_text()[:characters]

    # ======================================================
    # Empty Check
    # ======================================================

    def is_empty(
        self,
    ) -> bool:
        """
        Check if PDF contains text.
        """

        return (

            len(

                self.extract_text().strip()

            )

            == 0

        )

    # ======================================================
    # PDF Version
    # ======================================================

    def pdf_version(
        self,
    ) -> str:
        """
        Return PDF format version.
        """

        document = self.open_pdf()

        return str(

            document.pdf_version()

        )

    # ======================================================
    # String Representation
    # ======================================================

    def __repr__(
        self,
    ):

        return (

            "PDFLoader("

            f"file='{self.file_name}', "

            f"pages={self.page_count}"

            ")"

        )

    # ======================================================
    # Context Manager
    # ======================================================

    def __enter__(
        self,
    ):

        self.open_pdf()

        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):

        self.close_pdf()

        logger.info(

            f"Closed PDF "

            f"{self.file_name}"

        )
