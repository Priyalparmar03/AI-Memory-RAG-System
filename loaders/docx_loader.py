from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from docx import Document

from .base_loader import BaseLoader, LoaderError

logger = logging.getLogger(__name__)


# ==========================================================
# DOCX Loader
# ==========================================================

class DOCXLoader(BaseLoader):
    """
    Production DOCX Loader.

    Supported Formats
    -----------------
    .docx
    """

    SUPPORTED_EXTENSIONS = [

        ".docx",

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
            Document
        ] = None

        logger.info(

            f"DOCXLoader initialized "

            f"for {self.file_name}"

        )

    # ======================================================
    # Open Document
    # ======================================================

    def open_document(
        self,
    ) -> Document:
        """
        Open DOCX document.
        """

        if self.document is None:

            try:

                self.document = Document(

                    self.file_path

                )

            except Exception as error:

                raise LoaderError(

                    f"Unable to open DOCX: "

                    f"{error}"

                )

        return self.document

    # ======================================================
    # Close Document
    # ======================================================

    def close_document(
        self,
    ) -> None:
        """
        Release document.
        """

        self.document = None

    # ======================================================
    # Extract Text
    # ======================================================

    def extract_text(
        self,
    ) -> str:
        """
        Extract complete text.
        """

        document = self.open_document()

        paragraphs = [

            paragraph.text

            for paragraph

            in document.paragraphs

            if paragraph.text.strip()

        ]

        return "\n".join(

            paragraphs

        )

    # ======================================================
    # Paragraphs
    # ======================================================

    def paragraphs(
        self,
    ) -> List[str]:
        """
        Return paragraph list.
        """

        document = self.open_document()

        return [

            paragraph.text

            for paragraph

            in document.paragraphs

        ]

    # ======================================================
    # Metadata
    # ======================================================

    def document_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Basic metadata.
        """

        document = self.open_document()

        properties = (

            document.core_properties

        )

        return {

            "title":

                properties.title,

            "author":

                properties.author,

            "subject":

                properties.subject,

            "category":

                properties.category,

            "keywords":

                properties.keywords,

            "comments":

                properties.comments,

            "created":

                properties.created,

            "modified":

                properties.modified,

            "last_modified_by":

                properties.last_modified_by,

            "revision":

                properties.revision,

            "paragraphs":

                len(

                    document.paragraphs

                ),

            "sections":

                len(

                    document.sections

                ),

        }

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Document statistics.
        """

        text = self.extract_text()

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

            "paragraphs":

                len(

                    self.paragraphs()

                ),

        }

    # ======================================================
    # Load
    # ======================================================

    def load(
        self,
    ) -> Dict[str, Any]:
        """
        Load DOCX document.
        """

        text = self.extract_text()

        metadata = self.document_metadata()

        metadata.update(

            self.statistics()

        )

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
        Preview document.
        """

        return self.extract_text()[:characters]

    # ======================================================
    # Empty Check
    # ======================================================

    def is_empty(
        self,
    ) -> bool:
        """
        Check empty document.
        """

        return (

            len(

                self.extract_text().strip()

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

            "DOCXLoader("

            f"file='{self.file_name}', "

            f"paragraphs={len(self.paragraphs())}"

            ")"

        )
