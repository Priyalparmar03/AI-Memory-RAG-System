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

# ======================================================
# Extract Headings
# ======================================================

def extract_headings(
    self,
) -> List[Dict[str, Any]]:
    """
    Extract document headings.
    """

    document = self.open_document()

    headings = []

    for paragraph in document.paragraphs:

        style = paragraph.style.name

        if style.startswith("Heading"):

            try:

                level = int(

                    style.split()[-1]

                )

            except Exception:

                level = 1

            headings.append(

                {

                    "level": level,

                    "text": paragraph.text,

                    "style": style,

                }

            )

    return headings


# ======================================================
# Extract Tables
# ======================================================

def extract_tables(
    self,
) -> List[List[List[str]]]:
    """
    Extract all tables.
    """

    document = self.open_document()

    tables = []

    for table in document.tables:

        rows = []

        for row in table.rows:

            rows.append(

                [

                    cell.text.strip()

                    for cell

                    in row.cells

                ]

            )

        tables.append(

            rows

        )

    return tables


# ======================================================
# Extract Images
# ======================================================

def extract_images(
    self,
) -> List[Dict[str, Any]]:
    """
    Extract image metadata.
    """

    document = self.open_document()

    images = []

    relationships = document.part.rels

    image_number = 0

    for rel in relationships.values():

        if "image" in rel.target_ref:

            image_number += 1

            images.append(

                {

                    "id": image_number,

                    "name":

                        Path(

                            rel.target_ref

                        ).name,

                    "path":

                        rel.target_ref,

                }

            )

    return images


# ======================================================
# Extract Hyperlinks
# ======================================================

def extract_hyperlinks(
    self,
) -> List[Dict[str, str]]:
    """
    Extract hyperlinks.
    """

    document = self.open_document()

    links = []

    relationships = document.part.rels

    for rel in relationships.values():

        if (

            "hyperlink"

            in rel.reltype

        ):

            links.append(

                {

                    "url":

                        rel.target_ref,

                }

            )

    return links


# ======================================================
# Extract Headers
# ======================================================

def extract_headers(
    self,
) -> List[str]:
    """
    Extract document headers.
    """

    document = self.open_document()

    headers = []

    for section in document.sections:

        header = section.header

        for paragraph in header.paragraphs:

            if paragraph.text.strip():

                headers.append(

                    paragraph.text

                )

    return headers


# ======================================================
# Extract Footers
# ======================================================

def extract_footers(
    self,
) -> List[str]:
    """
    Extract document footers.
    """

    document = self.open_document()

    footers = []

    for section in document.sections:

        footer = section.footer

        for paragraph in footer.paragraphs:

            if paragraph.text.strip():

                footers.append(

                    paragraph.text

                )

    return footers


# ======================================================
# Search Text
# ======================================================

def search(
    self,
    keyword: str,
) -> List[Dict[str, Any]]:
    """
    Search document.
    """

    keyword = keyword.lower()

    results = []

    for index, paragraph in enumerate(

        self.paragraphs(),

        start=1,

    ):

        if (

            keyword

            in

            paragraph.lower()

        ):

            results.append(

                {

                    "paragraph":

                        index,

                    "text":

                        paragraph,

                }

            )

    return results


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
# Table Count
# ======================================================

def table_count(
    self,
) -> int:
    """
    Number of tables.
    """

    return len(

        self.extract_tables()

    )


# ======================================================
# Image Count
# ======================================================

def image_count(
    self,
) -> int:
    """
    Number of embedded images.
    """

    return len(

        self.extract_images()

    )


# ======================================================
# Hyperlink Count
# ======================================================

def hyperlink_count(
    self,
) -> int:
    """
    Number of hyperlinks.
    """

    return len(

        self.extract_hyperlinks()

    )
