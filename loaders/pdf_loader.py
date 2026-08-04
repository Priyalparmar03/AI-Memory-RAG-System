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

# ======================================================
# Extract Pages
# ======================================================

def extract_pages(
    self,
) -> List[Dict[str, Any]]:
    """
    Extract page-wise text.
    """

    document = self.open_pdf()

    pages = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):

        pages.append(

            {

                "page": page_number,

                "text": page.get_text(),

                "width": page.rect.width,

                "height": page.rect.height,

                "rotation": page.rotation,

            }

        )

    return pages


# ======================================================
# Extract Metadata
# ======================================================

def extract_metadata(
    self,
) -> Dict[str, Any]:
    """
    Extended PDF metadata.
    """

    document = self.open_pdf()

    metadata = self.document_info()

    metadata.update(

        {

            "is_pdf":

                True,

            "page_count":

                self.page_count,

            "file_size":

                self.file_size,

            "mime_type":

                self.mime_type,

        }

    )

    return metadata


# ======================================================
# Extract Tables
# ======================================================

def extract_tables(
    self,
) -> List[Dict[str, Any]]:
    """
    Extract tables using pdfplumber.
    """

    try:

        import pdfplumber

    except ImportError:

        raise LoaderError(

            "Install pdfplumber."

        )

    tables = []

    with pdfplumber.open(
        self.file_path
    ) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1,
        ):

            extracted = page.extract_tables()

            for index, table in enumerate(
                extracted,
                start=1,
            ):

                tables.append(

                    {

                        "page": page_number,

                        "table": index,

                        "rows": table,

                    }

                )

    return tables


# ======================================================
# Extract Links
# ======================================================

def extract_links(
    self,
) -> List[Dict[str, Any]]:
    """
    Extract hyperlinks.
    """

    document = self.open_pdf()

    links = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):

        for link in page.get_links():

            uri = link.get("uri")

            if uri:

                links.append(

                    {

                        "page": page_number,

                        "url": uri,

                    }

                )

    return links


# ======================================================
# Extract Outline
# ======================================================

def extract_outline(
    self,
) -> List[Dict[str, Any]]:
    """
    Extract PDF bookmarks.
    """

    document = self.open_pdf()

    toc = document.get_toc()

    outline = []

    for level, title, page in toc:

        outline.append(

            {

                "level": level,

                "title": title,

                "page": page,

            }

        )

    return outline


# ======================================================
# Search Text
# ======================================================

def search_text(
    self,
    keyword: str,
) -> List[Dict[str, Any]]:
    """
    Search keyword in PDF.
    """

    document = self.open_pdf()

    results = []

    keyword = keyword.lower()

    for page_number, page in enumerate(
        document,
        start=1,
    ):

        text = page.get_text()

        if keyword in text.lower():

            results.append(

                {

                    "page": page_number,

                    "matches": len(

                        page.search_for(
                            keyword
                        )

                    ),

                    "preview": text[
                        :300
                    ],

                }

            )

    return results


# ======================================================
# Extract Page
# ======================================================

def extract_page(
    self,
    page_number: int,
) -> Dict[str, Any]:
    """
    Extract one page.
    """

    document = self.open_pdf()

    if (

        page_number < 1

        or

        page_number > self.page_count

    ):

        raise LoaderError(

            "Invalid page number."

        )

    page = document[
        page_number - 1
    ]

    return {

        "page": page_number,

        "text": page.get_text(),

        "width": page.rect.width,

        "height": page.rect.height,

        "rotation": page.rotation,

    }


# ======================================================
# Extract Page Range
# ======================================================

def extract_page_range(
    self,
    start_page: int,
    end_page: int,
) -> List[Dict[str, Any]]:
    """
    Extract page range.
    """

    if start_page > end_page:

        raise LoaderError(

            "Invalid page range."

        )

    pages = []

    for page in range(
        start_page,
        end_page + 1,
    ):

        pages.append(

            self.extract_page(
                page
            )

        )

    return pages


# ======================================================
# Text Per Page
# ======================================================

def text_per_page(
    self,
) -> Dict[int, str]:
    """
    Dictionary of page -> text.
    """

    pages = {}

    for page in self.extract_pages():

        pages[
            page["page"]
        ] = page["text"]

    return pages


# ======================================================
# Table Count
# ======================================================

def table_count(
    self,
) -> int:
    """
    Total tables.
    """

    return len(

        self.extract_tables()

    )


# ======================================================
# Link Count
# ======================================================

def link_count(
    self,
) -> int:
    """
    Total hyperlinks.
    """

    return len(

        self.extract_links()

    )

# ======================================================
# Extract Images
# ======================================================

def extract_images(
    self,
) -> List[Dict[str, Any]]:
    """
    Extract all images from PDF.
    """

    document = self.open_pdf()

    images = []

    image_id = 0

    for page_number in range(
        len(document)
    ):

        page = document[page_number]

        image_list = page.get_images(
            full=True
        )

        for image in image_list:

            image_id += 1

            xref = image[0]

            pix = fitz.Pixmap(
                document,
                xref,
            )

            images.append(

                {

                    "image_id": image_id,

                    "page": page_number + 1,

                    "xref": xref,

                    "width": pix.width,

                    "height": pix.height,

                    "colorspace": pix.colorspace.name
                    if pix.colorspace
                    else None,

                }

            )

            pix = None

    return images


# ======================================================
# Save Images
# ======================================================

def save_images(
    self,
    output_dir: str,
) -> List[str]:
    """
    Save extracted images.
    """

    document = self.open_pdf()

    output = Path(output_dir)

    output.mkdir(

        parents=True,

        exist_ok=True,

    )

    saved = []

    counter = 0

    for page_number in range(
        len(document)
    ):

        page = document[
            page_number
        ]

        image_list = page.get_images(
            full=True
        )

        for image in image_list:

            counter += 1

            xref = image[0]

            pix = fitz.Pixmap(
                document,
                xref,
            )

            filename = (

                output

                /

                f"page_{page_number+1}_"

                f"image_{counter}.png"

            )

            if pix.n < 5:

                pix.save(filename)

            else:

                rgb = fitz.Pixmap(
                    fitz.csRGB,
                    pix,
                )

                rgb.save(filename)

                rgb = None

            saved.append(

                str(filename)

            )

            pix = None

    return saved


# ======================================================
# Render Page
# ======================================================

def render_page(
    self,
    page_number: int,
    zoom: float = 2.0,
):
    """
    Render page as image.
    """

    document = self.open_pdf()

    if (

        page_number < 1

        or

        page_number > self.page_count

    ):

        raise LoaderError(

            "Invalid page number."

        )

    page = document[
        page_number - 1
    ]

    matrix = fitz.Matrix(

        zoom,

        zoom,

    )

    pixmap = page.get_pixmap(

        matrix=matrix

    )

    return pixmap


# ======================================================
# Save Rendered Page
# ======================================================

def save_page(
    self,
    page_number: int,
    output_path: str,
    zoom: float = 2.0,
) -> str:
    """
    Save rendered page.
    """

    pix = self.render_page(

        page_number,

        zoom,

    )

    pix.save(

        output_path

    )

    return output_path


# ======================================================
# OCR Page
# ======================================================

def ocr_page(
    self,
    page_number: int,
) -> str:
    """
    OCR one PDF page.
    """

    try:

        import pytesseract

        from PIL import Image

    except ImportError:

        raise LoaderError(

            "Install pytesseract "

            "and pillow."

        )

    pix = self.render_page(
        page_number
    )

    image = Image.frombytes(

        "RGB",

        [

            pix.width,

            pix.height,

        ],

        pix.samples,

    )

    return pytesseract.image_to_string(
        image
    )


# ======================================================
# OCR Entire Document
# ======================================================

def ocr_document(
    self,
) -> str:
    """
    OCR every page.
    """

    text = []

    for page in range(

        1,

        self.page_count + 1,

    ):

        text.append(

            self.ocr_page(
                page
            )

        )

    return "\n".join(text)


# ======================================================
# Thumbnail
# ======================================================

def thumbnail(
    self,
    page_number: int = 1,
    width: int = 250,
):
    """
    Generate page thumbnail.
    """

    document = self.open_pdf()

    page = document[
        page_number - 1
    ]

    scale = (

        width

        /

        page.rect.width

    )

    matrix = fitz.Matrix(

        scale,

        scale,

    )

    return page.get_pixmap(

        matrix=matrix

    )


# ======================================================
# Save Thumbnail
# ======================================================

def save_thumbnail(
    self,
    output_path: str,
    page_number: int = 1,
) -> str:
    """
    Save thumbnail.
    """

    thumb = self.thumbnail(

        page_number

    )

    thumb.save(

        output_path

    )

    return output_path


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
# Is Scanned PDF
# ======================================================

def is_scanned(
    self,
) -> bool:
    """
    Detect scanned PDF.
    """

    text = self.extract_text()

    return len(

        text.strip()

    ) == 0
