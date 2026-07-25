"""
services/document_service.py

Production Document Service

Responsibilities
----------------
- Validate uploaded documents
- Save documents
- Detect file type
- Extract text
- Extract metadata
- Generate checksums
- Prepare documents for chunking
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from docx import Document

logger = logging.getLogger(__name__)


class DocumentService:

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".csv",
        ".html",
    }

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

    def __init__(
        self,
        upload_folder: str = "uploads",
    ):

        self.upload_folder = Path(upload_folder)

        self.upload_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "DocumentService initialized."
        )

    # =====================================================
    # Validate Upload
    # =====================================================

    def validate_file(
        self,
        file_path: Path,
    ) -> None:

        if not file_path.exists():

            raise FileNotFoundError(
                file_path
            )

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:

            raise ValueError(
                f"Unsupported file type: {file_path.suffix}"
            )

        if file_path.stat().st_size > self.MAX_FILE_SIZE:

            raise ValueError(
                "File exceeds maximum size."
            )

    # =====================================================
    # Generate Document ID
    # =====================================================

    @staticmethod
    def generate_document_id() -> str:

        return str(uuid.uuid4())

    # =====================================================
    # Checksum
    # =====================================================

    @staticmethod
    def checksum(
        file_path: Path,
    ) -> str:

        sha = hashlib.sha256()

        with open(file_path, "rb") as f:

            while True:

                chunk = f.read(8192)

                if not chunk:
                    break

                sha.update(chunk)

        return sha.hexdigest()

    # =====================================================
    # Save Uploaded File
    # =====================================================

    def save_document(
        self,
        uploaded_file,
    ) -> Dict:

        filename = uploaded_file.filename

        suffix = Path(filename).suffix

        document_id = self.generate_document_id()

        save_path = (
            self.upload_folder /
            f"{document_id}{suffix}"
        )

        uploaded_file.save(save_path)

        self.validate_file(save_path)

        logger.info(
            "Document saved: %s",
            filename,
        )

        return {

            "document_id": document_id,

            "filename": filename,

            "path": str(save_path),

            "checksum": self.checksum(save_path),

        }

    # =====================================================
    # MIME Type
    # =====================================================

    @staticmethod
    def mime_type(
        file_path: Path,
    ) -> str:

        mime, _ = mimetypes.guess_type(
            str(file_path)
        )

        return mime or "unknown"

    # =====================================================
    # Extract Metadata
    # =====================================================

    def metadata(
        self,
        file_path: Path,
    ) -> Dict:

        stat = file_path.stat()

        return {

            "filename":
                file_path.name,

            "extension":
                file_path.suffix,

            "size":
                stat.st_size,

            "created":
                datetime.fromtimestamp(
                    stat.st_ctime
                ),

            "modified":
                datetime.fromtimestamp(
                    stat.st_mtime
                ),

            "mime":
                self.mime_type(file_path),

            "checksum":
                self.checksum(file_path),

        }

    # =====================================================
    # Extract Text Dispatcher
    # =====================================================

    def extract_text(
        self,
        file_path: Path,
    ) -> str:

        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return self._extract_pdf(file_path)

        if extension == ".docx":
            return self._extract_docx(file_path)

        if extension == ".txt":
            return self._extract_txt(file_path)

        if extension == ".md":
            return self._extract_markdown(file_path)

        if extension == ".csv":
            return self._extract_csv(file_path)

        if extension == ".html":
            return self._extract_html(file_path)

        raise ValueError(
            f"Unsupported type: {extension}"
        )

    # =====================================================
    # PDF Loader
    # =====================================================

    @staticmethod
    def _extract_pdf(
        file_path: Path,
    ) -> str:

        text = []

        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text.append(page_text)

        return "\n".join(text)

    # =====================================================
    # DOCX Loader
    # =====================================================

    @staticmethod
    def _extract_docx(
        file_path: Path,
    ) -> str:

        document = Document(file_path)

        return "\n".join(

            paragraph.text

            for paragraph in document.paragraphs

            if paragraph.text.strip()

        )

    # =====================================================
    # TXT Loader
    # =====================================================

    @staticmethod
    def _extract_txt(
        file_path: Path,
    ) -> str:

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    # =====================================================
    # Markdown Loader
    # =====================================================

    @staticmethod
    def _extract_markdown(
        file_path: Path,
    ) -> str:

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    # =====================================================
    # CSV Loader
    # =====================================================

    @staticmethod
    def _extract_csv(
        file_path: Path,
    ) -> str:

        df = pd.read_csv(file_path)

        return df.to_csv(index=False)

    # =====================================================
    # HTML Loader
    # =====================================================

    @staticmethod
    def _extract_html(
        file_path: Path,
    ) -> str:

        html = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        return soup.get_text(
            separator="\n",
        )
