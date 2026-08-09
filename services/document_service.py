from __future__ import annotations
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.embedding_service import EmbeddingService
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

    # =====================================================
    # Clean Text
    # =====================================================

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text.
        """

        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove excessive spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # =====================================================
    # Create Text Splitter
    # =====================================================

    @staticmethod
    def create_splitter(
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> RecursiveCharacterTextSplitter:
        """
        Create Recursive Character Splitter.
        """

        return RecursiveCharacterTextSplitter(

            chunk_size=chunk_size,

            chunk_overlap=chunk_overlap,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    # =====================================================
    # Chunk Document
    # =====================================================

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> List[str]:
        """
        Split text into chunks.
        """

        splitter = self.create_splitter(

            chunk_size=chunk_size,

            chunk_overlap=chunk_overlap,

        )

        return splitter.split_text(text)

    # =====================================================
    # Build Chunk Metadata
    # =====================================================

    def build_chunk_metadata(
        self,
        document_id: str,
        filename: str,
        chunks: List[str],
    ) -> List[Dict]:
        """
        Create metadata for each chunk.
        """

        metadata = []

        total = len(chunks)

        for index, chunk in enumerate(chunks):

            metadata.append(

                {

                    "document_id": document_id,

                    "filename": filename,

                    "chunk_id": index,

                    "chunk_index": index,

                    "total_chunks": total,

                    "characters": len(chunk),

                    "words": len(chunk.split()),

                }

            )

        return metadata

    # =====================================================
    # Duplicate Detection
    # =====================================================

    def is_duplicate(
        self,
        file_path: Path,
        existing_checksums: List[str],
    ) -> bool:
        """
        Detect duplicate documents.
        """

        checksum = self.checksum(file_path)

        return checksum in existing_checksums

    # =====================================================
    # Generate Embeddings
    # =====================================================

    def generate_embeddings(
        self,
        chunks: List[str],
    ):
        """
        Generate embeddings for chunks.
        """

        embedding_service = EmbeddingService()

        return embedding_service.embed_documents(chunks)

    # =====================================================
    # Process Document
    # =====================================================

    def process_document(
        self,
        file_path: Path,
    ) -> Dict:
        """
        Complete document processing pipeline.
        """

        logger.info(
            "Processing %s",
            file_path.name,
        )

        self.validate_file(file_path)

        metadata = self.metadata(file_path)

        text = self.extract_text(file_path)

        text = self.clean_text(text)

        chunks = self.chunk_text(text)

        embeddings = self.generate_embeddings(chunks)

        chunk_metadata = self.build_chunk_metadata(

            document_id=self.generate_document_id(),

            filename=file_path.name,

            chunks=chunks,

        )

        return {

            "metadata": metadata,

            "text": text,

            "chunks": chunks,

            "chunk_metadata": chunk_metadata,

            "embeddings": embeddings,

        }
    # =====================================================
    # Delete Document
    # =====================================================

    def delete_document(
        self,
        file_path: Path,
    ) -> bool:
        """
        Delete a document from storage.
        """

        try:

            if file_path.exists():

                file_path.unlink()

                logger.info(
                    "Deleted document: %s",
                    file_path.name,
                )

                return True

            return False

        except Exception as exc:

            logger.exception(exc)

            return False

    # =====================================================
    # List Documents
    # =====================================================

    def list_documents(self) -> List[Dict]:
        """
        List all uploaded documents.
        """

        documents = []

        for file in self.upload_folder.iterdir():

            if not file.is_file():
                continue

            try:

                documents.append(
                    self.metadata(file)
                )

            except Exception:

                logger.exception(
                    "Failed reading metadata for %s",
                    file.name,
                )

        return documents

    # =====================================================
    # Search Documents
    # =====================================================

    def search_documents(
        self,
        keyword: str,
    ) -> List[Dict]:
        """
        Search documents by filename.
        """

        keyword = keyword.lower()

        results = []

        for document in self.list_documents():

            filename = document["filename"].lower()

            if keyword in filename:

                results.append(document)

        return results

    # =====================================================
    # Storage Statistics
    # =====================================================

    def statistics(self) -> Dict:
        """
        Storage statistics.
        """

        documents = self.list_documents()

        total_size = sum(

            doc["size"]

            for doc in documents

        )

        return {

            "documents": len(documents),

            "total_storage_bytes": total_size,

            "total_storage_mb":
                round(total_size / (1024 * 1024), 2),

            "supported_formats":
                list(self.SUPPORTED_EXTENSIONS),

        }

    # =====================================================
    # Health Check
    # =====================================================

    def health(self) -> Dict:
        """
        Verify storage availability.
        """

        writable = self.upload_folder.exists()

        return {

            "status":
                "healthy" if writable else "unhealthy",

            "upload_folder":
                str(self.upload_folder),

            "exists":
                self.upload_folder.exists(),

            "writable":
                writable,

        }

    # =====================================================
    # Cleanup Empty Files
    # =====================================================

    def cleanup_empty_files(self) -> int:
        """
        Remove zero-byte files.
        """

        removed = 0

        for file in self.upload_folder.iterdir():

            if not file.is_file():
                continue

            if file.stat().st_size == 0:

                file.unlink()

                removed += 1

        logger.info(
            "Removed %d empty files.",
            removed,
        )

        return removed

    # =====================================================
    # Clear Upload Directory
    # =====================================================

    def clear_storage(self):
        """
        Delete every uploaded document.
        """

        removed = 0

        for file in self.upload_folder.iterdir():

            if not file.is_file():
                continue

            file.unlink()

            removed += 1

        logger.info(
            "Storage cleared (%d files).",
            removed,
        )

        return removed

    # =====================================================
    # Export Metadata
    # =====================================================

    def export_metadata(self) -> List[Dict]:
        """
        Export metadata for all documents.
        """

        return self.list_documents()

    # =====================================================
    # Build Index Payload
    # =====================================================

    def build_index_payload(
        self,
        processed_document: Dict,
    ) -> List[Dict]:
        """
        Convert processed document into
        a payload ready for RagService.
        """

        payload = []

        chunks = processed_document["chunks"]

        metadata = processed_document["chunk_metadata"]

        embeddings = processed_document["embeddings"]

        for chunk, meta, embedding in zip(
            chunks,
            metadata,
            embeddings,
        ):

            payload.append(

                {

                    "content": chunk,

                    "embedding": embedding,

                    "metadata": meta,

                }

            )

        return payload

    # =====================================================
    # Supported Formats
    # =====================================================

    @classmethod
    def supported_formats(cls) -> List[str]:
        """
        Return supported extensions.
        """

        return sorted(
            cls.SUPPORTED_EXTENSIONS
        )

    # =====================================================
    # Service Information
    # =====================================================

    def info(self) -> Dict:
        """
        General service information.
        """

        return {

            "service": "DocumentService",

            "version": "1.0.0",

            "upload_folder":
                str(self.upload_folder),

            "supported_formats":
                self.supported_formats(),

            "max_file_size_mb":
                self.MAX_FILE_SIZE / (1024 * 1024),

        }
