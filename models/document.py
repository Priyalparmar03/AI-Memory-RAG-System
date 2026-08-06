from __future__ import annotations

import hashlib
import json
import logging
import uuid

from dataclasses import (
    dataclass,
    field,
)

from datetime import datetime

from enum import Enum

from pathlib import Path

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from .chunk import Chunk


logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class DocumentError(Exception):
    """
    Document model exception.
    """
    pass


# ==========================================================
# Document Status
# ==========================================================

class DocumentStatus(str, Enum):

    NEW = "new"

    PROCESSED = "processed"

    INDEXED = "indexed"

    EMBEDDED = "embedded"

    FAILED = "failed"

    DELETED = "deleted"


# ==========================================================
# Document Type
# ==========================================================

class DocumentType(str, Enum):

    PDF = "pdf"

    DOCX = "docx"

    TXT = "txt"

    CSV = "csv"

    MARKDOWN = "markdown"

    IMAGE = "image"

    HTML = "html"

    JSON = "json"

    UNKNOWN = "unknown"


# ==========================================================
# Document
# ==========================================================

@dataclass(slots=True)
class Document:
    """
    Production Document Model.
    """

    file_name: str

    file_path: str

    text: str

    document_type: DocumentType = (

        DocumentType.UNKNOWN

    )

    status: DocumentStatus = (

        DocumentStatus.NEW

    )

    metadata: Dict[str, Any] = field(

        default_factory=dict

    )

    chunks: List[Chunk] = field(

        default_factory=list

    )

    language: str = "unknown"

    loader: str = ""

    version: str = "1.0"

    embedding_model: str = ""

    id: str = field(

        default_factory=lambda:

        str(uuid.uuid4())

    )

    created_at: datetime = field(

        default_factory=datetime.utcnow

    )

    updated_at: datetime = field(

        default_factory=datetime.utcnow

    )

    indexed_at: Optional[

        datetime

    ] = None

    checksum: str = ""

    tags: List[str] = field(

        default_factory=list

    )

    source: str = ""

    is_public: bool = False


    # ======================================================
    # Initialization
    # ======================================================

    def __post_init__(
        self,
    ):

        self.validate()

        if not self.checksum:

            self.checksum = (

                self.generate_hash()

            )


    # ======================================================
    # Validation
    # ======================================================

    def validate(
        self,
    ) -> None:
        """
        Validate document.
        """

        if not self.file_name.strip():

            raise DocumentError(

                "File name cannot "

                "be empty."

            )

        if not self.file_path.strip():

            raise DocumentError(

                "File path cannot "

                "be empty."

            )


    # ======================================================
    # Generate Hash
    # ======================================================

    def generate_hash(
        self,
    ) -> str:
        """
        Generate SHA-256 checksum.
        """

        return hashlib.sha256(

            self.text.encode(

                "utf-8"

            )

        ).hexdigest()


    # ======================================================
    # Update Timestamp
    # ======================================================

    def touch(
        self,
    ) -> None:
        """
        Update timestamps.
        """

        self.updated_at = (

            datetime.utcnow()

        )


    # ======================================================
    # Rename Document
    # ======================================================

    def rename(
        self,
        file_name: str,
    ) -> None:
        """
        Rename document.
        """

        if not file_name.strip():

            raise DocumentError(

                "Invalid filename."

            )

        self.file_name = file_name

        self.touch()


    # ======================================================
    # Update Metadata
    # ======================================================

    def update_metadata(
        self,
        **kwargs,
    ) -> None:
        """
        Update metadata.
        """

        self.metadata.update(

            kwargs

        )

        self.touch()


    # ======================================================
    # Add Tag
    # ======================================================

    def add_tag(
        self,
        tag: str,
    ) -> None:
        """
        Add document tag.
        """

        if (

            tag

            and

            tag not in self.tags

        ):

            self.tags.append(

                tag

            )

            self.touch()


    # ======================================================
    # Remove Tag
    # ======================================================

    def remove_tag(
        self,
        tag: str,
    ) -> None:
        """
        Remove document tag.
        """

        if tag in self.tags:

            self.tags.remove(

                tag

            )

            self.touch()


    # ======================================================
    # Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Basic document statistics.
        """

        return {

            "id":

                self.id,

            "file_name":

                self.file_name,

            "document_type":

                self.document_type.value,

            "status":

                self.status.value,

            "characters":

                len(

                    self.text

                ),

            "words":

                len(

                    self.text.split()

                ),

            "chunks":

                len(

                    self.chunks

                ),

            "language":

                self.language,

            "loader":

                self.loader,

            "version":

                self.version,

            "created_at":

                self.created_at.isoformat(),

            "updated_at":

                self.updated_at.isoformat(),

        }


    # ======================================================
    # Serialization
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize document.
        """

        return {

            "id":

                self.id,

            "file_name":

                self.file_name,

            "file_path":

                self.file_path,

            "text":

                self.text,

            "document_type":

                self.document_type.value,

            "status":

                self.status.value,

            "metadata":

                self.metadata,

            "chunks":

                [

                    chunk.to_dict()

                    for chunk

                    in self.chunks

                ],

            "language":

                self.language,

            "loader":

                self.loader,

            "version":

                self.version,

            "embedding_model":

                self.embedding_model,

            "checksum":

                self.checksum,

            "tags":

                self.tags,

            "source":

                self.source,

            "is_public":

                self.is_public,

            "created_at":

                self.created_at.isoformat(),

            "updated_at":

                self.updated_at.isoformat(),

            "indexed_at":

                self.indexed_at.isoformat()

                if self.indexed_at

                else None,

        }

# ======================================================
# Add Chunk
# ======================================================

def add_chunk(
    self,
    chunk: Chunk,
) -> None:
    """
    Add a chunk to the document.
    """

    if not isinstance(

        chunk,

        Chunk,

    ):

        raise DocumentError(

            "Expected Chunk object."

        )

    self.chunks.append(

        chunk

    )

    self.touch()


# ======================================================
# Remove Chunk
# ======================================================

def remove_chunk(
    self,
    chunk_id: str,
) -> bool:
    """
    Remove chunk by ID.
    """

    for index, chunk in enumerate(

        self.chunks

    ):

        if chunk.id == chunk_id:

            del self.chunks[index]

            self.touch()

            return True

    return False


# ======================================================
# Get Chunk
# ======================================================

def get_chunk(
    self,
    chunk_id: str,
) -> Optional[Chunk]:
    """
    Get chunk by ID.
    """

    for chunk in self.chunks:

        if chunk.id == chunk_id:

            return chunk

    return None


# ======================================================
# Clear Chunks
# ======================================================

def clear_chunks(
    self,
) -> None:
    """
    Remove all chunks.
    """

    self.chunks.clear()

    self.touch()


# ======================================================
# Chunk Count
# ======================================================

@property
def chunk_count(
    self,
) -> int:
    """
    Total number of chunks.
    """

    return len(

        self.chunks

    )


# ======================================================
# Search Chunks
# ======================================================

def search_chunks(
    self,
    keyword: str,
) -> List[Chunk]:
    """
    Search keyword inside chunks.
    """

    keyword = keyword.lower()

    results = []

    for chunk in self.chunks:

        if keyword in chunk.text.lower():

            results.append(

                chunk

            )

    return results


# ======================================================
# Update Language
# ======================================================

def set_language(
    self,
    language: str,
) -> None:
    """
    Update detected language.
    """

    self.language = language

    self.touch()


# ======================================================
# Update Embedding Model
# ======================================================

def set_embedding_model(
    self,
    model_name: str,
) -> None:
    """
    Set embedding model.
    """

    self.embedding_model = model_name

    self.touch()


# ======================================================
# Mark Indexed
# ======================================================

def mark_indexed(
    self,
) -> None:
    """
    Mark document as indexed.
    """

    self.status = (

        DocumentStatus.INDEXED

    )

    self.indexed_at = (

        datetime.utcnow()

    )

    self.touch()


# ======================================================
# Mark Embedded
# ======================================================

def mark_embedded(
    self,
) -> None:
    """
    Mark document as embedded.
    """

    self.status = (

        DocumentStatus.EMBEDDED

    )

    self.touch()


# ======================================================
# Mark Failed
# ======================================================

def mark_failed(
    self,
) -> None:
    """
    Mark document as failed.
    """

    self.status = (

        DocumentStatus.FAILED

    )

    self.touch()


# ======================================================
# Refresh Hash
# ======================================================

def refresh_hash(
    self,
) -> None:
    """
    Recompute checksum.
    """

    self.checksum = (

        self.generate_hash()

    )

    self.touch()


# ======================================================
# Word Count
# ======================================================

@property
def word_count(
    self,
) -> int:
    """
    Total words.
    """

    return len(

        self.text.split()

    )


# ======================================================
# Character Count
# ======================================================

@property
def character_count(
    self,
) -> int:
    """
    Total characters.
    """

    return len(

        self.text

    )


# ======================================================
# Line Count
# ======================================================

@property
def line_count(
    self,
) -> int:
    """
    Total lines.
    """

    return len(

        self.text.splitlines()

    )


# ======================================================
# Has Chunks
# ======================================================

@property
def has_chunks(
    self,
) -> bool:
    """
    Whether chunks exist.
    """

    return (

        len(

            self.chunks

        )

        >

        0

    )


# ======================================================
# Search Raw Text
# ======================================================

def search(
    self,
    keyword: str,
) -> bool:
    """
    Search raw document text.
    """

    return (

        keyword.lower()

        in

        self.text.lower()

    )


# ======================================================
# File Extension
# ======================================================

@property
def extension(
    self,
) -> str:
    """
    Document file extension.
    """

    return Path(

        self.file_name

    ).suffix.lower()


# ======================================================
# Advanced Statistics
# ======================================================

def advanced_statistics(
    self,
) -> Dict[str, Any]:
    """
    Advanced document statistics.
    """

    return {

        **self.statistics(),

        "lines":

            self.line_count,

        "characters":

            self.character_count,

        "words":

            self.word_count,

        "chunk_count":

            self.chunk_count,

        "tag_count":

            len(

                self.tags

            ),

        "checksum":

            self.checksum,

        "extension":

            self.extension,

        "is_indexed":

            self.status

            ==

            DocumentStatus.INDEXED,

        "is_embedded":

            self.status

            ==

            DocumentStatus.EMBEDDED,

    }
