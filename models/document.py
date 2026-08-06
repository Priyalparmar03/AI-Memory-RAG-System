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
