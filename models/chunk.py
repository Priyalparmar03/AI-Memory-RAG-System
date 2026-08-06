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

from typing import (
    Any,
    Dict,
    List,
    Optional,
)


logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class ChunkError(Exception):
    """
    Chunk model exception.
    """
    pass


# ==========================================================
# Chunk Status
# ==========================================================

class ChunkStatus(str, Enum):

    NEW = "new"

    EMBEDDED = "embedded"

    INDEXED = "indexed"

    RETRIEVED = "retrieved"

    FAILED = "failed"

    DELETED = "deleted"


# ==========================================================
# Chunk Type
# ==========================================================

class ChunkType(str, Enum):

    TEXT = "text"

    TABLE = "table"

    IMAGE = "image"

    CODE = "code"

    MARKDOWN = "markdown"

    METADATA = "metadata"


# ==========================================================
# Chunk
# ==========================================================

@dataclass(slots=True)
class Chunk:
    """
    Production Chunk Model.
    """

    document_id: str

    chunk_index: int

    text: str

    chunk_type: ChunkType = (

        ChunkType.TEXT

    )

    status: ChunkStatus = (

        ChunkStatus.NEW

    )

    metadata: Dict[str, Any] = field(

        default_factory=dict

    )

    embedding: Optional[List[float]] = None

    language: str = "unknown"

    version: str = "1.0"

    start_offset: int = 0

    end_offset: int = 0

    source_page: int = 1

    checksum: str = ""

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
        Validate chunk.
        """

        if not self.document_id.strip():

            raise ChunkError(

                "Document ID cannot "

                "be empty."

            )

        if self.chunk_index < 0:

            raise ChunkError(

                "Chunk index cannot "

                "be negative."

            )

        if not self.text.strip():

            raise ChunkError(

                "Chunk text cannot "

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
    # Rename Language
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
    # Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Chunk statistics.
        """

        return {

            "id":

                self.id,

            "document_id":

                self.document_id,

            "chunk_index":

                self.chunk_index,

            "chunk_type":

                self.chunk_type.value,

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

            "language":

                self.language,

            "source_page":

                self.source_page,

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
        Serialize chunk.
        """

        return {

            "id":

                self.id,

            "document_id":

                self.document_id,

            "chunk_index":

                self.chunk_index,

            "text":

                self.text,

            "chunk_type":

                self.chunk_type.value,

            "status":

                self.status.value,

            "metadata":

                self.metadata,

            "embedding":

                self.embedding,

            "language":

                self.language,

            "version":

                self.version,

            "start_offset":

                self.start_offset,

            "end_offset":

                self.end_offset,

            "source_page":

                self.source_page,

            "checksum":

                self.checksum,

            "created_at":

                self.created_at.isoformat(),

            "updated_at":

                self.updated_at.isoformat(),

        }
