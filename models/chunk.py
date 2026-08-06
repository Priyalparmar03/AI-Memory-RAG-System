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

# ======================================================
# Update Text
# ======================================================

def update_text(
    self,
    text: str,
) -> None:
    """
    Update chunk text.
    """

    if not text.strip():

        raise ChunkError(

            "Chunk text cannot "

            "be empty."

        )

    self.text = text

    self.refresh_hash()

    self.touch()


# ======================================================
# Set Embedding
# ======================================================

def set_embedding(
    self,
    embedding: List[float],
) -> None:
    """
    Set embedding vector.
    """

    if not isinstance(

        embedding,

        list,

    ):

        raise ChunkError(

            "Embedding must "

            "be a list."

        )

    self.embedding = embedding

    self.touch()


# ======================================================
# Clear Embedding
# ======================================================

def clear_embedding(
    self,
) -> None:
    """
    Remove embedding vector.
    """

    self.embedding = None

    self.touch()


# ======================================================
# Refresh Hash
# ======================================================

def refresh_hash(
    self,
) -> None:
    """
    Recalculate checksum.
    """

    self.checksum = (

        self.generate_hash()

    )

    self.touch()


# ======================================================
# Update Status
# ======================================================

def update_status(
    self,
    status: ChunkStatus,
) -> None:
    """
    Update chunk status.
    """

    self.status = status

    self.touch()


# ======================================================
# Mark Embedded
# ======================================================

def mark_embedded(
    self,
) -> None:
    """
    Mark chunk as embedded.
    """

    self.status = (

        ChunkStatus.EMBEDDED

    )

    self.touch()


# ======================================================
# Mark Indexed
# ======================================================

def mark_indexed(
    self,
) -> None:
    """
    Mark chunk as indexed.
    """

    self.status = (

        ChunkStatus.INDEXED

    )

    self.touch()


# ======================================================
# Mark Retrieved
# ======================================================

def mark_retrieved(
    self,
) -> None:
    """
    Mark chunk as retrieved.
    """

    self.status = (

        ChunkStatus.RETRIEVED

    )

    self.touch()


# ======================================================
# Mark Failed
# ======================================================

def mark_failed(
    self,
) -> None:
    """
    Mark chunk as failed.
    """

    self.status = (

        ChunkStatus.FAILED

    )

    self.touch()


# ======================================================
# Search Text
# ======================================================

def contains(
    self,
    keyword: str,
) -> bool:
    """
    Search keyword in chunk.
    """

    return (

        keyword.lower()

        in

        self.text.lower()

    )


# ======================================================
# Token Count
# ======================================================

@property
def token_count(
    self,
) -> int:
    """
    Approximate token count.
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
    Number of characters.
    """

    return len(

        self.text

    )


# ======================================================
# Word Count
# ======================================================

@property
def word_count(
    self,
) -> int:
    """
    Number of words.
    """

    return len(

        self.text.split()

    )


# ======================================================
# Embedding Dimension
# ======================================================

@property
def embedding_dimension(
    self,
) -> int:
    """
    Size of embedding vector.
    """

    if self.embedding is None:

        return 0

    return len(

        self.embedding

    )


# ======================================================
# Has Embedding
# ======================================================

@property
def has_embedding(
    self,
) -> bool:
    """
    Whether embedding exists.
    """

    return (

        self.embedding

        is not None

    )


# ======================================================
# Advanced Statistics
# ======================================================

def advanced_statistics(
    self,
) -> Dict[str, Any]:
    """
    Advanced chunk statistics.
    """

    return {

        **self.statistics(),

        "token_count":

            self.token_count,

        "embedding_dimension":

            self.embedding_dimension,

        "has_embedding":

            self.has_embedding,

        "metadata_entries":

            len(

                self.metadata

            ),

        "checksum":

            self.checksum,

        "offset_range":

            (

                self.start_offset,

                self.end_offset,

            ),

    }

# ======================================================
# JSON Serialization
# ======================================================

def to_json(
    self,
    indent: int = 4,
) -> str:
    """
    Serialize chunk to JSON.
    """

    return json.dumps(

        self.to_dict(),

        indent=indent,

        ensure_ascii=False,

    )


# ======================================================
# Create From Dictionary
# ======================================================

@classmethod
def from_dict(
    cls,
    data: Dict[str, Any],
) -> "Chunk":
    """
    Create Chunk from dictionary.
    """

    return cls(

        document_id=data["document_id"],

        chunk_index=data["chunk_index"],

        text=data["text"],

        chunk_type=ChunkType(

            data.get(

                "chunk_type",

                ChunkType.TEXT,

            )

        ),

        status=ChunkStatus(

            data.get(

                "status",

                ChunkStatus.NEW,

            )

        ),

        metadata=data.get(

            "metadata",

            {},

        ),

        embedding=data.get(

            "embedding"

        ),

        language=data.get(

            "language",

            "unknown",

        ),

        version=data.get(

            "version",

            "1.0",

        ),

        start_offset=data.get(

            "start_offset",

            0,

        ),

        end_offset=data.get(

            "end_offset",

            0,

        ),

        source_page=data.get(

            "source_page",

            1,

        ),

        checksum=data.get(

            "checksum",

            "",

        ),

        id=data.get(

            "id",

            str(

                uuid.uuid4()

            ),

        ),

        created_at=datetime.fromisoformat(

            data.get(

                "created_at",

                datetime.utcnow().isoformat(),

            )

        ),

        updated_at=datetime.fromisoformat(

            data.get(

                "updated_at",

                datetime.utcnow().isoformat(),

            )

        ),

    )


# ======================================================
# Create From JSON
# ======================================================

@classmethod
def from_json(
    cls,
    json_string: str,
) -> "Chunk":
    """
    Create Chunk from JSON.
    """

    return cls.from_dict(

        json.loads(

            json_string

        )

    )


# ======================================================
# Clone
# ======================================================

def clone(
    self,
) -> "Chunk":
    """
    Deep copy chunk.
    """

    return Chunk.from_dict(

        self.to_dict()

    )


# ======================================================
# Summary
# ======================================================

def summary(
    self,
) -> Dict[str, Any]:
    """
    Human-readable summary.
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

        "tokens":

            self.token_count,

        "words":

            self.word_count,

        "characters":

            self.character_count,

        "page":

            self.source_page,

    }


# ======================================================
# Diagnostics
# ======================================================

def diagnostics(
    self,
) -> Dict[str, Any]:
    """
    Chunk diagnostics.
    """

    return {

        "model":

            self.__class__.__name__,

        "id":

            self.id,

        "checksum":

            self.checksum,

        "created_at":

            self.created_at.isoformat(),

        "updated_at":

            self.updated_at.isoformat(),

        "statistics":

            self.advanced_statistics(),

    }


# ======================================================
# Export
# ======================================================

def export(
    self,
) -> Dict[str, Any]:
    """
    Export complete chunk.
    """

    return {

        "chunk":

            self.to_dict(),

        "summary":

            self.summary(),

        "statistics":

            self.advanced_statistics(),

        "diagnostics":

            self.diagnostics(),

    }


# ======================================================
# Preview
# ======================================================

@property
def preview(
    self,
) -> str:
    """
    Preview chunk text.
    """

    if len(

        self.text

    ) <= 250:

        return self.text

    return (

        self.text[:250]

        + "..."

    )


# ======================================================
# Analytics
# ======================================================

def analytics(
    self,
) -> Dict[str, Any]:
    """
    Chunk analytics.
    """

    return {

        "summary":

            self.summary(),

        "statistics":

            self.advanced_statistics(),

        "embedding_dimension":

            self.embedding_dimension,

        "language":

            self.language,

        "page":

            self.source_page,

        "offsets": {

            "start":

                self.start_offset,

            "end":

                self.end_offset,

        },

    }


# ======================================================
# Similarity Placeholder
# ======================================================

def similarity(
    self,
    other: "Chunk",
) -> float:
    """
    Placeholder similarity method.

    Override with cosine similarity
    implementation if embeddings
    are available.
    """

    if not isinstance(

        other,

        Chunk,

    ):

        raise ChunkError(

            "Expected Chunk."

        )

    if (

        self.checksum

        ==

        other.checksum

    ):

        return 1.0

    return 0.0


# ======================================================
# Compare
# ======================================================

def compare(
    self,
    other: "Chunk",
) -> Dict[str, Any]:
    """
    Compare two chunks.
    """

    if not isinstance(

        other,

        Chunk,

    ):

        raise ChunkError(

            "Expected Chunk."

        )

    return {

        "same_document":

            self.document_id

            ==

            other.document_id,

        "same_type":

            self.chunk_type

            ==

            other.chunk_type,

        "same_status":

            self.status

            ==

            other.status,

        "word_difference":

            abs(

                self.word_count

                -

                other.word_count

            ),

        "character_difference":

            abs(

                self.character_count

                -

                other.character_count

            ),

        "similarity":

            self.similarity(

                other

            ),

    }


# ======================================================
# Has Metadata
# ======================================================

@property
def has_metadata(
    self,
) -> bool:
    """
    Whether metadata exists.
    """

    return (

        len(

            self.metadata

        )

        >

        0

    )


# ======================================================
# Is Searchable
# ======================================================

@property
def is_searchable(
    self,
) -> bool:
    """
    Searchable chunk.
    """

    return (

        self.status

        !=

        ChunkStatus.DELETED

    )
