from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


# ==========================================================
# Search Result
# ==========================================================

@dataclass(slots=True)
class SearchResult:
    """
    Represents a retrieved search result.
    """

    id: str

    score: float

    document: str

    metadata: Dict[str, Any] = field(default_factory=dict)

    embedding: Optional[List[float]] = None


# ==========================================================
# Vector Document
# ==========================================================

@dataclass(slots=True)
class VectorDocument:
    """
    Document stored inside the vector database.
    """

    id: str

    text: str

    embedding: Optional[List[float]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.utcnow)

    updated_at: datetime = field(default_factory=datetime.utcnow)


# ==========================================================
# Base Vector Store
# ==========================================================

class BaseVectorStore(ABC):
    """
    Abstract base class for all vector stores.
    """

    def __init__(
        self,
        collection_name: str,
    ):

        self.collection_name = collection_name

        logger.info(

            "%s initialized.",

            self.__class__.__name__,

        )

    # ======================================================
    # Collection
    # ======================================================

    @abstractmethod
    def create_collection(
        self,
    ) -> None:
        """
        Create a collection/index.
        """

    @abstractmethod
    def delete_collection(
        self,
    ) -> None:
        """
        Delete collection.
        """

    @abstractmethod
    def collection_exists(
        self,
    ) -> bool:
        """
        Check if collection exists.
        """

    # ======================================================
    # CRUD
    # ======================================================

    @abstractmethod
    def add_document(
        self,
        document: VectorDocument,
    ) -> None:
        """
        Insert one document.
        """

    @abstractmethod
    def add_documents(
        self,
        documents: List[VectorDocument],
    ) -> None:
        """
        Batch insert.
        """

    @abstractmethod
    def update_document(
        self,
        document_id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Update document.
        """

    @abstractmethod
    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete one document.
        """

    @abstractmethod
    def delete_documents(
        self,
        ids: List[str],
    ) -> None:
        """
        Delete multiple documents.
        """

    # ======================================================
    # Retrieval
    # ======================================================

    @abstractmethod
    def get_document(
        self,
        document_id: str,
    ) -> Optional[VectorDocument]:
        """
        Retrieve document.
        """

    @abstractmethod
    def get_documents(
        self,
        ids: Optional[List[str]] = None,
    ) -> List[VectorDocument]:
        """
        Retrieve documents.
        """

    @abstractmethod
    def count(
        self,
    ) -> int:
        """
        Number of stored documents.
        """
