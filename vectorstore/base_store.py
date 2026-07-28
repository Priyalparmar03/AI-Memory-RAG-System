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

    # ======================================================
    # Similarity Search
    # ======================================================

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search using an embedding vector.
        """

    # ======================================================
    # Text Search
    # ======================================================

    @abstractmethod
    def similarity_search_text(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search using raw text.
        Implementations should embed the query internally.
        """

    # ======================================================
    # Hybrid Search
    # ======================================================

    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 5,
        alpha: float = 0.5,
    ) -> List[SearchResult]:
        """
        Hybrid search combining lexical and vector similarity.
        """

    # ======================================================
    # Metadata Search
    # ======================================================

    @abstractmethod
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
    ) -> List[VectorDocument]:
        """
        Search documents using metadata filters.
        """

    # ======================================================
    # Range Search
    # ======================================================

    @abstractmethod
    def range_search(
        self,
        query_embedding: List[float],
        radius: float,
    ) -> List[SearchResult]:
        """
        Return all vectors within a similarity/distance threshold.
        """

    # ======================================================
    # Maximum Marginal Relevance (MMR)
    # ======================================================

    @abstractmethod
    def mmr_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> List[SearchResult]:
        """
        Diversity-aware retrieval.
        """

    # ======================================================
    # Batch Search
    # ======================================================

    @abstractmethod
    def batch_search(
        self,
        query_embeddings: List[List[float]],
        k: int = 5,
    ) -> List[List[SearchResult]]:
        """
        Search multiple queries simultaneously.
        """

    # ======================================================
    # Search by IDs
    # ======================================================

    @abstractmethod
    def get_by_ids(
        self,
        ids: List[str],
    ) -> List[VectorDocument]:
        """
        Retrieve documents using IDs.
        """

    # ======================================================
    # Collection Statistics
    # ======================================================

    @abstractmethod
    def collection_statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Statistics about the collection.
        """

    # ======================================================
    # List Collections
    # ======================================================

    @abstractmethod
    def list_collections(
        self,
    ) -> List[str]:
        """
        Return available collections.
        """

    # ======================================================
    # Rename Collection
    # ======================================================

    @abstractmethod
    def rename_collection(
        self,
        new_name: str,
    ) -> None:
        """
        Rename collection.
        """

    # ======================================================
    # Clear Collection
    # ======================================================

    @abstractmethod
    def clear(
        self,
    ) -> None:
        """
        Remove all stored vectors while keeping the collection.
        """

    # ======================================================
    # Collection Size
    # ======================================================

    @property
    def size(self) -> int:
        """
        Alias for count().
        """

        return self.count()

    # ======================================================
    # Is Empty
    # ======================================================

    @property
    def is_empty(self) -> bool:
        """
        Check whether the collection is empty.
        """

        return self.count() == 0
