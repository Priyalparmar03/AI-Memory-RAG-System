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

    # ======================================================
    # Health Check
    # ======================================================

    @abstractmethod
    def health(
        self,
    ) -> Dict[str, Any]:
        """
        Return health status of the vector store.

        Example:
        {
            "status": "healthy",
            "collection": "...",
            "documents": 1250,
            "backend": "FAISS"
        }
        """

    # ======================================================
    # Diagnostics
    # ======================================================

    @abstractmethod
    def diagnostics(
        self,
    ) -> Dict[str, Any]:
        """
        Return diagnostic information about the store.
        """

    # ======================================================
    # Statistics
    # ======================================================

    @abstractmethod
    def statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Return usage statistics.
        """

    # ======================================================
    # Benchmark
    # ======================================================

    @abstractmethod
    def benchmark(
        self,
        num_queries: int = 100,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Benchmark retrieval performance.
        """

    # ======================================================
    # Validation
    # ======================================================

    @abstractmethod
    def validate(
        self,
    ) -> bool:
        """
        Validate internal index consistency.
        """

    # ======================================================
    # Optimize Index
    # ======================================================

    @abstractmethod
    def optimize(
        self,
    ) -> None:
        """
        Optimize index for faster retrieval.
        """

    # ======================================================
    # Backup
    # ======================================================

    @abstractmethod
    def backup(
        self,
        destination: str,
    ) -> None:
        """
        Backup vector store.
        """

    # ======================================================
    # Restore
    # ======================================================

    @abstractmethod
    def restore(
        self,
        source: str,
    ) -> None:
        """
        Restore vector store.
        """

    # ======================================================
    # Export
    # ======================================================

    @abstractmethod
    def export(
        self,
    ) -> List[VectorDocument]:
        """
        Export all stored documents.
        """

    # ======================================================
    # Import
    # ======================================================

    @abstractmethod
    def import_documents(
        self,
        documents: List[VectorDocument],
    ) -> None:
        """
        Import previously exported documents.
        """

    # ======================================================
    # Index Information
    # ======================================================

    @abstractmethod
    def index_information(
        self,
    ) -> Dict[str, Any]:
        """
        Return backend-specific index information.
        """

    # ======================================================
    # Supports Metadata Filters
    # ======================================================

    @property
    @abstractmethod
    def supports_metadata_filtering(
        self,
    ) -> bool:
        """
        Whether backend supports metadata filtering.
        """

    # ======================================================
    # Supports Hybrid Search
    # ======================================================

    @property
    @abstractmethod
    def supports_hybrid_search(
        self,
    ) -> bool:
        """
        Whether backend supports hybrid search.
        """

    # ======================================================
    # Supports Persistence
    # ======================================================

    @property
    @abstractmethod
    def supports_persistence(
        self,
    ) -> bool:
        """
        Whether backend supports persistence.
        """

    # ======================================================
    # Supports Deletion
    # ======================================================

    @property
    @abstractmethod
    def supports_deletion(
        self,
    ) -> bool:
        """
        Whether backend supports deleting vectors.
        """

    # ======================================================
    # Backend Name
    # ======================================================

    @property
    @abstractmethod
    def backend_name(
        self,
    ) -> str:
        """
        Backend implementation name.

        Example:
            FAISS
            ChromaDB
            Qdrant
        """

    # ======================================================
    # Embedding Dimension
    # ======================================================

    @property
    @abstractmethod
    def embedding_dimension(
        self,
    ) -> int:
        """
        Embedding vector dimension.
        """

    # ======================================================
    # Save
    # ======================================================

    @abstractmethod
    def save(
        self,
        path: str,
    ) -> None:
        """
        Persist the vector store to disk.

        Parameters
        ----------
        path : str
            Destination directory or file.
        """

    # ======================================================
    # Load
    # ======================================================

    @abstractmethod
    def load(
        self,
        path: str,
    ) -> None:
        """
        Load an existing vector store.
        """

    # ======================================================
    # Flush
    # ======================================================

    @abstractmethod
    def flush(
        self,
    ) -> None:
        """
        Flush pending writes.

        Required for persistent backends.
        """

    # ======================================================
    # Reset
    # ======================================================

    @abstractmethod
    def reset(
        self,
    ) -> None:
        """
        Reset the vector store to an empty state.
        """

    # ======================================================
    # Close
    # ======================================================

    @abstractmethod
    def close(
        self,
    ) -> None:
        """
        Release resources.

        Close database connections,
        free memory,
        flush buffers.
        """

    # ======================================================
    # Sync
    # ======================================================

    def sync(self) -> None:
        """
        Synchronize pending changes.

        Default implementation delegates to flush().
        """

        self.flush()

    # ======================================================
    # Ready
    # ======================================================

    @property
    def ready(self) -> bool:
        """
        Returns True if the store is ready.
        """

        try:

            return self.validate()

        except Exception:

            return False

    # ======================================================
    # Collection Name
    # ======================================================

    @property
    def name(self) -> str:
        """
        Alias for collection name.
        """

        return self.collection_name

    # ======================================================
    # Empty
    # ======================================================

    @property
    def empty(self) -> bool:
        """
        Returns True when no vectors exist.
        """

        return self.count() == 0

    # ======================================================
    # Default Context Manager
    # ======================================================

    def __enter__(self):
        """
        Context manager entry.
        """

        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        """
        Context manager exit.
        """

        self.close()

    # ======================================================
    # String Representation
    # ======================================================

    def __repr__(self):

        return (

            f"{self.__class__.__name__}("

            f"collection='{self.collection_name}', "

            f"backend='{self.backend_name}', "

            f"documents={self.count()})"

        )

    # ======================================================
    # Length
    # ======================================================

    def __len__(self):

        return self.count()

    # ======================================================
    # Contains
    # ======================================================

    def __contains__(
        self,
        document_id: str,
    ) -> bool:

        return self.get_document(document_id) is not None

    # ======================================================
    # Iterator
    # ======================================================

    def __iter__(self):
        """
        Iterate over all stored documents.
        """

        return iter(self.get_documents())

    # ======================================================
    # Equality
    # ======================================================

    def __eq__(
        self,
        other,
    ):

        if not isinstance(
            other,
            BaseVectorStore,
        ):
            return False

        return (

            self.collection_name
            == other.collection_name

            and

            self.backend_name
            == other.backend_name

        )

    # ======================================================
    # Hash
    # ======================================================

    def __hash__(self):

        return hash(

            (

                self.collection_name,

                self.backend_name,

            )

        )
