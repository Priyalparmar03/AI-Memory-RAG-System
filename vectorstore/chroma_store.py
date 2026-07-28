from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

from .base_store import (
    BaseVectorStore,
    SearchResult,
    VectorDocument,
    VectorStoreError,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Chroma Vector Store
# ==========================================================

class ChromaStore(BaseVectorStore):
    """
    Production-ready ChromaDB implementation.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./data/chroma",
        embedding_dimension: int = 384,
        client_settings: Optional[Settings] = None,
    ):

        super().__init__(collection_name)

        self.persist_directory = Path(persist_directory)

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._dimension = embedding_dimension

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=client_settings,
        )

        self.collection = self._load_collection()

        logger.info(
            "ChromaStore initialized "
            "(collection=%s)",
            collection_name,
        )

    # ======================================================
    # Internal Helpers
    # ======================================================

    def _load_collection(
        self,
    ) -> Collection:
        """
        Create or load collection.
        """

        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine"
            },
        )

    # ======================================================
    # Collection Management
    # ======================================================

    def create_collection(
        self,
    ) -> None:

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def delete_collection(
        self,
    ) -> None:

        self.client.delete_collection(
            self.collection_name
        )

    def collection_exists(
        self,
    ) -> bool:

        try:

            self.client.get_collection(
                self.collection_name
            )

            return True

        except Exception:

            return False

    def list_collections(
        self,
    ) -> List[str]:

        collections = self.client.list_collections()

        return [

            c.name

            for c in collections

        ]

    def rename_collection(
        self,
        new_name: str,
    ) -> None:

        docs = self.export()

        self.delete_collection()

        self.collection_name = new_name

        self.create_collection()

        self.import_documents(docs)

    # ======================================================
    # Collection Statistics
    # ======================================================

    def collection_statistics(
        self,
    ) -> Dict[str, Any]:

        return {

            "collection": self.collection_name,

            "documents": self.count(),

            "dimension": self._dimension,

            "backend": "ChromaDB",

        }

    # ======================================================
    # Count
    # ======================================================

    def count(
        self,
    ) -> int:

        return self.collection.count()
