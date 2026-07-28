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

    # ======================================================
    # Add Single Document
    # ======================================================

    def add_document(
        self,
        document: VectorDocument,
    ) -> None:
        """
        Add a single document.
        """

        try:

            self.collection.add(

                ids=[document.id],

                documents=[document.text],

                embeddings=[document.embedding]
                if document.embedding
                else None,

                metadatas=[document.metadata],

            )

        except Exception as exc:

            logger.exception(exc)

            raise VectorStoreError(

                f"Failed to add document: {exc}"

            )

    # ======================================================
    # Batch Insert
    # ======================================================

    def add_documents(
        self,
        documents: List[VectorDocument],
    ) -> None:
        """
        Batch insert documents.
        """

        if not documents:

            return

        try:

            self.collection.add(

                ids=[d.id for d in documents],

                documents=[d.text for d in documents],

                embeddings=[
                    d.embedding
                    for d in documents
                ]
                if documents[0].embedding is not None
                else None,

                metadatas=[
                    d.metadata
                    for d in documents
                ],

            )

        except Exception as exc:

            logger.exception(exc)

            raise VectorStoreError(

                f"Batch insert failed: {exc}"

            )

    # ======================================================
    # Update Document
    # ======================================================

    def update_document(
        self,
        document_id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Update an existing document.
        """

        try:

            self.collection.update(

                ids=[document_id],

                documents=[text] if text else None,

                embeddings=[embedding]
                if embedding is not None
                else None,

                metadatas=[metadata]
                if metadata is not None
                else None,

            )

        except Exception as exc:

            logger.exception(exc)

            raise VectorStoreError(

                f"Update failed: {exc}"

            )

    # ======================================================
    # Delete One
    # ======================================================

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete one document.
        """

        self.collection.delete(

            ids=[document_id]

        )

    # ======================================================
    # Delete Many
    # ======================================================

    def delete_documents(
        self,
        ids: List[str],
    ) -> None:
        """
        Delete multiple documents.
        """

        if ids:

            self.collection.delete(

                ids=ids

            )

    # ======================================================
    # Get Document
    # ======================================================

    def get_document(
        self,
        document_id: str,
    ) -> Optional[VectorDocument]:
        """
        Retrieve one document.
        """

        result = self.collection.get(

            ids=[document_id],

            include=[

                "documents",

                "metadatas",

                "embeddings",

            ],

        )

        if not result["ids"]:

            return None

        return VectorDocument(

            id=result["ids"][0],

            text=result["documents"][0],

            embedding=result["embeddings"][0],

            metadata=result["metadatas"][0]
            or {},

        )

    # ======================================================
    # Get Documents
    # ======================================================

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
    ) -> List[VectorDocument]:
        """
        Retrieve documents.
        """

        result = self.collection.get(

            ids=ids,

            include=[

                "documents",

                "metadatas",

                "embeddings",

            ],

        )

        documents = []

        for idx in range(

            len(result["ids"])

        ):

            documents.append(

                VectorDocument(

                    id=result["ids"][idx],

                    text=result["documents"][idx],

                    embedding=result["embeddings"][idx],

                    metadata=result["metadatas"][idx]
                    or {},

                )

            )

        return documents

    # ======================================================
    # Get By IDs
    # ======================================================

    def get_by_ids(
        self,
        ids: List[str],
    ) -> List[VectorDocument]:
        """
        Retrieve multiple documents by IDs.
        """

        return self.get_documents(ids)

    # ======================================================
    # Export
    # ======================================================

    def export(
        self,
    ) -> List[VectorDocument]:
        """
        Export all stored documents.
        """

        return self.get_documents()

    # ======================================================
    # Import
    # ======================================================

    def import_documents(
        self,
        documents: List[VectorDocument],
    ) -> None:
        """
        Import documents into the collection.
        """

        self.add_documents(documents)

    # ======================================================
    # Clear Collection
    # ======================================================

    def clear(
        self,
    ) -> None:
        """
        Remove all vectors from the collection.
        """

        ids = self.collection.get()["ids"]

        if ids:

            self.collection.delete(

                ids=ids

            )
            
