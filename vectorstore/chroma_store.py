from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
import random
import time
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
            
    # ======================================================
    # Similarity Search
    # ======================================================

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search using an embedding vector.
        """

        try:

            results = self.collection.query(

                query_embeddings=[query_embedding],

                n_results=k,

                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],

            )

            output = []

            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc_id, doc, meta, distance in zip(
                ids,
                docs,
                metas,
                distances,
            ):

                score = 1.0 - float(distance)

                if (
                    score_threshold is not None
                    and score < score_threshold
                ):
                    continue

                output.append(

                    SearchResult(

                        id=doc_id,

                        score=score,

                        document=doc,

                        metadata=meta or {},

                    )

                )

            return output

        except Exception as exc:

            logger.exception(exc)

            raise VectorStoreError(

                f"Similarity search failed: {exc}"

            )

    # ======================================================
    # Similarity Search From Text
    # ======================================================

    def similarity_search_text(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Text-based similarity search.

        Requires Chroma embedding function.
        """

        results = self.collection.query(

            query_texts=[query],

            n_results=k,

            include=[
                "documents",
                "metadatas",
                "distances",
            ],

        )

        output = []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc_id, doc, meta, distance in zip(

            ids,
            docs,
            metas,
            distances,

        ):

            score = 1.0 - float(distance)

            if (
                score_threshold is not None
                and score < score_threshold
            ):
                continue

            output.append(

                SearchResult(

                    id=doc_id,

                    score=score,

                    document=doc,

                    metadata=meta or {},

                )

            )

        return output

    # ======================================================
    # Metadata Search
    # ======================================================

    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
    ) -> List[VectorDocument]:
        """
        Filter by metadata.
        """

        results = self.collection.get(

            where=filters,

            limit=limit,

            include=[
                "documents",
                "embeddings",
                "metadatas",
            ],

        )

        documents = []

        for i in range(

            len(results["ids"])

        ):

            documents.append(

                VectorDocument(

                    id=results["ids"][i],

                    text=results["documents"][i],

                    embedding=results["embeddings"][i],

                    metadata=results["metadatas"][i] or {},

                )

            )

        return documents

    # ======================================================
    # Batch Search
    # ======================================================

    def batch_search(
        self,
        query_embeddings: List[List[float]],
        k: int = 5,
    ) -> List[List[SearchResult]]:
        """
        Search multiple embedding queries.
        """

        all_results = []

        for embedding in query_embeddings:

            all_results.append(

                self.similarity_search(

                    embedding,

                    k,

                )

            )

        return all_results

    # ======================================================
    # Hybrid Search
    # ======================================================

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 5,
        alpha: float = 0.5,
    ) -> List[SearchResult]:
        """
        Hybrid search placeholder.

        Full implementation belongs in search.py.
        """

        return self.similarity_search(

            query_embedding,

            k,

        )

    # ======================================================
    # Range Search
    # ======================================================

    def range_search(
        self,
        query_embedding: List[float],
        radius: float,
    ) -> List[SearchResult]:
        """
        Return all vectors above threshold.
        """

        results = self.similarity_search(

            query_embedding,

            k=self.count(),

        )

        return [

            r

            for r in results

            if r.score >= radius

        ]

    # ======================================================
    # MMR Search
    # ======================================================

    def mmr_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> List[SearchResult]:
        """
        Placeholder.

        Full MMR is implemented in search.py.
        """

        return self.similarity_search(

            query_embedding,

            fetch_k,

        )[:k]

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:

        return {

            "backend": "ChromaDB",

            "collection": self.collection_name,

            "documents": self.count(),

            "dimension": self._dimension,

            "persist_directory": str(

                self.persist_directory

            ),

        }

    # ======================================================
    # Health
    # ======================================================

    def health(
        self,
    ) -> Dict[str, Any]:

        try:

            self.collection.count()

            return {

                "status": "healthy",

                "backend": "ChromaDB",

                "collection": self.collection_name,

                "documents": self.count(),

            }

        except Exception as exc:

            return {

                "status": "failed",

                "error": str(exc),

            }

    # ======================================================
    # Diagnostics
    # ======================================================

    def diagnostics(
        self,
    ) -> Dict[str, Any]:

        return {

            "health": self.health(),

            "statistics": self.statistics(),

            "collections": self.list_collections(),

        }

    # ======================================================
    # Benchmark
    # ======================================================

    def benchmark(
        self,
        num_queries: int = 100,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Benchmark retrieval latency.
        """

        if self.count() == 0:

            return {

                "status": "empty_collection"

            }

        dummy = [

            random.random()

            for _ in range(

                self._dimension

            )

        ]

        start = time.perf_counter()

        for _ in range(num_queries):

            self.similarity_search(

                dummy,

                top_k,

            )

        elapsed = time.perf_counter() - start

        return {

            "queries": num_queries,

            "seconds": round(

                elapsed,

                4,

            ),

            "queries_per_second": round(

                num_queries / elapsed,

                2,

            ),

        }

    # ======================================================
    # Save
    # ======================================================

    def save(
        self,
        path: str,
    ) -> None:
        """
        Persist collection.

        Chroma PersistentClient automatically
        persists changes, so this validates the
        persistence directory.
        """

        path = Path(path)

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Collection already persisted at %s",
            self.persist_directory,
        )

    # ======================================================
    # Load
    # ======================================================

    def load(
        self,
        path: str,
    ) -> None:
        """
        Load another persistence directory.
        """

        self.persist_directory = Path(path)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        logger.info(
            "Loaded collection '%s'",
            self.collection_name,
        )

    # ======================================================
    # Flush
    # ======================================================

    def flush(
        self,
    ) -> None:
        """
        Flush pending writes.

        PersistentClient automatically flushes,
        therefore nothing is required.
        """

        logger.debug(
            "Flush requested (automatic in ChromaDB)."
        )

    # ======================================================
    # Validate
    # ======================================================

    def validate(
        self,
    ) -> bool:
        """
        Validate collection integrity.
        """

        try:

            self.collection.count()

            return True

        except Exception as exc:

            logger.exception(exc)

            return False

    # ======================================================
    # Optimize
    # ======================================================

    def optimize(
        self,
    ) -> None:
        """
        Placeholder for future optimization.
        """

        logger.info(
            "Optimization handled internally by ChromaDB."
        )

    # ======================================================
    # Backup
    # ======================================================

    def backup(
        self,
        destination: str,
    ) -> None:
        """
        Backup persistence directory.
        """

        import shutil

        destination = Path(destination)

        if destination.exists():

            shutil.rmtree(destination)

        shutil.copytree(

            self.persist_directory,

            destination,

        )

        logger.info(
            "Backup created at %s",
            destination,
        )

    # ======================================================
    # Restore
    # ======================================================

    def restore(
        self,
        source: str,
    ) -> None:
        """
        Restore persistence directory.
        """

        import shutil

        source = Path(source)

        if not source.exists():

            raise VectorStoreError(
                f"Backup not found: {source}"
            )

        if self.persist_directory.exists():

            shutil.rmtree(
                self.persist_directory
            )

        shutil.copytree(

            source,

            self.persist_directory,

        )

        self.load(
            str(self.persist_directory)
        )

    # ======================================================
    # Reset
    # ======================================================

    def reset(
        self,
    ) -> None:
        """
        Remove all vectors while preserving
        the collection.
        """

        self.clear()

        logger.info(
            "Collection reset."
        )

    # ======================================================
    # Index Information
    # ======================================================

    def index_information(
        self,
    ) -> Dict[str, Any]:

        return {

            "backend": "ChromaDB",

            "collection": self.collection_name,

            "dimension": self._dimension,

            "documents": self.count(),

            "persist_directory": str(
                self.persist_directory
            ),

            "distance_metric": "cosine",

        }

    # ======================================================
    # Backend Properties
    # ======================================================

    @property
    def backend_name(
        self,
    ) -> str:

        return "ChromaDB"

    @property
    def embedding_dimension(
        self,
    ) -> int:

        return self._dimension

    @property
    def supports_metadata_filtering(
        self,
    ) -> bool:

        return True

    @property
    def supports_hybrid_search(
        self,
    ) -> bool:

        return False

    @property
    def supports_persistence(
        self,
    ) -> bool:

        return True

    @property
    def supports_deletion(
        self,
    ) -> bool:

        return True

    # ======================================================
    # Close
    # ======================================================

    def close(
        self,
    ) -> None:
        """
        Release resources.
        """

        logger.info(
            "Closing ChromaStore..."
        )

        self.collection = None

        self.client = None

    # ======================================================
    # Context Manager
    # ======================================================

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):

        self.close()

    # ======================================================
    # Representation
    # ======================================================

    def __repr__(
        self,
    ) -> str:

        return (

            "ChromaStore("

            f"collection='{self.collection_name}', "

            f"documents={self.count()}, "

            f"dimension={self._dimension}"

            ")"

        )
