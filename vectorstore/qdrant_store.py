from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    OptimizersConfigDiff,
)
import random
import time
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)
from .base_store import (
    BaseVectorStore,
    SearchResult,
    VectorDocument,
    VectorStoreError,
)
import json
import shutil
from pathlib import Path
logger = logging.getLogger(__name__)

# ==========================================================
# Configuration
# ==========================================================

@dataclass
class QdrantConfig:
    """
    Configuration for Qdrant backend.
    """

    host: str = "localhost"

    port: int = 6333

    grpc_port: int = 6334

    https: bool = False

    api_key: Optional[str] = None

    prefer_grpc: bool = False

    timeout: int = 30

    collection_name: str = "documents"

    vector_size: int = 384

    distance: str = "Cosine"

    on_disk_payload: bool = True

    shard_number: int = 1

    replication_factor: int = 1

# ==========================================================
# Qdrant Store
# ==========================================================

class QdrantStore(BaseVectorStore):
    """
    Production-ready Qdrant implementation.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        config: Optional[QdrantConfig] = None,
    ):

        super().__init__(collection_name)

        self.config = config or QdrantConfig()

        self.collection_name = collection_name

        self.client = self._create_client()

        self._ensure_collection()

        logger.info(
            "QdrantStore initialized (%s)",
            self.collection_name,
        )

    # ======================================================
    # Client
    # ======================================================

    def _create_client(
        self,
    ) -> QdrantClient:
        """
        Create Qdrant client.
        """

        return QdrantClient(

            host=self.config.host,

            port=self.config.port,

            grpc_port=self.config.grpc_port,

            prefer_grpc=self.config.prefer_grpc,

            https=self.config.https,

            api_key=self.config.api_key,

            timeout=self.config.timeout,

        )

    # ======================================================
    # Distance Metric
    # ======================================================

    def _distance(
        self,
    ) -> Distance:
        """
        Convert string metric to Qdrant Distance enum.
        """

        metric = self.config.distance.lower()

        mapping = {

            "cosine": Distance.COSINE,

            "dot": Distance.DOT,

            "euclid": Distance.EUCLID,

            "manhattan": Distance.MANHATTAN,

        }

        return mapping.get(

            metric,

            Distance.COSINE,

        )

    # ======================================================
    # Collection
    # ======================================================

    def _ensure_collection(
        self,
    ) -> None:
        """
        Create collection if missing.
        """

        if self.collection_exists():

            return

        self.client.create_collection(

            collection_name=self.collection_name,

            vectors_config=VectorParams(

                size=self.config.vector_size,

                distance=self._distance(),

            ),

            optimizers_config=OptimizersConfigDiff(),

            shard_number=self.config.shard_number,

            replication_factor=self.config.replication_factor,

            on_disk_payload=self.config.on_disk_payload,

        )

        logger.info(

            "Created collection '%s'",

            self.collection_name,

        )


    # ======================================================
    # Collection Management
    # ======================================================

    def create_collection(
        self,
    ) -> None:

        self._ensure_collection()

    def delete_collection(
        self,
    ) -> None:

        self.client.delete_collection(

            self.collection_name,

        )

    def collection_exists(
        self,
    ) -> bool:

        collections = self.client.get_collections()

        return any(

            c.name == self.collection_name

            for c in collections.collections

        )

    def list_collections(
        self,
    ) -> List[str]:

        collections = self.client.get_collections()

        return [

            c.name

            for c in collections.collections

        ]

    def rename_collection(
        self,
        new_name: str,
    ) -> None:
        """
        Qdrant does not support rename.

        Export -> Create -> Import should be
        used in Part 4.
        """

        raise NotImplementedError(

            "Collection rename is not "

            "supported by Qdrant."

        )

    # ======================================================
    # Count
    # ======================================================

    def count(
        self,
    ) -> int:

        info = self.client.get_collection(

            self.collection_name

        )

        return info.points_count or 0

    # ======================================================
    # Statistics
    # ======================================================

    def collection_statistics(
        self,
    ) -> Dict[str, Any]:

        return {

            "backend": "Qdrant",

            "collection": self.collection_name,

            "documents": self.count(),

            "vector_size": self.config.vector_size,

            "distance": self.config.distance,

            "host": self.config.host,

            "port": self.config.port,

            "grpc": self.config.prefer_grpc,

        }

    # ======================================================
    # Add Document
    # ======================================================

    def add_document(
        self,
        document: VectorDocument,
    ) -> None:
        """
        Add a single document to Qdrant.
        """

        if document.embedding is None:

            raise VectorStoreError(
                "Document embedding is required."
            )

        point = PointStruct(

            id=document.id,

            vector=document.embedding,

            payload={

                "text": document.text,

                **document.metadata,

            },

        )

        self.client.upsert(

            collection_name=self.collection_name,

            points=[point],

            wait=True,

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

        points = []

        for doc in documents:

            if doc.embedding is None:

                raise VectorStoreError(
                    f"Embedding missing for {doc.id}"
                )

            points.append(

                PointStruct(

                    id=doc.id,

                    vector=doc.embedding,

                    payload={

                        "text": doc.text,

                        **doc.metadata,

                    },

                )

            )

        self.client.upsert(

            collection_name=self.collection_name,

            points=points,

            wait=True,

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

        current = self.get_document(document_id)

        if current is None:

            raise VectorStoreError(

                f"Document '{document_id}' not found."

            )

        updated = VectorDocument(

            id=document_id,

            text=text if text is not None else current.text,

            embedding=(
                embedding
                if embedding is not None
                else current.embedding
            ),

            metadata=(
                metadata
                if metadata is not None
                else current.metadata
            ),

        )

        self.add_document(updated)

    # ======================================================
    # Delete One
    # ======================================================

    def delete_document(
        self,
        document_id: str,
    ) -> None:

        self.client.delete(

            collection_name=self.collection_name,

            points_selector=[document_id],

            wait=True,

        )

    # ======================================================
    # Delete Many
    # ======================================================

    def delete_documents(
        self,
        ids: List[str],
    ) -> None:

        if not ids:
            return

        self.client.delete(

            collection_name=self.collection_name,

            points_selector=ids,

            wait=True,

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

        results = self.client.retrieve(

            collection_name=self.collection_name,

            ids=[document_id],

            with_payload=True,

            with_vectors=True,

        )

        if not results:

            return None

        point = results[0]

        payload = point.payload or {}

        return VectorDocument(

            id=str(point.id),

            text=payload.pop("text", ""),

            embedding=point.vector,

            metadata=payload,

        )

    # ======================================================
    # Get Documents
    # ======================================================

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
    ) -> List[VectorDocument]:
        """
        Retrieve multiple documents.
        """

        if ids is None:

            scroll, _ = self.client.scroll(

                collection_name=self.collection_name,

                with_vectors=True,

                with_payload=True,

                limit=self.count(),

            )

        else:

            scroll = self.client.retrieve(

                collection_name=self.collection_name,

                ids=ids,

                with_vectors=True,

                with_payload=True,

            )

        documents = []

        for point in scroll:

            payload = point.payload or {}

            documents.append(

                VectorDocument(

                    id=str(point.id),

                    text=payload.pop("text", ""),

                    embedding=point.vector,

                    metadata=payload,

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

        return self.get_documents(ids)

    # ======================================================
    # Export
    # ======================================================

    def export(
        self,
    ) -> List[VectorDocument]:
        """
        Export every document.
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
        Import documents.
        """

        self.add_documents(documents)

    # ======================================================
    # Clear Collection
    # ======================================================

    def clear(
        self,
    ) -> None:
        """
        Remove every point from collection.
        """

        self.client.delete(

            collection_name=self.collection_name,

            points_selector=Filter(),

            wait=True,

        )

    # ======================================================
    # Similarity Search
    # ======================================================

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Dense vector similarity search.
        """

        query_filter = None

        if filters:

            query_filter = Filter(

                must=[

                    FieldCondition(

                        key=key,

                        match=MatchValue(value=value),

                    )

                    for key, value in filters.items()

                ]

            )

        results = self.client.search(

            collection_name=self.collection_name,

            query_vector=query_embedding,

            query_filter=query_filter,

            limit=k,

            with_payload=True,

            with_vectors=True,

        )

        output = []

        for point in results:

            if (
                score_threshold is not None
                and point.score < score_threshold
            ):
                continue

            payload = dict(point.payload or {})

            output.append(

                SearchResult(

                    id=str(point.id),

                    score=float(point.score),

                    document=payload.pop("text", ""),

                    metadata=payload,

                    embedding=point.vector,

                )

            )

        return output

    # ======================================================
    # Similarity Search (Text)
    # ======================================================

    def similarity_search_text(
        self,
        query: str,
        embedding_function,
        k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:

        embedding = embedding_function(query)

        return self.similarity_search(

            embedding,

            k,

            score_threshold,

            filters,

        )

    # ======================================================
    # Metadata Search
    # ======================================================

    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
    ) -> List[VectorDocument]:

        query_filter = Filter(

            must=[

                FieldCondition(

                    key=key,

                    match=MatchValue(value=value),

                )

                for key, value in filters.items()

            ]

        )

        points, _ = self.client.scroll(

            collection_name=self.collection_name,

            scroll_filter=query_filter,

            with_payload=True,

            with_vectors=True,

            limit=limit,

        )

        documents = []

        for point in points:

            payload = dict(point.payload or {})

            documents.append(

                VectorDocument(

                    id=str(point.id),

                    text=payload.pop("text", ""),

                    embedding=point.vector,

                    metadata=payload,

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

        requests = [

            SearchRequest(

                vector=vector,

                limit=k,

                with_payload=True,

                with_vector=True,

            )

            for vector in query_embeddings

        ]

        responses = self.client.search_batch(

            collection_name=self.collection_name,

            requests=requests,

        )

        output = []

        for response in responses:

            batch = []

            for point in response:

                payload = dict(point.payload or {})

                batch.append(

                    SearchResult(

                        id=str(point.id),

                        score=float(point.score),

                        document=payload.pop("text", ""),

                        metadata=payload,

                        embedding=point.vector,

                    )

                )

            output.append(batch)

        return output

    # ======================================================
    # Range Search
    # ======================================================

    def range_search(
        self,
        query_embedding: List[float],
        radius: float,
    ) -> List[SearchResult]:

        results = self.similarity_search(

            query_embedding,

            self.count(),

        )

        return [

            result

            for result in results

            if result.score >= radius

        ]

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
        Placeholder.

        Full hybrid search will be
        implemented in search.py.
        """

        return self.similarity_search(

            query_embedding,

            k,

        )

    # ======================================================
    # Maximum Marginal Relevance
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

        MMR belongs in search.py.
        """

        return self.similarity_search(

            query_embedding,

            fetch_k,

        )[:k]

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(self) -> Dict[str, Any]:

        return {

            "backend": "Qdrant",

            "collection": self.collection_name,

            "documents": self.count(),

            "vector_size": self.config.vector_size,

            "distance": self.config.distance,

            "grpc": self.config.prefer_grpc,

            "host": self.config.host,

        }

    # ======================================================
    # Health
    # ======================================================

    def health(self) -> Dict[str, Any]:

        try:

            self.client.get_collection(

                self.collection_name

            )

            return {

                "status": "healthy",

                "backend": "Qdrant",

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

    def diagnostics(self) -> Dict[str, Any]:

        return {

            "statistics": self.statistics(),

            "health": self.health(),

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

        if self.count() == 0:

            return {

                "status": "empty_collection"

            }

        dummy = [

            random.random()

            for _ in range(

                self.config.vector_size

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

            "seconds": round(elapsed, 4),

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
        path: Optional[str] = None,
    ) -> None:
        """
        Save local configuration.

        Note:
        Qdrant persists vectors automatically.
        """

        save_dir = Path(path or "./qdrant_backup")

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        config = {

            "collection": self.collection_name,

            "host": self.config.host,

            "port": self.config.port,

            "grpc_port": self.config.grpc_port,

            "distance": self.config.distance,

            "vector_size": self.config.vector_size,

            "replication_factor": self.config.replication_factor,

            "shard_number": self.config.shard_number,

        }

        with open(

            save_dir / "config.json",

            "w",

        ) as file:

            json.dump(

                config,

                file,

                indent=4,

            )

        logger.info(

            "Configuration saved."

        )

    # ======================================================
    # Load
    # ======================================================

    def load(
        self,
        path: Optional[str] = None,
    ) -> None:
        """
        Load configuration.

        Collection data remains inside Qdrant.
        """

        load_dir = Path(path or "./qdrant_backup")

        with open(

            load_dir / "config.json",

        ) as file:

            config = json.load(file)

        self.collection_name = config["collection"]

        logger.info(

            "Configuration loaded."

        )

    # ======================================================
    # Flush
    # ======================================================

    def flush(
        self,
    ) -> None:
        """
        Qdrant automatically persists data.
        """

        logger.info(

            "Flush completed."

        )

    # ======================================================
    # Snapshot
    # ======================================================

    def create_snapshot(
        self,
    ) -> str:
        """
        Create Qdrant snapshot.
        """

        snapshot = self.client.create_snapshot(

            self.collection_name

        )

        logger.info(

            "Snapshot created."

        )

        return snapshot.name

    # ======================================================
    # List Snapshots
    # ======================================================

    def list_snapshots(
        self,
    ) -> List[str]:

        snapshots = self.client.list_snapshots(

            self.collection_name

        )

        return [

            snapshot.name

            for snapshot in snapshots

        ]

    # ======================================================
    # Restore Snapshot
    # ======================================================

    def restore_snapshot(
        self,
        snapshot_name: str,
    ) -> None:
        """
        Restore collection snapshot.
        """

        self.client.recover_snapshot(

            collection_name=self.collection_name,

            location=snapshot_name,

        )

        logger.info(

            "Snapshot restored."

        )

    # ======================================================
    # Backup
    # ======================================================

    def backup(
        self,
        destination: str,
    ) -> None:
        """
        Save configuration + snapshot.
        """

        destination = Path(destination)

        destination.mkdir(

            parents=True,

            exist_ok=True,

        )

        self.save(

            str(destination)

        )

        snapshot = self.create_snapshot()

        with open(

            destination / "snapshot.txt",

            "w",

        ) as file:

            file.write(snapshot)

        logger.info(

            "Backup created."

        )

    # ======================================================
    # Restore
    # ======================================================

    def restore(
        self,
        source: str,
    ) -> None:
        """
        Restore configuration and snapshot.
        """

        source = Path(source)

        self.load(

            str(source)

        )

        snapshot = (

            source /

            "snapshot.txt"

        )

        if snapshot.exists():

            with open(snapshot) as file:

                name = file.read().strip()

            self.restore_snapshot(

                name

            )

        logger.info(

            "Restore completed."

        )

    # ======================================================
    # Optimize
    # ======================================================

    def optimize(
        self,
    ) -> None:
        """
        Trigger optimizer.
        """

        self.client.update_collection(

            collection_name=self.collection_name,

            optimizer_config=OptimizersConfigDiff(

                indexing_threshold=10000,

            ),

        )

        logger.info(

            "Optimization triggered."

        )

    # ======================================================
    # Validate
    # ======================================================

    def validate(
        self,
    ) -> bool:

        try:

            info = self.client.get_collection(

                self.collection_name

            )

            return (

                info.config.params.vectors.size

                ==

                self.config.vector_size

            )

        except Exception:

            return False

    # ======================================================
    # Index Information
    # ======================================================

    def index_information(
        self,
    ) -> Dict[str, Any]:

        info = self.client.get_collection(

            self.collection_name

        )

        return {

            "backend": "Qdrant",

            "collection": self.collection_name,

            "points": info.points_count,

            "segments": info.segments_count,

            "distance": self.config.distance,

            "vector_size": self.config.vector_size,

            "replication": self.config.replication_factor,

            "shards": self.config.shard_number,

        }

    # ======================================================
    # Reset
    # ======================================================

    def reset(
        self,
    ) -> None:
        """
        Delete every point while
        preserving collection.
        """

        self.clear()

        logger.info(

            "Collection reset."

        )
