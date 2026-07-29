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

from .base_store import (
    BaseVectorStore,
    SearchResult,
    VectorDocument,
    VectorStoreError,
)

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

