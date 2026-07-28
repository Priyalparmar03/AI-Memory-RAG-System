from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

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
class FaissConfig:
    """
    Configuration for the FAISS backend.
    """

    dimension: int = 384

    metric: str = "cosine"

    index_type: str = "flat"

    persist_directory: str = "./data/faiss"

    use_gpu: bool = False

    nlist: int = 100

    nprobe: int = 10


# ==========================================================
# FAISS Store
# ==========================================================

class FaissStore(BaseVectorStore):
    """
    Production FAISS implementation.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        config: Optional[FaissConfig] = None,
    ):

        super().__init__(collection_name)

        self.config = config or FaissConfig()

        self.persist_directory = Path(
            self.config.persist_directory
        )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.dimension = self.config.dimension

        self.metric = self.config.metric.lower()

        self.index_type = self.config.index_type.lower()

        self.use_gpu = (
            self.config.use_gpu
            and faiss.get_num_gpus() > 0
        )

        self.index = self._create_index()

        if self.use_gpu:

            self._to_gpu()

        # ==================================================
        # Internal Storage
        # ==================================================

        self.id_to_doc: Dict[
            str,
            VectorDocument,
        ] = {}

        self.doc_to_faiss: Dict[
            str,
            int,
        ] = {}

        self.faiss_to_doc: Dict[
            int,
            str,
        ] = {}

        logger.info(

            "Initialized FAISS (%s)",

            self.index_type,

        )
    # ======================================================
    # Internal Helpers
    # ======================================================

    def _ensure_trained(
        self,
        vectors: np.ndarray,
    ) -> None:
        """
        Train IVF/PQ indexes if required.
        """

        if hasattr(self.index, "is_trained"):

            if not self.index.is_trained:

                self.index.train(vectors)

                logger.info("FAISS index trained.")

    # ======================================================
    # Rebuild Index
    # ======================================================

    def _rebuild_index(self) -> None:
        """
        Rebuild the entire index from stored documents.
        """

        documents = list(self.id_to_doc.values())

        self.index = self._create_index()

        if self.use_gpu:
            self._to_gpu()

        self.doc_to_faiss.clear()
        self.faiss_to_doc.clear()

        if not documents:
            return

        vectors = np.array(
            [doc.embedding for doc in documents],
            dtype=np.float32,
        )

        vectors = self._normalize(vectors)

        self._ensure_trained(vectors)

        self.index.add(vectors)

        for i, doc in enumerate(documents):

            self.doc_to_faiss[doc.id] = i
            self.faiss_to_doc[i] = doc.id

    # ======================================================
    # Add Document
    # ======================================================

    def add_document(
        self,
        document: VectorDocument,
    ) -> None:
        """
        Add one document.
        """

        if document.embedding is None:

            raise VectorStoreError(
                "Embedding is required."
            )

        vector = np.asarray(
            [document.embedding],
            dtype=np.float32,
        )

        vector = self._normalize(vector)

        self._ensure_trained(vector)

        self.index.add(vector)

        position = self.index.ntotal - 1

        self.id_to_doc[document.id] = document

        self.doc_to_faiss[document.id] = position

        self.faiss_to_doc[position] = document.id

    # ======================================================
    # Batch Insert
    # ======================================================

    def add_documents(
        self,
        documents: List[VectorDocument],
    ) -> None:
        """
        Insert multiple documents.
        """

        if not documents:
            return

        vectors = np.asarray(
            [d.embedding for d in documents],
            dtype=np.float32,
        )

        vectors = self._normalize(vectors)

        self._ensure_trained(vectors)

        start = self.index.ntotal

        self.index.add(vectors)

        for offset, doc in enumerate(documents):

            idx = start + offset

            self.id_to_doc[doc.id] = doc

            self.doc_to_faiss[doc.id] = idx

            self.faiss_to_doc[idx] = doc.id

    # ======================================================
    # Update
    # ======================================================

    def update_document(
        self,
        document_id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Update a document.

        FAISS requires rebuilding.
        """

        if document_id not in self.id_to_doc:

            raise VectorStoreError(
                f"{document_id} not found."
            )

        doc = self.id_to_doc[document_id]

        if text is not None:
            doc.text = text

        if metadata is not None:
            doc.metadata = metadata

        if embedding is not None:
            doc.embedding = embedding

        self._rebuild_index()

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

        if document_id not in self.id_to_doc:

            return

        del self.id_to_doc[document_id]

        self._rebuild_index()

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

        for doc_id in ids:

            self.id_to_doc.pop(
                doc_id,
                None,
            )

        self._rebuild_index()

    # ======================================================
    # Get One
    # ======================================================

    def get_document(
        self,
        document_id: str,
    ) -> Optional[VectorDocument]:
        """
        Retrieve one document.
        """

        return self.id_to_doc.get(document_id)

    # ======================================================
    # Get Many
    # ======================================================

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
    ) -> List[VectorDocument]:
        """
        Retrieve documents.
        """

        if ids is None:

            return list(
                self.id_to_doc.values()
            )

        return [

            self.id_to_doc[i]

            for i in ids

            if i in self.id_to_doc

        ]

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
        Export all stored documents.
        """

        return list(
            self.id_to_doc.values()
        )

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
    # Clear
    # ======================================================

    def clear(
        self,
    ) -> None:
        """
        Remove every vector.
        """

        self.id_to_doc.clear()

        self.doc_to_faiss.clear()

        self.faiss_to_doc.clear()

        self.index.reset()

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
    Perform similarity search using FAISS.
    """

    if self.count() == 0:
        return []

    query = np.asarray(
        [query_embedding],
        dtype=np.float32,
    )

    query = self._normalize(query)

    scores, indices = self.index.search(
        query,
        min(k, self.count()),
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        doc_id = self.faiss_to_doc.get(int(idx))

        if doc_id is None:
            continue

        document = self.id_to_doc[doc_id]

        similarity = (
            float(score)
            if self.metric in ("cosine", "ip")
            else 1.0 / (1.0 + float(score))
        )

        if (
            score_threshold is not None
            and similarity < score_threshold
        ):
            continue

        results.append(

            SearchResult(

                id=document.id,

                score=similarity,

                document=document.text,

                metadata=document.metadata,

                embedding=document.embedding,

            )

        )

    return results


# ======================================================
# Similarity Search (Text)
# ======================================================

def similarity_search_text(
    self,
    query: str,
    embedding_function,
    k: int = 5,
    score_threshold: Optional[float] = None,
) -> List[SearchResult]:
    """
    Search using raw text.

    embedding_function must return List[float].
    """

    embedding = embedding_function(query)

    return self.similarity_search(

        embedding,

        k,

        score_threshold,

    )


# ======================================================
# Metadata Search
# ======================================================

def search_by_metadata(
    self,
    filters: Dict[str, Any],
    limit: int = 100,
) -> List[VectorDocument]:
    """
    Filter documents by metadata.
    """

    output = []

    for doc in self.id_to_doc.values():

        matched = True

        for key, value in filters.items():

            if doc.metadata.get(key) != value:

                matched = False

                break

        if matched:

            output.append(doc)

        if len(output) >= limit:

            break

    return output


# ======================================================
# Batch Search
# ======================================================

def batch_search(
    self,
    query_embeddings: List[List[float]],
    k: int = 5,
) -> List[List[SearchResult]]:
    """
    Search multiple vectors.
    """

    outputs = []

    for embedding in query_embeddings:

        outputs.append(

            self.similarity_search(

                embedding,

                k,

            )

        )

    return outputs


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

        self.count(),

    )

    return [

        r

        for r in results

        if r.score >= radius

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

    Full implementation belongs to search.py
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

    Full MMR implemented in search.py.
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

        "backend": "FAISS",

        "collection": self.collection_name,

        "documents": self.count(),

        "dimension": self.dimension,

        "metric": self.metric,

        "index_type": self.index_type,

        "gpu": self.use_gpu,

    }


# ======================================================
# Health
# ======================================================

def health(
    self,
) -> Dict[str, Any]:

    try:

        self.index.ntotal

        return {

            "status": "healthy",

            "backend": "FAISS",

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

        "statistics": self.statistics(),

        "health": self.health(),

        "trained": getattr(

            self.index,

            "is_trained",

            True,

        ),

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
    Benchmark FAISS retrieval speed.
    """

    if self.count() == 0:

        return {

            "status": "empty"

        }

    dummy = np.random.random(

        self.dimension

    ).astype(np.float32)

    start = time.perf_counter()

    for _ in range(num_queries):

        self.similarity_search(

            dummy.tolist(),

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
