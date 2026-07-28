from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import random
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


    # ======================================================
    # Save Index
    # ======================================================

    def save(
        self,
        path: Optional[str] = None,
    ) -> None:
        """
        Save FAISS index, metadata and configuration.
        """

        save_dir = Path(path) if path else self.persist_directory
        save_dir.mkdir(parents=True, exist_ok=True)

        # GPU index must be moved to CPU before saving
        index = self.index
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(
            index,
            str(save_dir / "index.faiss"),
        )

        with open(save_dir / "documents.pkl", "wb") as f:
            pickle.dump(self.id_to_doc, f)

        with open(save_dir / "doc_to_faiss.pkl", "wb") as f:
            pickle.dump(self.doc_to_faiss, f)

        with open(save_dir / "faiss_to_doc.pkl", "wb") as f:
            pickle.dump(self.faiss_to_doc, f)

        config = {
            "dimension": self.dimension,
            "metric": self.metric,
            "index_type": self.index_type,
            "use_gpu": self.use_gpu,
            "collection": self.collection_name,
        }

        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        logger.info("FAISS store saved successfully.")

    # ======================================================
    # Load Index
    # ======================================================

    def load(
        self,
        path: Optional[str] = None,
    ) -> None:
        """
        Load FAISS index and metadata.
        """

        load_dir = Path(path) if path else self.persist_directory

        self.index = faiss.read_index(
            str(load_dir / "index.faiss")
        )

        with open(load_dir / "documents.pkl", "rb") as f:
            self.id_to_doc = pickle.load(f)

        with open(load_dir / "doc_to_faiss.pkl", "rb") as f:
            self.doc_to_faiss = pickle.load(f)

        with open(load_dir / "faiss_to_doc.pkl", "rb") as f:
            self.faiss_to_doc = pickle.load(f)

        with open(load_dir / "config.json") as f:
            config = json.load(f)

        self.dimension = config["dimension"]
        self.metric = config["metric"]
        self.index_type = config["index_type"]

        if self.use_gpu:
            self._to_gpu()

        logger.info("FAISS store loaded successfully.")

    # ======================================================
    # Flush
    # ======================================================

    def flush(
        self,
    ) -> None:
        """
        Persist current state.
        """

        self.save()

    # ======================================================
    # Backup
    # ======================================================

    def backup(
        self,
        destination: str,
    ) -> None:
        """
        Create backup.
        """

        self.save()

        destination = Path(destination)

        if destination.exists():
            import shutil
            shutil.rmtree(destination)

        import shutil

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
        Restore from backup.
        """

        import shutil

        source = Path(source)

        if not source.exists():
            raise VectorStoreError(
                f"Backup not found: {source}"
            )

        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)

        shutil.copytree(
            source,
            self.persist_directory,
        )

        self.load()

        logger.info("Restore completed.")

    # ======================================================
    # Optimize
    # ======================================================

    def optimize(
        self,
    ) -> None:
        """
        Optimize FAISS index.
        """

        if hasattr(self.index, "make_direct_map"):
            try:
                self.index.make_direct_map()
            except Exception:
                pass

        logger.info("Optimization complete.")

    # ======================================================
    # Rebuild
    # ======================================================

    def rebuild(
        self,
    ) -> None:
        """
        Rebuild the index.
        """

        self._rebuild_index()

        logger.info("Index rebuilt.")

    # ======================================================
    # Reset
    # ======================================================

    def reset(
        self,
    ) -> None:
        """
        Remove every stored vector.
        """

        self.clear()

        logger.info("Collection reset.")

    # ======================================================
    # Validate
    # ======================================================

    def validate(
        self,
    ) -> bool:
        """
        Validate store consistency.
        """

        if self.count() != len(self.id_to_doc):
            return False

        if len(self.doc_to_faiss) != len(self.id_to_doc):
            return False

        if len(self.faiss_to_doc) != len(self.id_to_doc):
            return False

        return True

    # ======================================================
    # Index Information
    # ======================================================

    def index_information(
        self,
    ) -> Dict[str, Any]:

        return {

            "backend": "FAISS",

            "index_type": self.index_type,

            "metric": self.metric,

            "dimension": self.dimension,

            "vectors": self.count(),

            "gpu": self.use_gpu,

            "trained": getattr(
                self.index,
                "is_trained",
                True,
            ),

        }
    # ======================================================
    # Backend Properties
    # ======================================================

    @property
    def backend_name(self) -> str:
        """Return backend name."""
        return "FAISS"

    @property
    def embedding_dimension(self) -> int:
        """Embedding dimension."""
        return self.dimension

    @property
    def supports_metadata_filtering(self) -> bool:
        return True

    @property
    def supports_hybrid_search(self) -> bool:
        return False

    @property
    def supports_persistence(self) -> bool:
        return True

    @property
    def supports_deletion(self) -> bool:
        return True

    # ======================================================
    # Memory Usage
    # ======================================================

    def memory_usage(self) -> Dict[str, Any]:
        """
        Approximate memory usage.
        """

        vectors = self.count() * self.dimension * 4

        return {

            "documents": len(self.id_to_doc),

            "vectors": self.count(),

            "embedding_dimension": self.dimension,

            "estimated_vector_memory_mb":
                round(vectors / (1024 * 1024), 2),

        }

    # ======================================================
    # Runtime Statistics
    # ======================================================

    def runtime_statistics(self) -> Dict[str, Any]:

        return {

            "backend": "FAISS",

            "index_type": self.index_type,

            "metric": self.metric,

            "gpu": self.use_gpu,

            "trained": getattr(
                self.index,
                "is_trained",
                True,
            ),

            "vectors": self.count(),

            "documents": len(self.id_to_doc),

        }

    # ======================================================
    # Advanced Diagnostics
    # ======================================================

    def advanced_diagnostics(self) -> Dict[str, Any]:
        """
        Complete diagnostics.
        """

        return {

            "health": self.health(),

            "statistics": self.statistics(),

            "memory": self.memory_usage(),

            "runtime": self.runtime_statistics(),

            "validation": self.validate(),

        }

    # ======================================================
    # Close
    # ======================================================

    def close(self) -> None:
        """
        Release resources.
        """

        if self.use_gpu:

            self._to_cpu()

        self.index = None

        self.id_to_doc.clear()

        self.doc_to_faiss.clear()

        self.faiss_to_doc.clear()

        logger.info(
            "FAISS store closed."
        )

    # ======================================================
    # Context Manager
    # ======================================================

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()

    # ======================================================
    # Python Protocols
    # ======================================================

    def __len__(self):

        return self.count()

    def __contains__(
        self,
        document_id: str,
    ):

        return document_id in self.id_to_doc

    def __iter__(self):

        return iter(
            self.id_to_doc.values()
        )

    def __repr__(self):

        return (

            "FaissStore("

            f"collection='{self.collection_name}', "

            f"backend='FAISS', "

            f"documents={self.count()}, "

            f"dimension={self.dimension}, "

            f"metric='{self.metric}', "

            f"index='{self.index_type}', "

            f"gpu={self.use_gpu}"

            ")"

        )
