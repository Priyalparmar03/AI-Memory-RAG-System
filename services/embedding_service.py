from __future__ import annotations
import logging
import threading
from typing import List, Optional, Dict, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import time
import hashlib

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Production-grade embedding service using SentenceTransformers.

    This service is responsible ONLY for generating embeddings.
    Storage (FAISS, ChromaDB, Qdrant, etc.) is handled by RagService.
    """

    _instance = None
    _lock = threading.Lock()

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __new__(cls, *args, **kwargs):
        """
        Singleton implementation.
        Ensures only one model is loaded into memory.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):

        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        self.device = device or self._detect_device()

        logger.info(
            "Loading embedding model '%s' on %s",
            self.model_name,
            self.device,
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        self._initialized = True

        logger.info(
            "Embedding model loaded successfully."
        )

    # ======================================================
    # Device Detection
    # ======================================================

    @staticmethod
    def _detect_device() -> str:
        """
        Detect the best available device.
        """

        if torch.cuda.is_available():

            logger.info(
                "CUDA detected."
            )

            return "cuda"

        if hasattr(torch.backends, "mps"):

            if torch.backends.mps.is_available():

                logger.info(
                    "Apple MPS detected."
                )

                return "mps"

        logger.info(
            "Using CPU."
        )

        return "cpu"

    # ======================================================
    # Model Information
    # ======================================================

    def model_info(self) -> Dict:
        """
        Return model metadata.
        """

        return {

            "model_name": self.model_name,

            "device": self.device,

            "embedding_dimension": self.embedding_dimension,

            "normalize_embeddings":
                self.normalize_embeddings,

        }

    # ======================================================
    # Health Check
    # ======================================================

    def health(self) -> Dict:
        """
        Verify embedding model status.
        """

        return {

            "status": "healthy",

            "model_loaded": self.model is not None,

            "device": self.device,

            "dimension": self.embedding_dimension,

        }

    # ======================================================
    # Warmup
    # ======================================================

    def warmup(self) -> None:
        """
        Warm up the model with a dummy embedding.
        Reduces first-request latency.
        """

        logger.info(
            "Running embedding model warmup..."
        )

        self.embed_query(
            "Embedding service warmup."
        )

        logger.info(
            "Warmup complete."
        )

    # ======================================================
    # Internal Encoder
    # ======================================================

    def _encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Internal encoding method.
        """

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        return embeddings

      # ======================================================
    # Input Validation
    # ======================================================

    @staticmethod
    def _validate_text(text: str) -> str:
        """
        Validate a single text input.
        """

        if text is None:
            raise ValueError("Input text cannot be None.")

        text = str(text).strip()

        if not text:
            raise ValueError("Input text cannot be empty.")

        return text

    @classmethod
    def _validate_documents(
        cls,
        documents: List[str],
    ) -> List[str]:
        """
        Validate a list of documents.
        """

        if not documents:
            raise ValueError("Document list is empty.")

        return [cls._validate_text(doc) for doc in documents]

    # ======================================================
    # Embed Query
    # ======================================================

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Generate an embedding for a search query.
        """

        query = self._validate_text(query)

        logger.debug("Embedding query.")

        embedding = self._encode(query)

        return embedding.astype(np.float32)

    # ======================================================
    # Embed Document
    # ======================================================

    def embed_document(
        self,
        document: str,
    ) -> np.ndarray:
        """
        Generate an embedding for a single document.
        """

        document = self._validate_text(document)

        logger.debug("Embedding document.")

        embedding = self._encode(document)

        return embedding.astype(np.float32)

    # ======================================================
    # Embed Multiple Documents
    # ======================================================

    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple documents.
        """

        documents = self._validate_documents(documents)

        logger.info(
            "Embedding %d documents.",
            len(documents),
        )

        embeddings = self._encode(
            documents,
            batch_size=batch_size,
        )

        return embeddings.astype(np.float32)

    # ======================================================
    # Batch Generator
    # ======================================================

    @staticmethod
    def batch_iterator(
        items: List[str],
        batch_size: int,
    ):
        """
        Yield batches of documents.
        """

        for start in range(0, len(items), batch_size):
            yield items[start:start + batch_size]

    # ======================================================
    # Streaming Batch Embedding
    # ======================================================

    def embed_in_batches(
        self,
        documents: List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Embed documents incrementally.
        Useful for very large datasets.
        """

        documents = self._validate_documents(documents)

        batches = []

        for batch in self.batch_iterator(
            documents,
            batch_size,
        ):

            embeddings = self._encode(
                batch,
                batch_size=batch_size,
            )

            batches.append(embeddings)

        return np.vstack(batches).astype(np.float32)

    # ======================================================
    # Normalize Embeddings
    # ======================================================

    @staticmethod
    def normalize(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        L2 normalize embeddings.
        """

        norms = np.linalg.norm(
            embeddings,
            axis=-1,
            keepdims=True,
        )

        norms[norms == 0] = 1

        return embeddings / norms

    # ======================================================
    # Embedding Dimension
    # ======================================================

    def dimension(self) -> int:
        """
        Return embedding dimension.
        """

        return self.embedding_dimension

    # ======================================================
    # Model Name
    # ======================================================

    def model_name_info(self) -> str:
        """
        Return active model name.
        """

        return self.model_name

    # ======================================================
    # Device Information
    # ======================================================

    def current_device(self) -> str:
        """
        Return current execution device.
        """

        return self.device

    # ======================================================
    # Empty Cache
    # ======================================================

    @staticmethod
    def clear_gpu_cache() -> None:
        """
        Free CUDA memory if available.
        """

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            logger.info(
                "CUDA cache cleared."
            )

          # ======================================================
    # Cosine Similarity
    # ======================================================

    @staticmethod
    def cosine_similarity(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        """

        embedding1 = np.asarray(embedding1).flatten()
        embedding2 = np.asarray(embedding2).flatten()

        denominator = (
            np.linalg.norm(embedding1)
            * np.linalg.norm(embedding2)
        )

        if denominator == 0:
            return 0.0

        similarity = np.dot(
            embedding1,
            embedding2,
        ) / denominator

        return float(similarity)

    # ======================================================
    # Pairwise Similarity Matrix
    # ======================================================

    @classmethod
    def similarity_matrix(
        cls,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity matrix.
        """

        normalized = cls.normalize(embeddings)

        return np.dot(
            normalized,
            normalized.T,
        )

    # ======================================================
    # Embedding Hash
    # ======================================================

    @staticmethod
    def text_hash(
        text: str,
    ) -> str:
        """
        Stable SHA256 hash for caching.
        """

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    # ======================================================
    # Cached Query Embedding
    # ======================================================

    @lru_cache(maxsize=2048)
    def cached_query_embedding(
        self,
        query: str,
    ) -> tuple:
        """
        Cache frequently embedded queries.

        lru_cache requires immutable return values,
        so we store tuples.
        """

        embedding = self.embed_query(query)

        return tuple(embedding.tolist())

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(self) -> Dict:
        """
        Runtime statistics.
        """

        return {

            "model": self.model_name,

            "dimension": self.embedding_dimension,

            "device": self.device,

            "normalize_embeddings":
                self.normalize_embeddings,

            "cache_size":
                self.cached_query_embedding.cache_info().currsize,

            "cache_hits":
                self.cached_query_embedding.cache_info().hits,

            "cache_misses":
                self.cached_query_embedding.cache_info().misses,
        }

    # ======================================================
    # Benchmark
    # ======================================================

    def benchmark(
        self,
        text: str = "Hello world",
    ) -> Dict:
        """
        Measure embedding latency.
        """

        start = time.perf_counter()

        embedding = self.embed_query(text)

        latency = (
            time.perf_counter()
            - start
        ) * 1000

        return {

            "latency_ms":
                round(latency, 2),

            "dimension":
                len(embedding),

        }

    # ======================================================
    # Batch Benchmark
    # ======================================================

    def benchmark_batch(
        self,
        documents: List[str],
    ) -> Dict:
        """
        Benchmark batch embedding.
        """

        start = time.perf_counter()

        embeddings = self.embed_documents(documents)

        latency = (
            time.perf_counter()
            - start
        ) * 1000

        return {

            "documents":
                len(documents),

            "latency_ms":
                round(latency, 2),

            "throughput_docs_per_sec":
                round(
                    len(documents)
                    / (latency / 1000),
                    2,
                ),

            "dimension":
                embeddings.shape[-1],
        }

    # ======================================================
    # Health Diagnostics
    # ======================================================

    def diagnostics(self) -> Dict:
        """
        Extended health report.
        """

        report = self.health()

        report.update(

            {

                "torch_version":
                    torch.__version__,

                "cuda_available":
                    torch.cuda.is_available(),

                "gpu_count":
                    torch.cuda.device_count(),

                "model_type":
                    type(self.model).__name__,
            }

        )

        return report

    # ======================================================
    # Reset Cache
    # ======================================================

    def clear_cache(self):
        """
        Clear LRU cache.
        """

        self.cached_query_embedding.cache_clear()

        logger.info(
            "Embedding cache cleared."
        )

    # ======================================================
    # Destructor
    # ======================================================

    def close(self):
        """
        Release resources.
        """

        self.clear_cache()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        logger.info(
            "Embedding service closed."
        )
