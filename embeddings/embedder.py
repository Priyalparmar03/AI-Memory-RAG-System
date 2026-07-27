from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class EmbeddingService:
    """
    Main embedding service for the RAG system.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        cache_folder: Optional[Union[str, Path]] = None,
    ) -> None:

        self.model_name = model_name or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self.normalize = normalize_embeddings

        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.cache_folder = (
            Path(cache_folder)
            if cache_folder
            else None
        )

        self._model: Optional[SentenceTransformer] = None
        self._lock = threading.Lock()

        logger.info(
            "EmbeddingService initialized "
            "(model=%s device=%s)",
            self.model_name,
            self.device,
        )

    ####################################################################
    # Model
    ####################################################################

    @property
    def model(self) -> SentenceTransformer:

        if self._model is None:

            with self._lock:

                if self._model is None:

                    logger.info(
                        "Loading embedding model %s",
                        self.model_name,
                    )

                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        cache_folder=str(self.cache_folder)
                        if self.cache_folder
                        else None,
                    )

        return self._model

    ####################################################################
    # Information
    ####################################################################

    @property
    def embedding_dimension(self) -> int:

        return self.model.get_sentence_embedding_dimension()

    @property
    def current_model(self) -> str:

        return self.model_name

    @property
    def using_gpu(self) -> bool:

        return self.device == "cuda"

    ####################################################################
    # Validation
    ####################################################################

    @staticmethod
    def _validate_text(text: str) -> str:

        if text is None:
            raise EmbeddingError("Text cannot be None.")

        if not isinstance(text, str):
            raise EmbeddingError(
                "Input must be a string."
            )

        text = text.strip()

        if len(text) == 0:
            raise EmbeddingError(
                "Input text is empty."
            )

        return text

    @staticmethod
    def _validate_batch(
        texts: Sequence[str],
    ) -> List[str]:

        if texts is None:
            raise EmbeddingError(
                "Batch cannot be None."
            )

        if len(texts) == 0:
            raise EmbeddingError(
                "Batch is empty."
            )

        return [
            EmbeddingService._validate_text(t)
            for t in texts
        ]

    ####################################################################
    # Internal Encode
    ####################################################################

    def _encode(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:

        try:

            embeddings = self.model.encode(

                list(texts),

                batch_size=self.batch_size,

                convert_to_numpy=True,

                normalize_embeddings=self.normalize,

                show_progress_bar=False,

            )

            return embeddings

        except Exception as exc:

            logger.exception(exc)

            raise EmbeddingError(str(exc))

    ####################################################################
    # Statistics
    ####################################################################

    def statistics(self) -> dict:

        return {

            "model": self.model_name,

            "dimension": self.embedding_dimension,

            "device": self.device,

            "batch_size": self.batch_size,

            "normalized": self.normalize,

        }

    ####################################################################
    # Health
    ####################################################################

    def health(self) -> dict:

        try:

            _ = self.embedding_dimension

            return {

                "status": "healthy",

                "model": self.model_name,

                "device": self.device,

            }

        except Exception as exc:

            return {

                "status": "failed",

                "error": str(exc),

            }

    ####################################################################
    # Single Text Embedding
    ####################################################################

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Generate embedding for a single text.

        Parameters
        ----------
        text : str

        Returns
        -------
        numpy.ndarray
        """

        text = self._validate_text(text)

        embedding = self._encode([text])

        return embedding[0]

    ####################################################################
    # Query Embedding
    ####################################################################

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Generate embedding for search queries.
        """

        return self.embed_text(query)

    ####################################################################
    # Document Embedding
    ####################################################################

    def embed_document(
        self,
        document: str,
    ) -> np.ndarray:
        """
        Generate embedding for an entire document.
        """

        return self.embed_text(document)

    ####################################################################
    # Multiple Documents
    ####################################################################

    def embed_documents(
        self,
        documents: Sequence[str],
    ) -> np.ndarray:
        """
        Embed a list of documents.
        """

        documents = self._validate_batch(documents)

        return self._encode(documents)

    ####################################################################
    # Batch Embedding
    ####################################################################

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Alias for embed_documents().
        """

        return self.embed_documents(texts)

    ####################################################################
    # Iterator Support
    ####################################################################

    def embed_iterable(
        self,
        texts: Iterable[str],
    ) -> np.ndarray:
        """
        Embed any iterable of strings.
        """

        texts = list(texts)

        return self.embed_documents(texts)

    ####################################################################
    # Large Dataset Embedding
    ####################################################################

    def embed_large_collection(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Embed very large collections efficiently
        using mini-batches.
        """

        texts = self._validate_batch(texts)

        vectors = []

        total = len(texts)

        logger.info(
            "Embedding %d documents...",
            total,
        )

        for start in range(

            0,

            total,

            self.batch_size,

        ):

            end = start + self.batch_size

            batch = texts[start:end]

            batch_vectors = self._encode(batch)

            vectors.append(batch_vectors)

        return np.vstack(vectors)

    ####################################################################
    # Embedding Dictionary
    ####################################################################

    def embed_dictionary(
        self,
        data: dict,
    ) -> dict:
        """
        Embed every string value in a dictionary.

        Returns
        -------
        {
            key : embedding
        }
        """

        results = {}

        for key, value in data.items():

            if isinstance(value, str):

                results[key] = self.embed_text(value)

        return results

    ####################################################################
    # Pair Embeddings
    ####################################################################

    def embed_pair(
        self,
        first: str,
        second: str,
    ):
        """
        Embed two texts simultaneously.
        """

        vectors = self.embed_documents(

            [

                first,

                second,

            ]

        )

        return vectors[0], vectors[1]

    ####################################################################
    # Matrix Embedding
    ####################################################################

    def embedding_matrix(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Returns embedding matrix.

        Shape

        (N, Dimension)
        """

        return self.embed_documents(texts)

    ####################################################################
    # Normalize Existing Embeddings
    ####################################################################

    @staticmethod
    def normalize_vectors(
        vectors: np.ndarray,
    ) -> np.ndarray:
        """
        L2 Normalize vectors.
        """

        norms = np.linalg.norm(

            vectors,

            axis=1,

            keepdims=True,

        )

        norms[norms == 0] = 1

        return vectors / norms

    ####################################################################
    # Mean Embedding
    ####################################################################

    def mean_embedding(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Compute average embedding
        across multiple texts.
        """

        vectors = self.embed_documents(texts)

        return np.mean(

            vectors,

            axis=0,

        )

    ####################################################################
    # Embedding Metadata
    ####################################################################

    def embedding_info(
        self,
        vector: np.ndarray,
    ) -> dict:
        """
        Return metadata
        about a vector.
        """

        return {

            "dimension": len(vector),

            "dtype": str(vector.dtype),

            "shape": vector.shape,

            "normalized": self.normalize,

        }

    ####################################################################
    # Empty Vector
    ####################################################################

    def empty_vector(self):

        return np.zeros(

            self.embedding_dimension,

            dtype=np.float32,

        )
