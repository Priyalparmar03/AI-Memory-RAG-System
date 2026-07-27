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
