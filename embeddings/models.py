from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ==========================================================
# Exceptions
# ==========================================================

class EmbeddingModelError(Exception):
    """Raised when model operations fail."""


# ==========================================================
# Model Information
# ==========================================================

@dataclass
class EmbeddingModelInfo:

    name: str

    dimension: int

    description: str

    multilingual: bool

    max_sequence_length: int

    normalized: bool


# ==========================================================
# Model Manager
# ==========================================================

class EmbeddingModelManager:
    """
    Production Embedding Model Manager.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    # ======================================================
    # Built-in Registry
    # ======================================================

    MODEL_REGISTRY = {

        "bge-small": {

            "hf": "BAAI/bge-small-en-v1.5",

            "dimension": 384,

            "description": "Fast production model",

            "multilingual": False,

        },

        "bge-base": {

            "hf": "BAAI/bge-base-en-v1.5",

            "dimension": 768,

            "description": "Balanced quality",

            "multilingual": False,

        },

        "bge-large": {

            "hf": "BAAI/bge-large-en-v1.5",

            "dimension": 1024,

            "description": "Highest quality",

            "multilingual": False,

        },

        "all-minilm": {

            "hf": "sentence-transformers/all-MiniLM-L6-v2",

            "dimension": 384,

            "description": "Lightweight",

            "multilingual": False,

        },

        "e5-base": {

            "hf": "intfloat/e5-base-v2",

            "dimension": 768,

            "description": "Excellent retrieval",

            "multilingual": True,

        },

        "e5-large": {

            "hf": "intfloat/e5-large-v2",

            "dimension": 1024,

            "description": "Large multilingual",

            "multilingual": True,

        },

        "gte-large": {

            "hf": "Alibaba-NLP/gte-large-en-v1.5",

            "dimension": 1024,

            "description": "Enterprise embeddings",

            "multilingual": False,

        },

        "nomic": {

            "hf": "nomic-ai/nomic-embed-text-v1",

            "dimension": 768,

            "description": "Open source",

            "multilingual": True,

        },

    }

    # ======================================================
    # Initialization
    # ======================================================

    def __init__(

        self,

        model_name: str | None = None,

        device: str | None = None,

        cache_folder: str | Path | None = None,

    ):

        self.model_name = model_name or self.DEFAULT_MODEL

        self.device = (

            device

            if device

            else (

                "cuda"

                if torch.cuda.is_available()

                else "cpu"

            )

        )

        self.cache_folder = (

            Path(cache_folder)

            if cache_folder

            else None

        )

        self._model = None

        self._lock = threading.Lock()

        logger.info(

            "EmbeddingModelManager initialized "

            "(model=%s device=%s)",

            self.model_name,

            self.device,

        )

    # ======================================================
    # Lazy Loading
    # ======================================================

    @property
    def model(self) -> SentenceTransformer:

        if self._model is None:

            with self._lock:

                if self._model is None:

                    logger.info(

                        "Loading model %s",

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

    # ======================================================
    # Basic Properties
    # ======================================================

    @property
    def current_model(self) -> str:

        return self.model_name

    @property
    def dimension(self) -> int:

        return self.model.get_sentence_embedding_dimension()

    @property
    def using_gpu(self) -> bool:

        return self.device == "cuda"

    @property
    def cache_directory(self):

        return self.cache_folder

    # ======================================================
    # Available Models
    # ======================================================

    def available_models(self) -> List[str]:
        """
        Return all supported models.
        """

        return sorted(
            self.MODEL_REGISTRY.keys()
        )

    # ======================================================
    # Model Exists
    # ======================================================

    def model_exists(
        self,
        model: str,
    ) -> bool:

        return model in self.MODEL_REGISTRY

    # ======================================================
    # Model Information
    # ======================================================

    def model_info(
        self,
        model: Optional[str] = None,
    ) -> EmbeddingModelInfo:
        """
        Return metadata about a model.
        """

        model = model or self.model_name

        if model in self.MODEL_REGISTRY:

            info = self.MODEL_REGISTRY[model]

            return EmbeddingModelInfo(

                name=model,

                dimension=info["dimension"],

                description=info["description"],

                multilingual=info["multilingual"],

                max_sequence_length=512,

                normalized=True,

            )

        if self._model:

            return EmbeddingModelInfo(

                name=model,

                dimension=self.dimension,

                description="Custom model",

                multilingual=False,

                max_sequence_length=getattr(
                    self._model,
                    "max_seq_length",
                    512,
                ),

                normalized=True,

            )

        raise EmbeddingModelError(
            f"Unknown model: {model}"
        )

    # ======================================================
    # Load Model
    # ======================================================

    def load_model(
        self,
        model_name: Optional[str] = None,
    ) -> SentenceTransformer:
        """
        Load an embedding model.
        """

        if model_name:

            if self.model_exists(model_name):

                model_name = self.MODEL_REGISTRY[
                    model_name
                ]["hf"]

            self.model_name = model_name

            self._model = None

        logger.info(
            "Loading model %s",
            self.model_name,
        )

        return self.model

    # ======================================================
    # Unload Model
    # ======================================================

    def unload_model(self):
        """
        Release model resources.
        """

        logger.info(
            "Unloading model..."
        )

        self._model = None

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ======================================================
    # Reload
    # ======================================================

    def reload_model(self):
        """
        Reload current model.
        """

        logger.info(
            "Reloading model..."
        )

        self.unload_model()

        return self.load_model(
            self.model_name
        )

    # ======================================================
    # Change Model
    # ======================================================

    def change_model(
        self,
        model: str,
    ):
        """
        Switch to another model.
        """

        logger.info(

            "Changing model %s -> %s",

            self.model_name,

            model,

        )

        return self.load_model(model)

    # ======================================================
    # Download
    # ======================================================

    def download_model(
        self,
        model: str,
    ):
        """
        Download model into cache.
        """

        logger.info(
            "Downloading %s",
            model,
        )

        if self.model_exists(model):

            model = self.MODEL_REGISTRY[
                model
            ]["hf"]

        SentenceTransformer(

            model,

            device=self.device,

            cache_folder=str(
                self.cache_folder
            )
            if self.cache_folder
            else None,

        )

        logger.info(
            "Download complete."
        )

    # ======================================================
    # Installed Models
    # ======================================================

    def installed_models(
        self,
    ) -> List[str]:
        """
        Return downloaded models.
        """

        if self.cache_folder is None:

            return []

        if not self.cache_folder.exists():

            return []

        return sorted(

            [

                p.name

                for p in self.cache_folder.iterdir()

                if p.is_dir()

            ]

        )

    # ======================================================
    # Validate Model
    # ======================================================

    def validate_model(
        self,
        model: str,
    ) -> bool:
        """
        Verify model can be loaded.
        """

        try:

            SentenceTransformer(

                model,

                device=self.device,

            )

            return True

        except Exception:

            return False

    # ======================================================
    # Current Configuration
    # ======================================================

    def configuration(self):

        return {

            "model": self.model_name,

            "device": self.device,

            "cache": str(
                self.cache_folder
            )
            if self.cache_folder
            else None,

            "gpu": self.using_gpu,

            "dimension": self.dimension,

        }

    # ======================================================
    # Supported Models
    # ======================================================

    def registry(self):

        return self.MODEL_REGISTRY.copy()
