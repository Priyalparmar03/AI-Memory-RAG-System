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

    # ======================================================
    # Health Check
    # ======================================================

    def health(self) -> dict:
        """
        Perform a health check on the currently loaded model.
        """

        try:

            model = self.model

            return {

                "status": "healthy",

                "model": self.model_name,

                "device": self.device,

                "dimension": model.get_sentence_embedding_dimension(),

                "gpu": self.using_gpu,

            }

        except Exception as exc:

            logger.exception(exc)

            return {

                "status": "failed",

                "error": str(exc),

            }

    # ======================================================
    # Diagnostics
    # ======================================================

    def diagnostics(self) -> dict:
        """
        Complete diagnostics report.
        """

        return {

            "health": self.health(),

            "configuration": self.configuration(),

            "memory": self.memory_usage(),

            "runtime": self.runtime_statistics(),

            "installed_models": self.installed_models(),

        }

    # ======================================================
    # GPU / Memory Statistics
    # ======================================================

    def memory_usage(self) -> dict:
        """
        GPU/CPU memory information.
        """

        stats = {

            "device": self.device,

            "gpu_enabled": self.using_gpu,

        }

        if torch.cuda.is_available():

            stats.update(

                {

                    "allocated_mb": round(

                        torch.cuda.memory_allocated()

                        / (1024 ** 2),

                        2,

                    ),

                    "reserved_mb": round(

                        torch.cuda.memory_reserved()

                        / (1024 ** 2),

                        2,

                    ),

                    "max_allocated_mb": round(

                        torch.cuda.max_memory_allocated()

                        / (1024 ** 2),

                        2,

                    ),

                }

            )

        return stats

    # ======================================================
    # Runtime Statistics
    # ======================================================

    def runtime_statistics(self) -> dict:
        """
        Runtime statistics.
        """

        model = self.model

        return {

            "model_loaded": self._model is not None,

            "dimension": model.get_sentence_embedding_dimension(),

            "max_sequence_length": getattr(

                model,

                "max_seq_length",

                512,

            ),

            "device": self.device,

            "gpu": self.using_gpu,

        }

    # ======================================================
    # Benchmark
    # ======================================================

    def benchmark(
        self,
        samples: int = 100,
    ) -> dict:
        """
        Benchmark embedding speed.
        """

        import time

        texts = [

            f"Embedding benchmark {i}"

            for i in range(samples)

        ]

        start = time.perf_counter()

        self.model.encode(

            texts,

            batch_size=32,

            show_progress_bar=False,

        )

        elapsed = time.perf_counter() - start

        return {

            "samples": samples,

            "seconds": round(

                elapsed,

                4,

            ),

            "embeddings_per_second": round(

                samples / max(elapsed, 1e-6),

                2,

            ),

        }

    # ======================================================
    # Warmup
    # ======================================================

    def warmup(self):
        """
        Warm up the embedding model.
        """

        logger.info("Running model warmup...")

        self.model.encode(

            [

                "Warmup sentence."

            ],

            show_progress_bar=False,

        )

        logger.info("Warmup complete.")

    # ======================================================
    # Device Information
    # ======================================================

    def device_information(self) -> dict:
        """
        Return device information.
        """

        info = {

            "device": self.device,

            "cuda_available": torch.cuda.is_available(),

        }

        if torch.cuda.is_available():

            info["gpu_name"] = torch.cuda.get_device_name(0)

            info["device_count"] = torch.cuda.device_count()

        return info

    # ======================================================
    # Context Length
    # ======================================================

    def context_length(self) -> int:
        """
        Maximum supported sequence length.
        """

        return getattr(

            self.model,

            "max_seq_length",

            512,

        )

    # ======================================================
    # Embedding Dimension
    # ======================================================

    def embedding_dimension(self) -> int:
        """
        Embedding vector size.
        """

        return self.model.get_sentence_embedding_dimension()

    # ======================================================
    # GPU Available
    # ======================================================

    @staticmethod
    def gpu_available() -> bool:

        return torch.cuda.is_available()

    # ======================================================
    # CPU Available
    # ======================================================

    @staticmethod
    def cpu_available() -> bool:

        return True

    # ======================================================
    # Reset Manager
    # ======================================================

    def reset(self) -> None:
        """
        Reset the manager to its default state.
        """

        logger.info("Resetting EmbeddingModelManager...")

        self.unload_model()

        self.model_name = self.DEFAULT_MODEL

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            "Reset complete. Using model=%s device=%s",
            self.model_name,
            self.device,
        )

    # ======================================================
    # Clear CUDA Cache
    # ======================================================

    def clear_gpu_cache(self) -> None:
        """
        Release unused CUDA memory.
        """

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            logger.info("CUDA cache cleared.")

    # ======================================================
    # Garbage Collection
    # ======================================================

    @staticmethod
    def collect_garbage() -> None:
        """
        Run Python garbage collection.
        """

        import gc

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ======================================================
    # Model Summary
    # ======================================================

    def summary(self) -> dict:
        """
        Human-readable summary.
        """

        info = self.model_info()

        return {

            "name": info.name,

            "description": info.description,

            "dimension": info.dimension,

            "multilingual": info.multilingual,

            "max_sequence_length": info.max_sequence_length,

            "normalized": info.normalized,

            "device": self.device,

            "gpu": self.using_gpu,

        }

    # ======================================================
    # Export Configuration
    # ======================================================

    def export_configuration(self) -> dict:
        """
        Export current configuration.
        """

        return {

            "model": self.model_name,

            "device": self.device,

            "cache_folder": (
                str(self.cache_folder)
                if self.cache_folder
                else None
            ),

        }

    # ======================================================
    # Import Configuration
    # ======================================================

    def import_configuration(
        self,
        config: dict,
    ) -> None:
        """
        Restore configuration.
        """

        self.model_name = config.get(
            "model",
            self.DEFAULT_MODEL,
        )

        self.device = config.get(
            "device",
            self.device,
        )

        cache = config.get("cache_folder")

        self.cache_folder = (
            Path(cache)
            if cache
            else None
        )

        self.reload_model()

    # ======================================================
    # Compare Models
    # ======================================================

    def compare_models(
        self,
        first: str,
        second: str,
    ) -> dict:
        """
        Compare two registered models.
        """

        if not self.model_exists(first):

            raise EmbeddingModelError(first)

        if not self.model_exists(second):

            raise EmbeddingModelError(second)

        model_a = self.model_info(first)

        model_b = self.model_info(second)

        return {

            "first": model_a,

            "second": model_b,

            "dimension_difference": (

                model_a.dimension
                - model_b.dimension

            ),

            "same_dimension": (

                model_a.dimension
                == model_b.dimension

            ),

        }

    # ======================================================
    # Close
    # ======================================================

    def close(self) -> None:
        """
        Release resources.
        """

        logger.info(
            "Closing EmbeddingModelManager..."
        )

        self.unload_model()

        self.collect_garbage()

        logger.info(
            "EmbeddingModelManager closed."
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
    # Representation
    # ======================================================

    def __repr__(self) -> str:

        return (

            "EmbeddingModelManager("

            f"model='{self.model_name}', "

            f"device='{self.device}', "

            f"gpu={self.using_gpu}, "

            f"dimension={self.dimension}"

            ")"

        )
