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
