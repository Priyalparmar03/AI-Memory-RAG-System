from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
