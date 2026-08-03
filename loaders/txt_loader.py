from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_loader import BaseLoader, LoaderError

logger = logging.getLogger(__name__)


# ==========================================================
# TXT Loader
# ==========================================================

class TXTLoader(BaseLoader):
    """
    Production TXT Loader.
    """

    SUPPORTED_EXTENSIONS = [
        ".txt",
        ".text",
    ]

    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ):

        super().__init__(file_path)

        self.validate_extension(
            self.SUPPORTED_EXTENSIONS
        )

        self.encoding = encoding
