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

    # ======================================================
    # Read File
    # ======================================================

    def read_file(
        self,
    ) -> str:

        try:

            with open(

                self.file_path,

                "r",

                encoding=self.encoding,

            ) as file:

                return file.read()

        except UnicodeDecodeError:

            raise LoaderError(

                "Unsupported encoding."

            )

        except Exception as error:

            raise LoaderError(

                str(error)

            )

    # ======================================================
    # Read Lines
    # ======================================================

    def read_lines(
        self,
    ) -> List[str]:

        try:

            with open(

                self.file_path,

                "r",

                encoding=self.encoding,

            ) as file:

                return [

                    line.rstrip()

                    for line in file

                ]

        except Exception as error:

            raise LoaderError(

                str(error)

            )

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:

        text = self.read_file()

        lines = text.splitlines()

        words = text.split()

        return {

            "characters": len(text),

            "lines": len(lines),

            "words": len(words),

            "empty_lines": sum(

                1

                for line in lines

                if not line.strip()

            ),

        }

    # ======================================================
    # Load
    # ======================================================

    def load(
        self,
    ) -> Dict[str, Any]:

        text = self.read_file()

        stats = self.statistics()

        return self.build_result(

            text=text,

            extra_metadata=stats,

        )


    # ======================================================
    # Preview
    # ======================================================

    def preview(
        self,
        characters: int = 500,
    ) -> str:

        return self.read_file()[:characters]

    # ======================================================
    # Is Empty
    # ======================================================

    def is_empty(
        self,
    ) -> bool:

        return self.file_size == 0

    # ======================================================
    # Number of Lines
    # ======================================================

    def line_count(
        self,
    ) -> int:

        return len(

            self.read_lines()

        )

    # ======================================================
    # String Representation
    # ======================================================

    def __repr__(
        self,
    ):

        return (

            f"TXTLoader("

            f"{self.file_name}"

            ")"

        )

