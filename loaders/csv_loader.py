from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_loader import BaseLoader, LoaderError

logger = logging.getLogger(__name__)


# ==========================================================
# CSV Loader
# ==========================================================

class CSVLoader(BaseLoader):
    """
    Production CSV Loader.

    Supported Formats
    -----------------
    .csv
    """

    SUPPORTED_EXTENSIONS = [

        ".csv",

    ]

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        delimiter: str = ",",
    ):

        super().__init__(file_path)

        self.validate_extension(

            self.SUPPORTED_EXTENSIONS

        )

        self.encoding = encoding

        self.delimiter = delimiter

        self.dataframe: Optional[
            pd.DataFrame
        ] = None

        logger.info(

            f"CSVLoader initialized "

            f"for {self.file_name}"

        )

    # ======================================================
    # Read CSV
    # ======================================================

    def read_csv(
        self,
    ) -> pd.DataFrame:
        """
        Read CSV file.
        """

        if self.dataframe is None:

            try:

                self.dataframe = pd.read_csv(

                    self.file_path,

                    encoding=self.encoding,

                    sep=self.delimiter,

                )

            except Exception as error:

                raise LoaderError(

                    f"Unable to load CSV: "

                    f"{error}"

                )

        return self.dataframe

    # ======================================================
    # Row Count
    # ======================================================

    @property
    def row_count(
        self,
    ) -> int:
        """
        Number of rows.
        """

        return len(

            self.read_csv()

        )

    # ======================================================
    # Column Count
    # ======================================================

    @property
    def column_count(
        self,
    ) -> int:
        """
        Number of columns.
        """

        return len(

            self.read_csv().columns

        )

    # ======================================================
    # Schema
    # ======================================================

    def schema(
        self,
    ) -> Dict[str, str]:
        """
        Column data types.
        """

        dataframe = self.read_csv()

        return {

            column:

                str(

                    dtype

                )

            for column, dtype

            in dataframe.dtypes.items()

        }

    # ======================================================
    # Metadata
    # ======================================================

    def csv_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        CSV metadata.
        """

        return {

            "rows":

                self.row_count,

            "columns":

                self.column_count,

            "file_size":

                self.file_size,

            "encoding":

                self.encoding,

            "delimiter":

                self.delimiter,

            "extension":

                self.extension,

        }

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Basic CSV statistics.
        """

        dataframe = self.read_csv()

        return {

            "rows":

                self.row_count,

            "columns":

                self.column_count,

            "memory_usage":

                int(

                    dataframe.memory_usage(

                        deep=True

                    ).sum()

                ),

            "numeric_columns":

                len(

                    dataframe.select_dtypes(

                        include="number"

                    ).columns

                ),

            "categorical_columns":

                len(

                    dataframe.select_dtypes(

                        exclude="number"

                    ).columns

                ),

        }

    # ======================================================
    # Extract Text
    # ======================================================

    def extract_text(
        self,
    ) -> str:
        """
        Convert CSV to text.
        """

        dataframe = self.read_csv()

        return dataframe.to_csv(

            index=False

        )

    # ======================================================
    # Load
    # ======================================================

    def load(
        self,
    ) -> Dict[str, Any]:
        """
        Load CSV.
        """

        dataframe = self.read_csv()

        metadata = self.csv_metadata()

        metadata.update(

            self.statistics()

        )

        return {

            "text":

                self.extract_text(),

            "dataframe":

                dataframe,

            "rows":

                self.row_count,

            "columns":

                self.column_count,

            "metadata":

                metadata,

        }

    # ======================================================
    # Preview
    # ======================================================

    def preview(
        self,
        rows: int = 5,
    ) -> pd.DataFrame:
        """
        Preview rows.
        """

        return self.read_csv().head(

            rows

        )

    # ======================================================
    # Empty Check
    # ======================================================

    def is_empty(
        self,
    ) -> bool:
        """
        Check empty CSV.
        """

        return self.row_count == 0

    # ======================================================
    # String Representation
    # ======================================================

    def __repr__(
        self,
    ):

        return (

            "CSVLoader("

            f"file='{self.file_name}', "

            f"rows={self.row_count}, "

            f"columns={self.column_count}"

            ")"

        )
