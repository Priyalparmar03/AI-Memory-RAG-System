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

# ======================================================
# Column Names
# ======================================================

def column_names(
    self,
) -> List[str]:
    """
    Return all column names.
    """

    return list(

        self.read_csv().columns

    )


# ======================================================
# Numeric Columns
# ======================================================

def numeric_columns(
    self,
) -> List[str]:
    """
    Return numeric columns.
    """

    dataframe = self.read_csv()

    return list(

        dataframe.select_dtypes(

            include="number"

        ).columns

    )


# ======================================================
# Categorical Columns
# ======================================================

def categorical_columns(
    self,
) -> List[str]:
    """
    Return categorical columns.
    """

    dataframe = self.read_csv()

    return list(

        dataframe.select_dtypes(

            exclude="number"

        ).columns

    )


# ======================================================
# Missing Values
# ======================================================

def missing_values(
    self,
) -> Dict[str, Any]:
    """
    Missing value analysis.
    """

    dataframe = self.read_csv()

    missing = dataframe.isnull().sum()

    percentage = (

        dataframe.isnull().mean()

        * 100

    )

    return {

        column: {

            "count":

                int(

                    missing[column]

                ),

            "percentage":

                round(

                    float(

                        percentage[column]

                    ),

                    2,

                ),

        }

        for column

        in dataframe.columns

    }


# ======================================================
# Duplicate Rows
# ======================================================

def duplicates(
    self,
) -> Dict[str, Any]:
    """
    Duplicate row analysis.
    """

    dataframe = self.read_csv()

    duplicate_rows = int(

        dataframe.duplicated().sum()

    )

    return {

        "duplicate_rows":

            duplicate_rows,

        "unique_rows":

            self.row_count

            -

            duplicate_rows,

    }


# ======================================================
# Sample Rows
# ======================================================

def sample(
    self,
    n: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Random sample.
    """

    dataframe = self.read_csv()

    n = min(

        n,

        len(dataframe),

    )

    return dataframe.sample(

        n=n,

        random_state=random_state,

    )


# ======================================================
# Search
# ======================================================

def search(
    self,
    keyword: str,
) -> pd.DataFrame:
    """
    Search keyword in CSV.
    """

    dataframe = self.read_csv()

    keyword = str(

        keyword

    ).lower()

    mask = dataframe.astype(

        str

    ).apply(

        lambda column:

            column.str.lower().str.contains(

                keyword,

                na=False,

            )

    ).any(

        axis=1

    )

    return dataframe[mask]


# ======================================================
# Filter Rows
# ======================================================

def filter_rows(
    self,
    column: str,
    value: Any,
) -> pd.DataFrame:
    """
    Filter rows by value.
    """

    dataframe = self.read_csv()

    if column not in dataframe.columns:

        raise LoaderError(

            f"Column '{column}' "

            "not found."

        )

    return dataframe[

        dataframe[column]

        ==

        value

    ]


# ======================================================
# Unique Values
# ======================================================

def unique_values(
    self,
    column: str,
) -> List[Any]:
    """
    Unique values in column.
    """

    dataframe = self.read_csv()

    if column not in dataframe.columns:

        raise LoaderError(

            f"Column '{column}' "

            "not found."

        )

    return dataframe[

        column

    ].dropna().unique().tolist()


# ======================================================
# Row by Index
# ======================================================

def row(
    self,
    index: int,
) -> Dict[str, Any]:
    """
    Return row by index.
    """

    dataframe = self.read_csv()

    if (

        index < 0

        or

        index >= len(dataframe)

    ):

        raise LoaderError(

            "Invalid row index."

        )

    return dataframe.iloc[

        index

    ].to_dict()


# ======================================================
# Column Values
# ======================================================

def column(
    self,
    column_name: str,
) -> List[Any]:
    """
    Return column values.
    """

    dataframe = self.read_csv()

    if (

        column_name

        not in dataframe.columns

    ):

        raise LoaderError(

            f"Column '{column_name}' "

            "not found."

        )

    return dataframe[

        column_name

    ].tolist()


# ======================================================
# Null Row Count
# ======================================================

def null_row_count(
    self,
) -> int:
    """
    Rows containing at least one
    missing value.
    """

    dataframe = self.read_csv()

    return int(

        dataframe.isnull().any(

            axis=1

        ).sum()

    )

# ======================================================
# Describe Dataset
# ======================================================

def describe(
    self,
) -> pd.DataFrame:
    """
    Statistical summary.
    """

    return self.read_csv().describe(

        include="all"

    )


# ======================================================
# Correlation Matrix
# ======================================================

def correlation(
    self,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute correlation matrix.
    """

    dataframe = self.read_csv()

    numeric = dataframe.select_dtypes(

        include="number"

    )

    if numeric.empty:

        raise LoaderError(

            "No numeric columns found."

        )

    return numeric.corr(

        method=method

    )


# ======================================================
# Value Counts
# ======================================================

def value_counts(
    self,
    column: str,
    normalize: bool = False,
) -> pd.Series:
    """
    Frequency count for a column.
    """

    dataframe = self.read_csv()

    if column not in dataframe.columns:

        raise LoaderError(

            f"Column '{column}' not found."

        )

    return dataframe[

        column

    ].value_counts(

        normalize=normalize,

        dropna=False,

    )


# ======================================================
# Export JSON
# ======================================================

def export_json(
    self,
    output_path: str,
    orient: str = "records",
) -> str:
    """
    Export CSV as JSON.
    """

    dataframe = self.read_csv()

    dataframe.to_json(

        output_path,

        orient=orient,

        indent=4,

    )

    return output_path


# ======================================================
# Export Dictionary
# ======================================================

def export_dict(
    self,
    orient: str = "records",
) -> Any:
    """
    Export CSV as dictionary.
    """

    dataframe = self.read_csv()

    return dataframe.to_dict(

        orient=orient,

    )


# ======================================================
# Preview Columns
# ======================================================

def preview_columns(
    self,
    columns: List[str],
    rows: int = 5,
) -> pd.DataFrame:
    """
    Preview selected columns.
    """

    dataframe = self.read_csv()

    missing = [

        column

        for column

        in columns

        if column not in dataframe.columns

    ]

    if missing:

        raise LoaderError(

            f"Columns not found: {missing}"

        )

    return dataframe[

        columns

    ].head(

        rows

    )


# ======================================================
# Dataset Summary
# ======================================================

def summary(
    self,
) -> Dict[str, Any]:
    """
    Human-readable dataset summary.
    """

    stats = self.statistics()

    return {

        "file":

            self.file_name,

        "rows":

            stats["rows"],

        "columns":

            stats["columns"],

        "numeric_columns":

            stats["numeric_columns"],

        "categorical_columns":

            stats["categorical_columns"],

        "memory_usage":

            stats["memory_usage"],

        "duplicates":

            self.duplicates()[

                "duplicate_rows"

            ],

        "null_rows":

            self.null_row_count(),

    }


# ======================================================
# Export Summary
# ======================================================

def export_summary(
    self,
) -> Dict[str, Any]:
    """
    Export complete dataset summary.
    """

    return {

        "metadata":

            self.csv_metadata(),

        "schema":

            self.schema(),

        "statistics":

            self.statistics(),

        "missing_values":

            self.missing_values(),

        "duplicates":

            self.duplicates(),

        "summary":

            self.summary(),

    }


# ======================================================
# Top Rows
# ======================================================

def head(
    self,
    rows: int = 5,
) -> pd.DataFrame:
    """
    Return first rows.
    """

    return self.read_csv().head(

        rows

    )


# ======================================================
# Bottom Rows
# ======================================================

def tail(
    self,
    rows: int = 5,
) -> pd.DataFrame:
    """
    Return last rows.
    """

    return self.read_csv().tail(

        rows

    )


# ======================================================
# Shape
# ======================================================

@property
def shape(
    self,
) -> tuple[int, int]:
    """
    Dataset shape.
    """

    return self.read_csv().shape
