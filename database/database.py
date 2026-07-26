"""
Responsibilities
----------------
- SQLite Connection Management
- Thread-safe Database Access
- Context Manager
- Schema Initialization
- Transactions
- WAL Mode
- Health Monitoring
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


class Database:
    """
    Enterprise SQLite Database Manager.

    Features
    --------
    • Singleton
    • Thread-safe
    • WAL Mode
    • Foreign Keys
    • Context Manager
    • Automatic Schema Initialization
    """

    _instance: Optional["Database"] = None
    _lock = threading.Lock()

    def __new__(
        cls,
        database_path: str = "database/database.db",
        schema_path: str = "database/schema.sql",
    ):

        if cls._instance is None:

            with cls._lock:

                if cls._instance is None:

                    cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(
        self,
        database_path: str = "database/database.db",
        schema_path: str = "database/schema.sql",
    ):

        if getattr(self, "_initialized", False):
            return

        self.database_path = Path(database_path)
        self.schema_path = Path(schema_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection: Optional[sqlite3.Connection] = None

        self._initialized = True

        logger.info(
            "Database initialized."
        )

    # =====================================================
    # Connection
    # =====================================================

    def connect(
        self,
    ) -> sqlite3.Connection:
        """
        Return active SQLite connection.
        """

        if self.connection is None:

            logger.info(
                "Opening SQLite connection..."
            )

            self.connection = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
            )

            self.connection.row_factory = sqlite3.Row

            self._configure_database()

        return self.connection

    # =====================================================
    # Database Configuration
    # =====================================================

    def _configure_database(self) -> None:
        """
        Configure SQLite.
        """

        connection = self.connection

        if connection is None:
            return

        cursor = connection.cursor()

        cursor.execute(
            "PRAGMA foreign_keys = ON;"
        )

        cursor.execute(
            "PRAGMA journal_mode = WAL;"
        )

        cursor.execute(
            "PRAGMA synchronous = NORMAL;"
        )

        cursor.execute(
            "PRAGMA temp_store = MEMORY;"
        )

        cursor.execute(
            "PRAGMA cache_size = -64000;"
        )

        cursor.close()

        connection.commit()

        logger.info(
            "SQLite configured successfully."
        )

    # =====================================================
    # Initialize Schema
    # =====================================================

    def initialize(self) -> None:
        """
        Initialize database schema.
        """

        if not self.schema_path.exists():

            logger.warning(
                "Schema file not found: %s",
                self.schema_path,
            )

            return

        logger.info(
            "Initializing database schema..."
        )

        connection = self.connect()

        with self.schema_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            sql = file.read()

        connection.executescript(sql)

        connection.commit()

        logger.info(
            "Database schema initialized."
        )

    # =====================================================
    # Cursor Context Manager
    # =====================================================

    @contextmanager
    def cursor(
        self,
    ) -> Generator[sqlite3.Cursor, None, None]:
        """
        Context-managed database cursor.
        """

        connection = self.connect()

        cursor = connection.cursor()

        try:

            yield cursor

            connection.commit()

        except Exception:

            connection.rollback()

            logger.exception(
                "Database transaction failed."
            )

            raise

        finally:

            cursor.close()

    # =====================================================
    # Connection Status
    # =====================================================

    def is_connected(self) -> bool:
        """
        Return True if database connection exists.
        """

        return self.connection is not None

    # =====================================================
    # Database Path
    # =====================================================

    @property
    def path(self) -> str:
        """
        Return database path.
        """

        return str(self.database_path)

    # =====================================================
    # Close Connection
    # =====================================================

    def close(self) -> None:
        """
        Close SQLite connection.
        """

        if self.connection is not None:

            logger.info(
                "Closing database connection..."
            )

            self.connection.close()

            self.connection = None

            logger.info(
                "Database connection closed."
            )
    
    # =====================================================
    # Execute Query
    # =====================================================
    
    def execute(
        self,
        query: str,
        parameters: tuple = (),
    ) -> sqlite3.Cursor:
        """
        Execute a single SQL query.
        """
    
        with self.cursor() as cursor:
    
            cursor.execute(
                query,
                parameters,
            )
    
            return cursor
    
    
    # =====================================================
    # Execute Many
    # =====================================================
    
    def executemany(
        self,
        query: str,
        parameters: list[tuple],
    ) -> None:
        """
        Execute multiple SQL statements.
        """
    
        with self.cursor() as cursor:
    
            cursor.executemany(
                query,
                parameters,
            )
    
        logger.info(
            "Executed %d statements.",
            len(parameters),
        )
    
    
    # =====================================================
    # Fetch One
    # =====================================================
    
    def fetchone(
        self,
        query: str,
        parameters: tuple = (),
    ) -> Optional[Dict[str, Any]]:
        """
        Execute query and return one row.
        """
    
        connection = self.connect()
    
        cursor = connection.cursor()
    
        try:
    
            cursor.execute(
                query,
                parameters,
            )
    
            row = cursor.fetchone()
    
            if row is None:
                return None
    
            return dict(row)
    
        finally:
    
            cursor.close()
    
    
    # =====================================================
    # Fetch All
    # =====================================================
    
    def fetchall(
        self,
        query: str,
        parameters: tuple = (),
    ) -> list[Dict[str, Any]]:
        """
        Execute query and return all rows.
        """
    
        connection = self.connect()
    
        cursor = connection.cursor()
    
        try:
    
            cursor.execute(
                query,
                parameters,
            )
    
            rows = cursor.fetchall()
    
            return [
                dict(row)
                for row in rows
            ]
    
        finally:
    
            cursor.close()

    # =====================================================
    # Fetch Value
    # =====================================================
    
    def fetch_value(
        self,
        query: str,
        parameters: tuple = (),
    ):
        """
        Return first column of first row.
        """
    
        connection = self.connect()
    
        cursor = connection.cursor()
    
        try:
    
            cursor.execute(
                query,
                parameters,
            )
    
            row = cursor.fetchone()
    
            if row is None:
                return None
    
            return row[0]
    
        finally:
    
            cursor.close()
            

# =====================================================
# Insert
# =====================================================

def insert(
    self,
    query: str,
    parameters: tuple,
) -> int:
    """
    Execute INSERT statement.

    Returns inserted row id.
    """

    with self.cursor() as cursor:

        cursor.execute(
            query,
            parameters,
        )

        return cursor.lastrowid


# =====================================================
# Update
# =====================================================

def update(
    self,
    query: str,
    parameters: tuple,
) -> int:
    """
    Execute UPDATE.

    Returns affected rows.
    """

    with self.cursor() as cursor:

        cursor.execute(
            query,
            parameters,
        )

        return cursor.rowcount

# =====================================================
# Delete
# =====================================================

def delete(
    self,
    query: str,
    parameters: tuple,
) -> int:
    """
    Execute DELETE.

    Returns affected rows.
    """

    with self.cursor() as cursor:

        cursor.execute(
            query,
            parameters,
        )

        return cursor.rowcount

# =====================================================
# Exists
# =====================================================

def exists(
    self,
    query: str,
    parameters: tuple = (),
) -> bool:
    """
    Return True if at least one record exists.
    """

    result = self.fetch_value(
        query,
        parameters,
    )

    return result is not None

# =====================================================
# Transaction
# =====================================================

@contextmanager
def transaction(self):
    """
    Context manager for database transactions.

    Usage:
        with db.transaction() as cursor:
            cursor.execute(...)
            cursor.execute(...)
    """

    connection = self.connect()
    cursor = connection.cursor()

    try:

        cursor.execute("BEGIN")

        yield cursor

        connection.commit()

        logger.debug(
            "Transaction committed."
        )

    except Exception:

        connection.rollback()

        logger.exception(
            "Transaction rolled back."
        )

        raise

    finally:

        cursor.close()


# =====================================================
# Execute With Retry
# =====================================================

def execute_retry(
    self,
    query: str,
    parameters: tuple = (),
    retries: int = 3,
    delay: float = 0.5,
):
    """
    Retry SQL execution if database is locked.
    """

    import time

    last_exception = None

    for attempt in range(retries):

        try:

            return self.execute(
                query,
                parameters,
            )

        except sqlite3.OperationalError as exc:

            last_exception = exc

            if "locked" not in str(exc).lower():

                raise

            logger.warning(

                "Database locked. Retry %d/%d",

                attempt + 1,

                retries,

            )

            time.sleep(delay)

    raise last_exception

# =====================================================
# Bulk Insert
# =====================================================

def bulk_insert(
    self,
    query: str,
    rows: list[tuple],
) -> int:
    """
    Insert multiple rows.

    Returns number of inserted rows.
    """

    if not rows:
        return 0

    self.executemany(
        query,
        rows,
    )

    return len(rows)



# =====================================================
# Pagination
# =====================================================

def fetch_page(
    self,
    query: str,
    page: int = 1,
    page_size: int = 25,
    parameters: tuple = (),
):
    """
    Fetch paginated results.
    """

    offset = (page - 1) * page_size

    query = (
        query +
        " LIMIT ? OFFSET ?"
    )

    parameters = (
        *parameters,
        page_size,
        offset,
    )

    return self.fetchall(
        query,
        parameters,
    )



# =====================================================
# Count
# =====================================================

def count(
    self,
    table: str,
) -> int:
    """
    Count rows in table.
    """

    return self.fetch_value(
        f"SELECT COUNT(*) FROM {table}"
    )

# =====================================================
# Table Exists
# =====================================================

def table_exists(
    self,
    table_name: str,
) -> bool:
    """
    Check whether table exists.
    """

    result = self.fetch_value(

        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        AND name=?
        """,

        (table_name,),
    )

    return result is not None


# =====================================================
# List Tables
# =====================================================

def list_tables(self):
    """
    Return all database tables.
    """

    rows = self.fetchall(

        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name
        """

    )

    return [

        row["name"]

        for row in rows

    ]


# =====================================================
# Database Size
# =====================================================

def database_size(self):
    """
    Return database size.
    """

    if not self.database_path.exists():

        return 0

    return self.database_path.stat().st_size



# =====================================================
# Last Insert ID
# =====================================================

def last_insert_id(self):
    """
    Return SQLite last inserted row id.
    """

    return self.fetch_value(

        """
        SELECT last_insert_rowid()
        """

    )


# =====================================================
# Execute Script
# =====================================================

def execute_script(
    self,
    script: str,
):
    """
    Execute SQL script.
    """

    connection = self.connect()

    connection.executescript(
        script
    )

    connection.commit()

    logger.info(
        "SQL script executed."
    )

with db.transaction() as cursor:

    cursor.execute(

        """
        INSERT INTO users(name)
        VALUES(?)
        """,

        ("Priyal",)

    )

    cursor.execute(

        """
        INSERT INTO users(name)
        VALUES(?)
        """,

        ("OpenAI",)

    )

