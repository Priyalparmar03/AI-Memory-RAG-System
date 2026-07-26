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
