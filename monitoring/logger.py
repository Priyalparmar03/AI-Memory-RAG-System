from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading

from dataclasses import (
    dataclass,
    field,
)

from datetime import datetime

from enum import Enum

from pathlib import Path

from typing import (
    Any,
    Dict,
    Optional,
)

# ==========================================================
# Log Level
# ==========================================================

class LogLevel(str, Enum):

    DEBUG = "DEBUG"

    INFO = "INFO"

    WARNING = "WARNING"

    ERROR = "ERROR"

    CRITICAL = "CRITICAL"


# ==========================================================
# Log Format
# ==========================================================

class LogFormat(str, Enum):

    TEXT = "text"

    JSON = "json"


# ==========================================================
# Logger Configuration
# ==========================================================

@dataclass(slots=True)
class LoggerConfig:
    """
    Logger configuration.
    """

    name: str = "AI_MEMORY"

    level: LogLevel = LogLevel.INFO

    log_format: LogFormat = LogFormat.TEXT

    log_directory: str = "logs"

    file_name: str = "application.log"

    max_bytes: int = 20 * 1024 * 1024

    backup_count: int = 10

    console_logging: bool = True

    file_logging: bool = True

    json_logging: bool = False

    propagate: bool = False

    encoding: str = "utf-8"


# ==========================================================
# JSON Formatter
# ==========================================================

class JsonFormatter(
    logging.Formatter
):
    """
    JSON log formatter.
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:

        log_record = {

            "timestamp":

                datetime.utcnow().isoformat(),

            "level":

                record.levelname,

            "logger":

                record.name,

            "module":

                record.module,

            "function":

                record.funcName,

            "line":

                record.lineno,

            "thread":

                threading.current_thread().name,

            "message":

                record.getMessage(),

        }

        if record.exc_info:

            log_record["exception"] = (

                self.formatException(

                    record.exc_info

                )

            )

        return json.dumps(

            log_record,

            ensure_ascii=False,

        )


# ==========================================================
# Logger Manager
# ==========================================================

class LoggerManager:
    """
    Production logger manager.
    """

    def __init__(
        self,
        config: Optional[
            LoggerConfig
        ] = None,
    ):

        self.config = (

            config

            or

            LoggerConfig()

        )

        self.logger = logging.getLogger(

            self.config.name

        )

        self.logger.setLevel(

            getattr(

                logging,

                self.config.level.value,

            )

        )

        self.logger.propagate = (

            self.config.propagate

        )

        self._configure()


    # ======================================================
    # Configure Logger
    # ======================================================

    def _configure(
        self,
    ) -> None:
        """
        Configure handlers.
        """

        self.logger.handlers.clear()

        Path(

            self.config.log_directory

        ).mkdir(

            parents=True,

            exist_ok=True,

        )

        formatter = (

            JsonFormatter()

            if self.config.json_logging

            else logging.Formatter(

                "%(asctime)s | "

                "%(levelname)-8s | "

                "%(name)s | "

                "%(module)s:%(lineno)d | "

                "%(message)s"

            )

        )

        if self.config.console_logging:

            console = (

                logging.StreamHandler(

                    sys.stdout

                )

            )

            console.setFormatter(

                formatter

            )

            self.logger.addHandler(

                console

            )

        if self.config.file_logging:

            file_handler = (

                logging.handlers.

                RotatingFileHandler(

                    filename=os.path.join(

                        self.config.log_directory,

                        self.config.file_name,

                    ),

                    maxBytes=self.config.max_bytes,

                    backupCount=self.config.backup_count,

                    encoding=self.config.encoding,

                )

            )

            file_handler.setFormatter(

                formatter

            )

            self.logger.addHandler(

                file_handler

            )


    # ======================================================
    # Debug
    # ======================================================

    def debug(
        self,
        message: str,
        *args,
        **kwargs,
    ) -> None:

        self.logger.debug(

            message,

            *args,

            **kwargs,

        )


    # ======================================================
    # Info
    # ======================================================

    def info(
        self,
        message: str,
        *args,
        **kwargs,
    ) -> None:

        self.logger.info(

            message,

            *args,

            **kwargs,

        )


    # ======================================================
    # Warning
    # ======================================================

    def warning(
        self,
        message: str,
        *args,
        **kwargs,
    ) -> None:

        self.logger.warning(

            message,

            *args,

            **kwargs,

        )


    # ======================================================
    # Error
    # ======================================================

    def error(
        self,
        message: str,
        *args,
        **kwargs,
    ) -> None:

        self.logger.error(

            message,

            *args,

            **kwargs,

        )


    # ======================================================
    # Critical
    # ======================================================

    def critical(
        self,
        message: str,
        *args,
        **kwargs,
    ) -> None:

        self.logger.critical(

            message,

            *args,

            **kwargs,

        )


    # ======================================================
    # Exception
    # ======================================================

    def exception(
        self,
        message: str,
        *args,
        **kwargs,
    ) -> None:

        self.logger.exception(

            message,

            *args,

            **kwargs,

        )
