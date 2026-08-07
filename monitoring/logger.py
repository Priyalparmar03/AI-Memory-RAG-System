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

import functools
import time
import traceback
from typing import Callable


# ======================================================
# Structured Logging
# ======================================================

def structured(
    self,
    level: LogLevel,
    message: str,
    **fields,
) -> None:
    """
    Structured logging with custom fields.
    """

    payload = {

        "timestamp":

            datetime.utcnow().isoformat(),

        "message":

            message,

        **fields,

    }

    log_message = json.dumps(

        payload,

        ensure_ascii=False,

    )

    self.logger.log(

        getattr(

            logging,

            level.value,

        ),

        log_message,

    )


# ======================================================
# Context Logging
# ======================================================

def context(
    self,
    message: str,
    context: Dict[str, Any],
    level: LogLevel = LogLevel.INFO,
) -> None:
    """
    Log with execution context.
    """

    self.structured(

        level=level,

        message=message,

        context=context,

    )


# ======================================================
# Request Logging
# ======================================================

def request(
    self,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    client_ip: str = "",
    user_id: str = "",
) -> None:
    """
    Log HTTP/API requests.
    """

    self.structured(

        level=LogLevel.INFO,

        message="HTTP Request",

        method=method,

        path=path,

        status=status_code,

        latency_ms=round(

            latency_ms,

            2,

        ),

        client_ip=client_ip,

        user_id=user_id,

    )


# ======================================================
# RAG Retrieval Logging
# ======================================================

def rag(
    self,
    query: str,
    retrieved_chunks: int,
    latency_ms: float,
) -> None:
    """
    Log RAG retrieval.
    """

    self.structured(

        level=LogLevel.INFO,

        message="RAG Retrieval",

        query=query,

        retrieved_chunks=retrieved_chunks,

        latency_ms=round(

            latency_ms,

            2,

        ),

    )


# ======================================================
# LLM Logging
# ======================================================

def llm(
    self,
    provider: str,
    model: str,
    tokens: int,
    cost: float,
    latency_ms: float,
) -> None:
    """
    Log LLM inference.
    """

    self.structured(

        level=LogLevel.INFO,

        message="LLM Inference",

        provider=provider,

        model=model,

        tokens=tokens,

        cost=cost,

        latency_ms=round(

            latency_ms,

            2,

        ),

    )


# ======================================================
# Exception Logging
# ======================================================

def log_exception(
    self,
    exception: Exception,
    message: str = "",
) -> None:
    """
    Log complete exception.
    """

    self.structured(

        level=LogLevel.ERROR,

        message=message or str(

            exception

        ),

        exception_type=type(

            exception

        ).__name__,

        traceback=traceback.format_exc(),

    )


# ======================================================
# Performance Logging
# ======================================================

def performance(
    self,
    operation: str,
    latency_ms: float,
    **extra,
) -> None:
    """
    Log performance metrics.
    """

    self.structured(

        level=LogLevel.INFO,

        message="Performance",

        operation=operation,

        latency_ms=round(

            latency_ms,

            2,

        ),

        **extra,

    )


# ======================================================
# Timer Decorator
# ======================================================

def timer(
    self,
    name: str = "",
):
    """
    Measure execution time.
    """

    def decorator(
        func: Callable,
    ):

        @functools.wraps(

            func

        )
        def wrapper(

            *args,

            **kwargs,

        ):

            start = time.perf_counter()

            try:

                result = func(

                    *args,

                    **kwargs,

                )

                elapsed = (

                    time.perf_counter()

                    -

                    start

                ) * 1000

                self.performance(

                    operation=

                    name

                    or

                    func.__name__,

                    latency_ms=elapsed,

                )

                return result

            except Exception as exc:

                self.log_exception(

                    exc,

                    message=f"Failed: "

                    f"{func.__name__}",

                )

                raise

        return wrapper

    return decorator


# ======================================================
# Logging Decorator
# ======================================================

def logged(
    self,
    level: LogLevel = LogLevel.INFO,
):
    """
    Automatically log
    function execution.
    """

    def decorator(
        func: Callable,
    ):

        @functools.wraps(

            func

        )
        def wrapper(

            *args,

            **kwargs,

        ):

            self.structured(

                level=level,

                message=

                f"Starting "

                f"{func.__name__}",

            )

            result = func(

                *args,

                **kwargs,

            )

            self.structured(

                level=level,

                message=

                f"Finished "

                f"{func.__name__}",

            )

            return result

        return wrapper

    return decorator


# ======================================================
# Audit Logging
# ======================================================

def audit(
    self,
    action: str,
    actor: str,
    resource: str,
    **extra,
) -> None:
    """
    Audit trail logging.
    """

    self.structured(

        level=LogLevel.INFO,

        message="Audit Event",

        action=action,

        actor=actor,

        resource=resource,

        **extra,

    )
