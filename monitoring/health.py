from __future__ import annotations

import json
import logging
import platform
import shutil
import socket
import threading
import time
import uuid

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
    List,
    Optional,
)

import psutil


logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class HealthError(Exception):
    """
    Health monitoring exception.
    """
    pass


# ==========================================================
# Health Status
# ==========================================================

class HealthStatus(str, Enum):

    HEALTHY = "healthy"

    WARNING = "warning"

    CRITICAL = "critical"

    UNKNOWN = "unknown"


# ==========================================================
# Health Check
# ==========================================================

@dataclass(slots=True)
class HealthCheck:
    """
    Individual health check result.
    """

    name: str

    status: HealthStatus

    message: str

    value: Optional[float] = None

    threshold: Optional[float] = None

    metadata: Dict[str, Any] = field(

        default_factory=dict

    )

    checked_at: datetime = field(

        default_factory=datetime.utcnow

    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {

            "name":

                self.name,

            "status":

                self.status.value,

            "message":

                self.message,

            "value":

                self.value,

            "threshold":

                self.threshold,

            "metadata":

                self.metadata,

            "checked_at":

                self.checked_at.isoformat(),

        }


# ==========================================================
# Health Report
# ==========================================================

@dataclass(slots=True)
class HealthReport:
    """
    Complete health report.
    """

    id: str = field(

        default_factory=lambda:

        str(uuid.uuid4())

    )

    overall_status: HealthStatus = (

        HealthStatus.UNKNOWN

    )

    checks: List[HealthCheck] = field(

        default_factory=list

    )

    generated_at: datetime = field(

        default_factory=datetime.utcnow

    )

    metadata: Dict[str, Any] = field(

        default_factory=dict

    )

    def add_check(
        self,
        check: HealthCheck,
    ) -> None:

        self.checks.append(

            check

        )

    @property
    def total_checks(
        self,
    ) -> int:

        return len(

            self.checks

        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {

            "id":

                self.id,

            "overall_status":

                self.overall_status.value,

            "checks":

                [

                    check.to_dict()

                    for check

                    in self.checks

                ],

            "generated_at":

                self.generated_at.isoformat(),

            "metadata":

                self.metadata,

        }


# ==========================================================
# Health Monitor
# ==========================================================

class HealthMonitor:
    """
    Production health monitor.
    """

    def __init__(
        self,
    ):

        self.report = (

            HealthReport()

        )

        self.hostname = (

            socket.gethostname()

        )

        self.platform = (

            platform.platform()

        )

        self.started_at = (

            datetime.utcnow()

        )

        self.lock = (

            threading.Lock()

        )


    # ======================================================
    # Update Timestamp
    # ======================================================

    def touch(
        self,
    ) -> None:

        self.report.generated_at = (

            datetime.utcnow()

        )


    # ======================================================
    # Add Health Check
    # ======================================================

    def add_check(
        self,
        check: HealthCheck,
    ) -> None:

        with self.lock:

            self.report.add_check(

                check

            )

            self.touch()


    # ======================================================
    # Clear Checks
    # ======================================================

    def clear_checks(
        self,
    ) -> None:

        with self.lock:

            self.report.checks.clear()

            self.touch()


    # ======================================================
    # Update Overall Status
    # ======================================================

    def update_status(
        self,
    ) -> None:
        """
        Calculate overall health.
        """

        statuses = [

            check.status

            for check

            in self.report.checks

        ]

        if not statuses:

            self.report.overall_status = (

                HealthStatus.UNKNOWN

            )

            return

        if HealthStatus.CRITICAL in statuses:

            self.report.overall_status = (

                HealthStatus.CRITICAL

            )

        elif HealthStatus.WARNING in statuses:

            self.report.overall_status = (

                HealthStatus.WARNING

            )

        else:

            self.report.overall_status = (

                HealthStatus.HEALTHY

            )


    # ======================================================
    # Basic System Information
    # ======================================================

    def system_information(
        self,
    ) -> Dict[str, Any]:

        return {

            "hostname":

                self.hostname,

            "platform":

                self.platform,

            "python_version":

                platform.python_version(),

            "cpu_count":

                psutil.cpu_count(),

            "logical_cpu":

                psutil.cpu_count(

                    logical=True

                ),

            "boot_time":

                datetime.fromtimestamp(

                    psutil.boot_time()

                ).isoformat(),

        }


    # ======================================================
    # Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:

        self.update_status()

        return {

            "overall_status":

                self.report.overall_status.value,

            "checks":

                self.report.total_checks,

            "uptime_seconds":

                (

                    datetime.utcnow()

                    -

                    self.started_at

                ).total_seconds(),

            "generated_at":

                self.report.generated_at.isoformat(),

        }


    # ======================================================
    # Serialization
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        self.update_status()

        return {

            "system":

                self.system_information(),

            "statistics":

                self.statistics(),

            "report":

                self.report.to_dict(),

        }


    # ======================================================
    # JSON Export
    # ======================================================

    def to_json(
        self,
        indent: int = 4,
    ) -> str:

        return json.dumps(

            self.to_dict(),

            indent=indent,

            ensure_ascii=False,

        )

# ======================================================
# CPU Health
# ======================================================

def check_cpu(
    self,
    warning: float = 75.0,
    critical: float = 90.0,
) -> HealthCheck:
    """
    Check CPU utilization.
    """

    usage = psutil.cpu_percent(

        interval=1

    )

    status = HealthStatus.HEALTHY

    if usage >= critical:

        status = HealthStatus.CRITICAL

    elif usage >= warning:

        status = HealthStatus.WARNING

    check = HealthCheck(

        name="CPU",

        status=status,

        message=f"CPU Usage: {usage:.2f}%",

        value=usage,

        threshold=warning,

        metadata={

            "physical_cores":

                psutil.cpu_count(

                    logical=False

                ),

            "logical_cores":

                psutil.cpu_count(

                    logical=True

                ),

        },

    )

    self.add_check(

        check

    )

    return check


# ======================================================
# Memory Health
# ======================================================

def check_memory(
    self,
    warning: float = 80.0,
    critical: float = 95.0,
) -> HealthCheck:
    """
    Check RAM usage.
    """

    memory = psutil.virtual_memory()

    usage = memory.percent

    status = HealthStatus.HEALTHY

    if usage >= critical:

        status = HealthStatus.CRITICAL

    elif usage >= warning:

        status = HealthStatus.WARNING

    check = HealthCheck(

        name="Memory",

        status=status,

        message=f"RAM Usage: {usage:.2f}%",

        value=usage,

        threshold=warning,

        metadata={

            "total":

                memory.total,

            "available":

                memory.available,

            "used":

                memory.used,

            "free":

                memory.free,

        },

    )

    self.add_check(

        check

    )

    return check


# ======================================================
# Disk Health
# ======================================================

def check_disk(
    self,
    path: str = "/",
    warning: float = 80.0,
    critical: float = 95.0,
) -> HealthCheck:
    """
    Check disk usage.
    """

    usage = shutil.disk_usage(

        path

    )

    percent = round(

        (

            usage.used

            /

            usage.total

        )

        * 100,

        2,

    )

    status = HealthStatus.HEALTHY

    if percent >= critical:

        status = HealthStatus.CRITICAL

    elif percent >= warning:

        status = HealthStatus.WARNING

    check = HealthCheck(

        name="Disk",

        status=status,

        message=f"Disk Usage: {percent:.2f}%",

        value=percent,

        threshold=warning,

        metadata={

            "path":

                path,

            "total":

                usage.total,

            "used":

                usage.used,

            "free":

                usage.free,

        },

    )

    self.add_check(

        check

    )

    return check


# ======================================================
# Network Health
# ======================================================

def check_network(
    self,
) -> HealthCheck:
    """
    Check network interfaces.
    """

    interfaces = psutil.net_if_stats()

    active = [

        name

        for name, stat

        in interfaces.items()

        if stat.isup

    ]

    status = (

        HealthStatus.HEALTHY

        if active

        else

        HealthStatus.CRITICAL

    )

    check = HealthCheck(

        name="Network",

        status=status,

        message=f"{len(active)} active interface(s)",

        metadata={

            "interfaces":

                active,

        },

    )

    self.add_check(

        check

    )

    return check


# ======================================================
# GPU Health
# ======================================================

def check_gpu(
    self,
) -> HealthCheck:
    """
    Basic GPU availability check.
    """

    try:

        import torch

        available = (

            torch.cuda.is_available()

        )

        device_count = (

            torch.cuda.device_count()

            if available

            else 0

        )

        status = (

            HealthStatus.HEALTHY

            if available

            else HealthStatus.WARNING

        )

        check = HealthCheck(

            name="GPU",

            status=status,

            message=(

                "CUDA Available"

                if available

                else

                "GPU Not Available"

            ),

            metadata={

                "device_count":

                    device_count,

            },

        )

    except Exception:

        check = HealthCheck(

            name="GPU",

            status=HealthStatus.UNKNOWN,

            message="PyTorch unavailable.",

        )

    self.add_check(

        check

    )

    return check


# ======================================================
# Run All System Checks
# ======================================================

def check_system(
    self,
) -> None:
    """
    Execute all local system checks.
    """

    self.clear_checks()

    self.check_cpu()

    self.check_memory()

    self.check_disk()

    self.check_network()

    self.check_gpu()

    self.update_status()


# ======================================================
# Resource Usage
# ======================================================

def resource_usage(
    self,
) -> Dict[str, Any]:
    """
    Resource utilization snapshot.
    """

    disk = shutil.disk_usage(

        "/"

    )

    memory = psutil.virtual_memory()

    return {

        "cpu_percent":

            psutil.cpu_percent(),

        "memory_percent":

            memory.percent,

        "disk_percent":

            round(

                (

                    disk.used

                    /

                    disk.total

                )

                * 100,

                2,

            ),

        "process_count":

            len(

                psutil.pids()

            ),

        "thread_count":

            threading.active_count(),

    }


# ======================================================
# Advanced Statistics
# ======================================================

def advanced_statistics(
    self,
) -> Dict[str, Any]:
    """
    Advanced health statistics.
    """

    self.update_status()

    return {

        **self.statistics(),

        "healthy_checks":

            len(

                [

                    c

                    for c

                    in self.report.checks

                    if c.status

                    ==

                    HealthStatus.HEALTHY

                ]

            ),

        "warning_checks":

            len(

                [

                    c

                    for c

                    in self.report.checks

                    if c.status

                    ==

                    HealthStatus.WARNING

                ]

            ),

        "critical_checks":

            len(

                [

                    c

                    for c

                    in self.report.checks

                    if c.status

                    ==

                    HealthStatus.CRITICAL

                ]

            ),

        "resource_usage":

            self.resource_usage(),

    }

# ======================================================
# Redis Health
# ======================================================

def check_redis(
    self,
    client: Optional[Any] = None,
) -> HealthCheck:
    """
    Check Redis connection.
    """

    try:

        if client is None:

            raise RuntimeError(

                "Redis client not configured."

            )

        client.ping()

        check = HealthCheck(

            name="Redis",

            status=HealthStatus.HEALTHY,

            message="Redis connection successful.",

        )

    except Exception as exc:

        check = HealthCheck(

            name="Redis",

            status=HealthStatus.CRITICAL,

            message=str(exc),

        )

    self.add_check(

        check

    )

    return check


# ======================================================
# PostgreSQL Health
# ======================================================

def check_postgresql(
    self,
    connection: Optional[Any] = None,
) -> HealthCheck:
    """
    Check PostgreSQL connection.
    """

    try:

        if connection is None:

            raise RuntimeError(

                "Database connection unavailable."

            )

        cursor = connection.cursor()

        cursor.execute(

            "SELECT 1"

        )

        cursor.fetchone()

        cursor.close()

        check = HealthCheck(

            name="PostgreSQL",

            status=HealthStatus.HEALTHY,

            message="Database connection successful.",

        )

    except Exception as exc:

        check = HealthCheck(

            name="PostgreSQL",

            status=HealthStatus.CRITICAL,

            message=str(exc),

        )

    self.add_check(

        check

    )

    return check


# ======================================================
# Vector Database Health
# ======================================================

def check_vector_database(
    self,
    vector_store: Optional[Any] = None,
) -> HealthCheck:
    """
    Check vector database.
    """

    try:

        if vector_store is None:

            raise RuntimeError(

                "Vector store unavailable."

            )

        if hasattr(

            vector_store,

            "health",

        ):

            vector_store.health()

        check = HealthCheck(

            name="Vector Database",

            status=HealthStatus.HEALTHY,

            message="Vector database operational.",

        )

    except Exception as exc:

        check = HealthCheck(

            name="Vector Database",

            status=HealthStatus.CRITICAL,

            message=str(exc),

        )

    self.add_check(

        check

    )

    return check


# ======================================================
# LLM Provider Health
# ======================================================

def check_llm(
    self,
    provider: Optional[Any] = None,
) -> HealthCheck:
    """
    Check LLM provider.
    """

    try:

        if provider is None:

            raise RuntimeError(

                "LLM provider unavailable."

            )

        if hasattr(

            provider,

            "health",

        ):

            provider.health()

        check = HealthCheck(

            name="LLM Provider",

            status=HealthStatus.HEALTHY,

            message="LLM provider reachable.",

        )

    except Exception as exc:

        check = HealthCheck(

            name="LLM Provider",

            status=HealthStatus.CRITICAL,

            message=str(exc),

        )

    self.add_check(

        check

    )

    return check


# ======================================================
# API Health
# ======================================================

def check_api(
    self,
    url: str,
    timeout: int = 5,
) -> HealthCheck:
    """
    Check HTTP API endpoint.
    """

    try:

        import requests

        response = requests.get(

            url,

            timeout=timeout,

        )

        status = (

            HealthStatus.HEALTHY

            if response.ok

            else HealthStatus.WARNING

        )

        check = HealthCheck(

            name="API",

            status=status,

            message=f"HTTP {response.status_code}",

            value=response.status_code,

            metadata={

                "url":

                    url,

            },

        )

    except Exception as exc:

        check = HealthCheck(

            name="API",

            status=HealthStatus.CRITICAL,

            message=str(exc),

            metadata={

                "url":

                    url,

            },

        )

    self.add_check(

        check

    )

    return check


# ======================================================
# Diagnostics
# ======================================================

def diagnostics(
    self,
) -> Dict[str, Any]:
    """
    Complete diagnostics.
    """

    self.update_status()

    return {

        "system":

            self.system_information(),

        "statistics":

            self.advanced_statistics(),

        "checks":

            [

                check.to_dict()

                for check

                in self.report.checks

            ],

    }


# ======================================================
# Export Health Report
# ======================================================

def export(
    self,
) -> Dict[str, Any]:
    """
    Export complete health report.
    """

    return {

        "report":

            self.report.to_dict(),

        "diagnostics":

            self.diagnostics(),

        "system":

            self.system_information(),

    }


# ======================================================
# Summary
# ======================================================

def summary(
    self,
) -> Dict[str, Any]:
    """
    Human-readable summary.
    """

    self.update_status()

    return {

        "overall_status":

            self.report.overall_status.value,

        "total_checks":

            self.report.total_checks,

        "healthy":

            len(

                [

                    c

                    for c

                    in self.report.checks

                    if c.status

                    ==

                    HealthStatus.HEALTHY

                ]

            ),

        "warnings":

            len(

                [

                    c

                    for c

                    in self.report.checks

                    if c.status

                    ==

                    HealthStatus.WARNING

                ]

            ),

        "critical":

            len(

                [

                    c

                    for c

                    in self.report.checks

                    if c.status

                    ==

                    HealthStatus.CRITICAL

                ]

            ),

    }

# ======================================================
# Start Monitoring
# ======================================================

def start(
    self,
    interval: int = 30,
) -> None:
    """
    Start background monitoring.
    """

    if getattr(

        self,

        "_running",

        False,

    ):

        return

    self._running = True

    self._interval = interval

    self._thread = threading.Thread(

        target=self._monitor_loop,

        daemon=True,

        name="HealthMonitor",

    )

    self._thread.start()

    logger.info(

        "Health monitoring started."

    )


# ======================================================
# Stop Monitoring
# ======================================================

def stop(
    self,
) -> None:
    """
    Stop background monitoring.
    """

    self._running = False

    logger.info(

        "Health monitoring stopped."

    )


# ======================================================
# Background Loop
# ======================================================

def _monitor_loop(
    self,
) -> None:
    """
    Continuous monitoring loop.
    """

    while self._running:

        try:

            self.check_system()

        except Exception as exc:

            logger.exception(

                "Health monitoring failed: %s",

                exc,

            )

        time.sleep(

            self._interval

        )


# ======================================================
# Cleanup
# ======================================================

def cleanup(
    self,
) -> None:
    """
    Cleanup monitor.
    """

    self.stop()

    self.clear_checks()

    logger.info(

        "Health monitor cleaned."

    )


# ======================================================
# Refresh
# ======================================================

def refresh(
    self,
) -> None:
    """
    Refresh health report.
    """

    self.check_system()

    self.touch()

    logger.info(

        "Health report refreshed."

    )


# ======================================================
# Context Manager
# ======================================================

def __enter__(
    self,
):
    """
    Context manager entry.
    """

    self.start()

    return self


def __exit__(
    self,
    exc_type,
    exc_value,
    traceback,
):
    """
    Context manager exit.
    """

    self.cleanup()


# ======================================================
# String Representation
# ======================================================

def __repr__(
    self,
):
    """
    Developer representation.
    """

    return (

        "HealthMonitor("

        f"status='{self.report.overall_status.value}', "

        f"checks={self.report.total_checks}"

        ")"

    )


# ======================================================
# Human Readable
# ======================================================

def __str__(
    self,
):
    """
    Human-readable string.
    """

    return (

        f"Health: "

        f"{self.report.overall_status.value}"

    )


# ======================================================
# Boolean
# ======================================================

def __bool__(
    self,
):
    """
    Healthy monitor.
    """

    return (

        self.report.overall_status

        !=

        HealthStatus.CRITICAL

    )


# ======================================================
# Singleton
# ======================================================

_health_monitor: Optional[
    HealthMonitor
] = None


def get_health_monitor(
) -> HealthMonitor:
    """
    Singleton instance.
    """

    global _health_monitor

    if _health_monitor is None:

        _health_monitor = (

            HealthMonitor()

        )

    return _health_monitor


# ======================================================
# Reset Singleton
# ======================================================

def reset_health_monitor(
) -> None:
    """
    Reset singleton.
    """

    global _health_monitor

    if _health_monitor:

        _health_monitor.cleanup()

    _health_monitor = None


# ======================================================
# Factory Methods
# ======================================================

@classmethod
def development(
    cls,
) -> "HealthMonitor":
    """
    Development monitor.
    """

    return cls()


@classmethod
def production(
    cls,
) -> "HealthMonitor":
    """
    Production monitor.
    """

    return cls()


@classmethod
def testing(
    cls,
) -> "HealthMonitor":
    """
    Testing monitor.
    """

    return cls()


# ======================================================
# Convenience Properties
# ======================================================

@property
def is_running(
    self,
) -> bool:
    """
    Monitoring active.
    """

    return getattr(

        self,

        "_running",

        False,

    )


@property
def uptime_seconds(
    self,
) -> float:
    """
    Monitor uptime.
    """

    return (

        datetime.utcnow()

        -

        self.started_at

    ).total_seconds()


@property
def is_healthy(
    self,
) -> bool:
    """
    Overall healthy.
    """

    self.update_status()

    return (

        self.report.overall_status

        ==

        HealthStatus.HEALTHY

    )


# ======================================================
# Module Instance
# ======================================================

health_monitor = (

    get_health_monitor()

)
