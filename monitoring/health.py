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
