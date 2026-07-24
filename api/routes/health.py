"""
api/routes/health.py

Health Check Routes

Author: Priyal Parmar
Project: AI Memory RAG System
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime
from http import HTTPStatus

from flask import Blueprint, jsonify

from services.analytics_service import AnalyticsService

health_bp = Blueprint(
    "health",
    __name__,
    url_prefix="/health",
)


def success(data=None, message="Success", status=HTTPStatus.OK):
    return (
        jsonify(
            {
                "success": True,
                "message": message,
                "data": data,
            }
        ),
        status,
    )


# ==========================================================
# Basic Health Check
# ==========================================================

@health_bp.route(
    "",
    methods=["GET"],
)
def health():

    return success(
        data={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


# ==========================================================
# Readiness Probe
# ==========================================================

@health_bp.route(
    "/ready",
    methods=["GET"],
)
def readiness():

    database = True
    vector_db = True
    llm = True

    return success(
        data={
            "database": database,
            "vector_database": vector_db,
            "llm": llm,
            "ready": database and vector_db and llm,
        }
    )


# ==========================================================
# Liveness Probe
# ==========================================================

@health_bp.route(
    "/live",
    methods=["GET"],
)
def liveness():

    return success(
        data={
            "alive": True
        }
    )


# ==========================================================
# System Information
# ==========================================================

@health_bp.route(
    "/system",
    methods=["GET"],
)
def system_info():

    return success(
        data={
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
    )


# ==========================================================
# Metrics
# ==========================================================

@health_bp.route(
    "/metrics",
    methods=["GET"],
)
def metrics():

    metrics = AnalyticsService.system_metrics()

    return success(metrics)


# ==========================================================
# Version
# ==========================================================

@health_bp.route(
    "/version",
    methods=["GET"],
)
def version():

    return success(
        {
            "application": "AI Memory RAG System",
            "version": "1.0.0",
            "api": "v1",
        }
    )