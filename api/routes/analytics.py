"""
api/routes/analytics.py

Analytics Routes

Responsibilities
----------------
- API Usage
- Token Usage
- Cost Tracking
- Latency
- Model Statistics
- User Analytics
"""

from __future__ import annotations
from http import HTTPStatus
from flask import Blueprint, jsonify, request
from services.analytics_service import AnalyticsService

analytics_bp = Blueprint(
    "analytics",
    __name__,
    url_prefix="/analytics",
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


# Overall Dashboard
@analytics_bp.route("/dashboard", methods=["GET"])
def dashboard():

    return success(
        AnalyticsService.dashboard()
    )

# Token Usage
@analytics_bp.route("/tokens", methods=["GET"])
def token_usage():
    return success(
        AnalyticsService.token_usage()
    )

# Cost Analysis
@analytics_bp.route("/cost", methods=["GET"])
def cost():
    return success(
        AnalyticsService.cost_analysis()
    )

# Latency
@analytics_bp.route("/latency", methods=["GET"])
def latency():
    return success(
        AnalyticsService.latency()
    )

# Model Usage
@analytics_bp.route("/models", methods=["GET"])
def models():
    return success(
        AnalyticsService.model_statistics()
    )

# Daily Usage
@analytics_bp.route("/daily", methods=["GET"])
def daily():

    return success(
        AnalyticsService.daily_usage()
    )

# User Analytics
@analytics_bp.route("/users", methods=["GET"])
def users():
    return success(
        AnalyticsService.user_statistics()
    )

# Conversation Analytics
@analytics_bp.route("/conversations", methods=["GET"])
def conversations():
    return success(
        AnalyticsService.conversation_statistics()
    )

# Search Analytics
@analytics_bp.route("/search", methods=["POST"])
def search():
    payload = request.get_json()
    start = payload.get("start_date")
    end = payload.get("end_date")
    return success(
        AnalyticsService.search(
            start=start,
            end=end,
        )
    )

# Export Analytics
@analytics_bp.route("/export", methods=["GET"])
def export():

    return success(
        AnalyticsService.export()
    )
