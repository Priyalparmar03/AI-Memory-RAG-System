"""
api/routes/admin.py
Administration Routes
Responsibilities
----------------
- Dashboard
- User Management
- Session Management
- Vector DB Management
- Cache Management
- System Maintenance
"""

from __future__ import annotations
from http import HTTPStatus
from flask import Blueprint, jsonify
from services.analytics_service import AnalyticsService
from services.document_service import DocumentService
from services.memory_service import MemoryService
from services.rag_service import RagService

admin_bp = Blueprint(
    "admin",
    __name__,
    url_prefix="/admin",
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


# Dashboard
@admin_bp.route("/dashboard", methods=["GET"])
def dashboard():

    dashboard = {
        "users": AnalyticsService.total_users(),
        "sessions": AnalyticsService.total_sessions(),
        "messages": AnalyticsService.total_messages(),
        "documents": DocumentService.total_documents(),
        "vector_chunks": RagService.total_chunks(),
    }

    return success(dashboard)
  
# Active Sessions
@admin_bp.route("/sessions", methods=["GET"])
def sessions():
    return success(
        MemoryService.active_sessions()
    )

# Clear All Sessions
@admin_bp.route("/sessions/clear", methods=["DELETE"])
def clear_sessions():
    MemoryService.clear_all_sessions()
    return success(
        message="All sessions cleared."
    )

# Reindex All Documents
@admin_bp.route("/reindex", methods=["POST"])
def reindex():
    RagService.reindex_all()
    return success(
        message="Re-index completed."
    )

# Clear Vector Database
@admin_bp.route("/vector-db/clear", methods=["DELETE"])
def clear_vector_db():
    RagService.clear_vector_database()
    return success(
        message="Vector database cleared."
    )

# System Cache
@admin_bp.route("/cache/clear", methods=["POST"])
def clear_cache():
    AnalyticsService.clear_cache()
    return success(
        message="Cache cleared."
    )

# Logs
@admin_bp.route("/logs", methods=["GET"])
def logs():
    return success(
        AnalyticsService.recent_logs()
    )

# Shutdown (Development Only)
@admin_bp.route("/shutdown", methods=["POST"])
def shutdown():
    return success(
        message="Shutdown endpoint disabled in production."
    )
