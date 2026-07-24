"""
api/routes/history.py

Conversation History Routes

Author: Priyal Parmar
Project: AI Memory RAG System
"""

from __future__ import annotations

from http import HTTPStatus

from flask import (
    Blueprint,
    jsonify,
    request,
)

from services.memory_service import MemoryService

history_bp = Blueprint(
    "history",
    __name__,
    url_prefix="/history",
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


def error(message, status=HTTPStatus.BAD_REQUEST):
    return (
        jsonify(
            {
                "success": False,
                "message": message,
            }
        ),
        status,
    )


# ==========================================================
# Get Current Session History
# ==========================================================

@history_bp.route(
    "",
    methods=["GET"],
)
def get_history():

    session_id = request.args.get("session_id")

    history = MemoryService.get_history(session_id)

    return success(history)


# ==========================================================
# Delete History
# ==========================================================

@history_bp.route(
    "/clear",
    methods=["DELETE"],
)
def clear_history():

    session_id = request.args.get("session_id")

    MemoryService.clear_history(session_id)

    return success(
        message="Conversation deleted."
    )


# ==========================================================
# Conversation Summary
# ==========================================================

@history_bp.route(
    "/summary",
    methods=["GET"],
)
def summary():

    session_id = request.args.get("session_id")

    summary = MemoryService.summarize(
        session_id=session_id,
    )

    return success(summary)


# ==========================================================
# Export Conversation
# ==========================================================

@history_bp.route(
    "/export",
    methods=["GET"],
)
def export():

    session_id = request.args.get("session_id")

    file_format = request.args.get(
        "format",
        "txt",
    )

    file = MemoryService.export(
        session_id=session_id,
        file_format=file_format,
    )

    return success(file)


# ==========================================================
# Search History
# ==========================================================

@history_bp.route(
    "/search",
    methods=["POST"],
)
def search():

    payload = request.get_json()

    query = payload.get("query")

    session_id = payload.get("session_id")

    if not query:
        return error("Query required.")

    results = MemoryService.search(
        session_id=session_id,
        query=query,
    )

    return success(results)


# ==========================================================
# Delete Single Message
# ==========================================================

@history_bp.route(
    "/message/<message_id>",
    methods=["DELETE"],
)
def delete_message(message_id):

    MemoryService.delete_message(
        message_id,
    )

    return success(
        message="Message deleted."
    )


# ==========================================================
# History Statistics
# ==========================================================

@history_bp.route(
    "/stats",
    methods=["GET"],
)
def stats():

    session_id = request.args.get("session_id")

    statistics = MemoryService.statistics(
        session_id,
    )

    return success(statistics)