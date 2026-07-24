"""
api/routes/chat.py
Handles all chatbot-related endpoints.
Features
--------
- Chat with Memory
- Session Management
- Sliding Window Memory
- RAG Integration
- Streaming Response
- Chat Export
- Clear History
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, time
from http import HTTPStatus
from typing import Any

from flask import (
    Blueprint,
    Response,
    jsonify,
    request,
    session,
)
import time
from flask import current_app
# Services
# These will be implemented later.
from flask import stream_with_context
import json
from services.chat_service import ChatService
from services.memory_service import MemoryService
from services.rag_service import RagService
from services.prompt_service import PromptService
from services.analytics_service import AnalyticsService

# Logger
logger = logging.getLogger(__name__)

# Blueprint
chat_bp = Blueprint(
    "chat",
    __name__,
    url_prefix="/chat",
)

# Configuration
MAX_HISTORY = 20

DEFAULT_MODEL = "gemini-2.5-flash"

ENABLE_RAG = True

ENABLE_STREAMING = True

# Helper Functions
def success_response(
    data: Any,
    message: str = "Success",
    status: int = HTTPStatus.OK,
):
    """
    Standard success response.
    """

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


def error_response(
    message: str,
    status: int = HTTPStatus.BAD_REQUEST,
):
    """
    Standard error response.
    """

    return (
        jsonify(
            {
                "success": False,
                "message": message,
            }
        ),
        status,
    )

# Session Helpers
def get_session_id() -> str:
    """
    Returns a unique session id.
    """

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    return session["session_id"]


def initialize_chat():
    """
    Initialize chat memory.
    """

    if "history" not in session:
        session["history"] = []

    if "created_at" not in session:
        session["created_at"] = datetime.utcnow().isoformat()


def get_history():
    """
    Returns current conversation.
    """

    initialize_chat()

    return session["history"]


def save_history(history):
    """
    Save updated history.
    """

    session["history"] = history
    session.modified = True

# Memory Helpers
def append_message(
    role: str,
    content: str,
):
    """
    Append message to history.
    """

    history = get_history()

    history.append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    save_history(history)


def sliding_window():
    """
    Keep only latest messages.
    """

    history = get_history()

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
        save_history(history)

# Validation
def validate_request(payload):
    """
    Validate chat request.
    """

    if payload is None:
        return False, "JSON body required."

    if "message" not in payload:
        return False, "message field required."

    message = payload["message"]

    if not isinstance(message, str):
        return False, "message must be string."

    if len(message.strip()) == 0:
        return False, "message cannot be empty."

    return True, ""

# Prompt Builder
def build_prompt(
    user_message: str,
):
    """
    Build prompt with memory
    and optional RAG context.
    """

    history = get_history()

    context = ""

    if ENABLE_RAG:

        context = RagService.retrieve_context(
            query=user_message,
            top_k=5,
        )

    prompt = PromptService.build_prompt(
        history=history,
        user_message=user_message,
        context=context,
    )

    return prompt

# Analytics
def log_request(
    model: str,
    latency: float,
):
    """
    Store analytics.
    """

    AnalyticsService.log_request(
        session_id=get_session_id(),
        model=model,
        latency=latency,
    )

# Health Check
@chat_bp.route(
    "/ping",
    methods=["GET"],
)
def ping():
    """
    Chat API health.
    """

    return success_response(
        {
            "status": "online",
            "service": "chat",
            "session": get_session_id(),
        }
    )

# POST /chat
@chat_bp.route(
    "/",
    methods=["POST"],
)
def chat():
    """
    Main Chat Endpoint.

    Flow
    ----
    User
        ↓
    Validate Request
        ↓
    Session Memory
        ↓
    Retrieve RAG Context
        ↓
    Build Prompt
        ↓
    LLM
        ↓
    Save Response
        ↓
    Return JSON
    """

    start_time = time.time()

    try:

        payload = request.get_json(silent=True)

        valid, message = validate_request(payload)

        if not valid:
            return error_response(message)

        user_message = payload["message"].strip()

        model = payload.get(
            "model",
            DEFAULT_MODEL,
        )

        temperature = payload.get(
            "temperature",
            0.3,
        )

        session_id = get_session_id()

        logger.info(
            "New Chat Request | Session=%s",
            session_id,
        )

        # -----------------------------------
        # Save User Message
        # -----------------------------------

        append_message(
            role="user",
            content=user_message,
        )

        sliding_window()

        # -----------------------------------
        # Build Prompt
        # -----------------------------------

        prompt = build_prompt(user_message)

        logger.info(
            "Prompt Created Successfully."
        )

        # -----------------------------------
        # Call LLM
        # -----------------------------------

        ai_response = ChatService.generate_response(
            prompt=prompt,
            model=model,
            temperature=temperature,
        )

        # -----------------------------------
        # Save AI Response
        # -----------------------------------

        append_message(
            role="assistant",
            content=ai_response,
        )

        sliding_window()

        history = get_history()

        latency = round(
            time.time() - start_time,
            3,
        )

        log_request(
            model=model,
            latency=latency,
        )

        logger.info(
            "Chat Completed | %.2fs",
            latency,
        )

        return success_response(
            data={
                "session_id": session_id,
                "response": ai_response,
                "history": history,
                "latency": latency,
                "model": model,
            },
            message="Chat completed successfully.",
        )

    except Exception as exc:

        logger.exception(exc)

        return error_response(
            "Internal Server Error.",
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )

# POST /chat/stream
@chat_bp.route(
    "/stream",
    methods=["POST"],
)
def stream_chat():
    """
    Streaming chat endpoint using
    Server Sent Events (SSE).
    """

    payload = request.get_json(silent=True)

    valid, message = validate_request(payload)

    if not valid:
        return error_response(message)

    user_message = payload["message"].strip()

    model = payload.get(
        "model",
        DEFAULT_MODEL,
    )

    append_message(
        role="user",
        content=user_message,
    )

    sliding_window()

    history = get_history()

    session_id = get_session_id()

    def generate():

        try:

            # -----------------------------------------
            # Retrieve RAG Context
            # -----------------------------------------

            context = ""

            sources = []

            if ENABLE_RAG:

                rag_result = RagService.retrieve(
                    query=user_message,
                    top_k=5,
                )

                context = rag_result["context"]

                sources = rag_result["sources"]

            # -----------------------------------------
            # Prompt Builder
            # -----------------------------------------

            prompt = PromptService.build_prompt(
                history=history,
                user_message=user_message,
                context=context,
            )

            # -----------------------------------------
            # Stream Response
            # -----------------------------------------

            complete_answer = ""

            for token in ChatService.stream_response(
                prompt=prompt,
                model=model,
            ):

                complete_answer += token

                yield (
                    f"data:{json.dumps({'token': token})}\n\n"
                )

            append_message(
                role="assistant",
                content=complete_answer,
            )

            yield (
                f"data:{json.dumps({'done':True})}\n\n"
            )

            yield (
                f"data:{json.dumps({'sources':sources})}\n\n"
            )

        except Exception as exc:

            logger.exception(exc)

            yield (
                f"data:{json.dumps({'error':str(exc)})}\n\n"
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
    )
# POST /chat/clear
@chat_bp.route(
    "/clear",
    methods=["POST"],
)
def clear_chat():

    session.pop("history", None)

    session.modified = True

    return success_response(
        data={},
        message="Conversation cleared.",
    )
# GET /chat/history
@chat_bp.route(
    "/history",
    methods=["GET"],
)
def history():

    return success_response(
        get_history()
    )

# DELETE /chat/history
@chat_bp.route(
    "/history",
    methods=["DELETE"],
)
def delete_history():

    session["history"] = []

    session.modified = True

    return success_response(
        [],
        message="History deleted.",
    )
# GET /chat/export
@chat_bp.route(
    "/export",
    methods=["GET"],
)
def export_chat():

    history = get_history()

    export_text = ""

    for message in history:

        export_text += (
            f"{message['role'].upper()}:\n"
            f"{message['content']}\n\n"
        )

    return Response(
        export_text,
        mimetype="text/plain",
        headers={
            "Content-Disposition":
            "attachment; filename=chat.txt"
        },
    )