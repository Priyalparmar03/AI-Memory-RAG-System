"""
api/routes/documents.py

Document Management API

Responsibilities
----------------
- Upload documents
- Parse files
- Create embeddings
- Store vectors
- Search documents
- Delete documents
- Reindex documents
"""

from __future__ import annotations

import logging
from http import HTTPStatus

from flask import (
    Blueprint,
    jsonify,
    request,
)

from werkzeug.utils import secure_filename

from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from services.rag_service import RagService

logger = logging.getLogger(__name__)

documents_bp = Blueprint(
    "documents",
    __name__,
    url_prefix="/documents",
)

ALLOWED_EXTENSIONS = {
    "pdf",
    "docx",
    "txt",
    "md",
    "csv",
}


def success(data=None, message="Success", status=200):
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


def error(message, status=400):
    return (
        jsonify(
            {
                "success": False,
                "message": message,
            }
        ),
        status,
    )


def allowed(filename):

    if "." not in filename:
        return False

    ext = filename.rsplit(".", 1)[1].lower()

    return ext in ALLOWED_EXTENSIONS

# Upload Document
@documents_bp.route(
    "/upload",
    methods=["POST"],
)
def upload_document():

    if "file" not in request.files:
        return error("No file uploaded.")

    file = request.files["file"]

    if file.filename == "":
        return error("Filename missing.")

    if not allowed(file.filename):
        return error("Unsupported file type.")

    filename = secure_filename(file.filename)

    logger.info("Uploading %s", filename)

    # ---------------------------------------
    # Save file
    # ---------------------------------------

    document = DocumentService.save_document(
        file=file,
        filename=filename,
    )

    # ---------------------------------------
    # Parse
    # ---------------------------------------

    text = DocumentService.extract_text(
        document.path,
    )

    # ---------------------------------------
    # Chunk
    # ---------------------------------------

    chunks = DocumentService.chunk_document(
        text=text,
    )

    # ---------------------------------------
    # Embeddings
    # ---------------------------------------

    embeddings = EmbeddingService.embed_documents(
        chunks,
    )

    # ---------------------------------------
    # Store Vector
    # ---------------------------------------

    RagService.index_document(
        document_id=document.id,
        chunks=chunks,
        embeddings=embeddings,
    )

    return success(
        data={
            "document_id": document.id,
            "filename": filename,
            "chunks": len(chunks),
        },
        message="Document indexed successfully.",
        status=HTTPStatus.CREATED,
    )

# List Documents
@documents_bp.route(
    "",
    methods=["GET"],
)
def list_documents():

    documents = DocumentService.list_documents()

    return success(documents)

# Get Document Metadata
@documents_bp.route(
    "/<int:document_id>",
    methods=["GET"],
)
def get_document(document_id):

    document = DocumentService.get_document(
        document_id,
    )

    if not document:
        return error(
            "Document not found.",
            HTTPStatus.NOT_FOUND,
        )

    return success(document)

# Delete Document
@documents_bp.route(
    "/<int:document_id>",
    methods=["DELETE"],
)
def delete_document(document_id):

    RagService.delete_document(
        document_id,
    )

    DocumentService.delete_document(
        document_id,
    )

    return success(
        message="Document deleted.",
    )

# Re-index
@documents_bp.route(
    "/<int:document_id>/reindex",
    methods=["POST"],
)
def reindex(document_id):

    RagService.reindex_document(
        document_id,
    )

    return success(
        message="Re-index completed.",
    )

# Semantic Search
@documents_bp.route(
    "/search",
    methods=["POST"],
)
def semantic_search():

    payload = request.get_json()

    query = payload.get("query")

    top_k = payload.get(
        "top_k",
        5,
    )

    if not query:
        return error("Query required.")

    results = RagService.search(
        query=query,
        top_k=top_k,
    )

    return success(results)

# Statistics
@documents_bp.route(
    "/stats",
    methods=["GET"],
)
def stats():

    statistics = DocumentService.statistics()

    return success(statistics)