"""
api/__init__.py

Initializes the API package and exposes the API Blueprint.

Author: Priyal Parmar
Project: AI Memory RAG System
"""

from flask import Blueprint

# Main API blueprint
api_bp = Blueprint(
    "api",
    __name__,
    url_prefix="/api/v1"
)

# Import route modules so they register with the blueprint.
# These imports are intentionally placed at the bottom to avoid
# circular import issues.
from api.routes import (  # noqa: E402,F401
    admin,
    analytics,
    auth,
    chat,
    documents,
    health,
    history,
)