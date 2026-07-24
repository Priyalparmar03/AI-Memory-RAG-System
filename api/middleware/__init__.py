"""
Middleware Registration
"""

from .rate_limit import init_rate_limiter
from .request_logger import register_request_logger


def register_middlewares(app):

    register_request_logger(app)

    init_rate_limiter(app)
