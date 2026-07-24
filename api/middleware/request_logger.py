"""
Request Logging Middleware
"""

import logging
import time
import uuid

from flask import g, request

logger = logging.getLogger(__name__)


def before_request():

    g.request_id = str(uuid.uuid4())

    g.start_time = time.time()

    logger.info(
        "[%s] %s %s",
        g.request_id,
        request.method,
        request.path,
    )


def after_request(response):

    elapsed = round(
        (time.time() - g.start_time) * 1000,
        2,
    )

    response.headers["X-Request-ID"] = g.request_id

    response.headers["X-Response-Time"] = f"{elapsed} ms"

    logger.info(
        "[%s] %s %s %s %.2fms",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        elapsed,
    )

    return response


def register_request_logger(app):

    app.before_request(before_request)

    app.after_request(after_request)
