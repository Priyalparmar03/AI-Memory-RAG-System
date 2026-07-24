"""
Validation Middleware
"""

from functools import wraps

from flask import jsonify, request


def require_json(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        if not request.is_json:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Request must be JSON.",
                    }
                ),
                415,
            )

        return func(*args, **kwargs)

    return wrapper


def require_fields(*fields):

    """
    Validate required JSON fields.
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            data = request.get_json()

            missing = []

            for field in fields:

                if field not in data:

                    missing.append(field)

            if missing:

                return (
                    jsonify(
                        {
                            "success": False,
                            "missing_fields": missing,
                        }
                    ),
                    400,
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
