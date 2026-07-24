"""
JWT Authentication Middleware
"""

from functools import wraps

from flask import current_app, g, jsonify, request
import jwt


def jwt_required(func):
    """
    Protect routes using JWT authentication.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        auth = request.headers.get("Authorization")

        if not auth:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Authorization header missing.",
                    }
                ),
                401,
            )

        if not auth.startswith("Bearer "):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Invalid authorization header.",
                    }
                ),
                401,
            )

        token = auth.split(" ")[1]

        try:

            payload = jwt.decode(
                token,
                current_app.config["JWT_SECRET"],
                algorithms=["HS256"],
            )

            g.current_user = payload

        except jwt.ExpiredSignatureError:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Token expired.",
                    }
                ),
                401,
            )

        except jwt.InvalidTokenError:

            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Invalid token.",
                    }
                ),
                401,
            )

        return func(*args, **kwargs)

    return wrapper


def admin_required(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        if g.current_user.get("role") != "admin":

            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Admin access required.",
                    }
                ),
                403,
            )

        return func(*args, **kwargs)

    return wrapper
