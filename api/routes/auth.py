"""
api/routes/auth.py

Authentication routes.

Endpoints
---------
POST    /api/v1/auth/register
POST    /api/v1/auth/login
POST    /api/v1/auth/refresh
POST    /api/v1/auth/logout
GET     /api/v1/auth/me

"""

from http import HTTPStatus

from flask import Blueprint, jsonify, request

# Later we will create these modules.
# from services.auth_service import AuthService
# from auth.jwt_handler import jwt_required, get_current_user

auth_bp = Blueprint(
    "auth",
    __name__,
    url_prefix="/auth",
)


# Register
@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Register a new user.

    Expected JSON
    -------------
    {
        "name":"Priyal",
        "email":"abc@gmail.com",
        "password":"StrongPassword123"
    }
    """

    data = request.get_json(silent=True)

    if not data:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "JSON body is required."
                }
            ),
            HTTPStatus.BAD_REQUEST,
        )

    required_fields = [
        "name",
        "email",
        "password",
    ]

    missing = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Missing fields: {', '.join(missing)}"
                }
            ),
            HTTPStatus.BAD_REQUEST,
        )

    # user = AuthService.register(data)

    return (
        jsonify(
            {
                "success": True,
                "message": "User registered successfully."
            }
        ),
        HTTPStatus.CREATED,
    )


# Login
@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Login endpoint.

    JSON
    ----

    {
        "email":"abc@gmail.com",
        "password":"password"
    }
    """

    data = request.get_json(silent=True)

    if not data:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "JSON body is required."
                }
            ),
            HTTPStatus.BAD_REQUEST,
        )

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Email and password are required."
                }
            ),
            HTTPStatus.BAD_REQUEST,
        )

    # tokens = AuthService.login(email,password)

    return (
        jsonify(
            {
                "success": True,
                "message": "Login successful.",
                "access_token": "<JWT_ACCESS_TOKEN>",
                "refresh_token": "<JWT_REFRESH_TOKEN>"
            }
        ),
        HTTPStatus.OK,
    )


# Refresh Token
@auth_bp.route("/refresh", methods=["POST"])
def refresh():
    """
    Generate a new access token.
    """

    return (
        jsonify(
            {
                "success": True,
                "message": "Access token refreshed.",
                "access_token": "<NEW_ACCESS_TOKEN>"
            }
        ),
        HTTPStatus.OK,
    )


# Logout
@auth_bp.route("/logout", methods=["POST"])
def logout():
    """
    Logout current user.

    Future:
        JWT blacklist
    """

    return (
        jsonify(
            {
                "success": True,
                "message": "Logged out successfully."
            }
        ),
        HTTPStatus.OK,
    )

# Current User
@auth_bp.route("/me", methods=["GET"])
def me():
    """
    Return current logged in user.

    Future:
        JWT Protected
    """

    return (
        jsonify(
            {
                "success": True,
                "data": {
                    "id": 1,
                    "name": "Priyal Parmar",
                    "email": "example@gmail.com"
                }
            }
        ),
        HTTPStatus.OK,
    )