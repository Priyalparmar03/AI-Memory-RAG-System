from __future__ import annotations

from pydantic import (
    BaseModel,
    EmailStr,
    Field,
)


# ==========================================================
# Register
# ==========================================================

class RegisterRequest(BaseModel):

    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
    )

    email: EmailStr

    password: str = Field(
        ...,
        min_length=8,
    )


# ==========================================================
# Login
# ==========================================================

class LoginRequest(BaseModel):

    email: EmailStr

    password: str


# ==========================================================
# Token
# ==========================================================

class TokenResponse(BaseModel):

    access_token: str

    refresh_token: str

    token_type: str = "Bearer"


# ==========================================================
# User
# ==========================================================

class UserResponse(BaseModel):

    id: int

    name: str

    email: EmailStr

    role: str
