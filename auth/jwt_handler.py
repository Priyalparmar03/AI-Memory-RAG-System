from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError


# ==========================================================
# Exceptions
# ==========================================================

class JWTError(Exception):
    """Base JWT exception."""


# ==========================================================
# Configuration
# ==========================================================

@dataclass(slots=True)
class JWTConfig:
    secret_key: str
    algorithm: str = "HS256"

    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    issuer: str = "AI-Memory-RAG"
    audience: str = "AI-Memory-RAG-Users"


# ==========================================================
# JWT Handler
# ==========================================================

class JWTHandler:

    def __init__(
        self,
        config: JWTConfig,
    ) -> None:

        self.config = config

    # ------------------------------------------------------
    # Access Token
    # ------------------------------------------------------

    def create_access_token(
        self,
        user_id: str,
        email: str,
        role: str,
        session_id: str,
        permissions: Optional[list[str]] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:

        expire = datetime.now(timezone.utc) + timedelta(
            minutes=self.config.access_token_expire_minutes
        )

        payload = {
            "sub": user_id,
            "email": email,
            "role": role,
            "permissions": permissions or [],
            "session_id": session_id,
            "type": "access",
            "jti": self.generate_jti(),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "iat": datetime.now(timezone.utc),
            "exp": expire,
        }

        if additional_claims:
            payload.update(additional_claims)

        return jwt.encode(
            payload,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

    # ------------------------------------------------------
    # Refresh Token
    # ------------------------------------------------------

    def create_refresh_token(
        self,
        user_id: str,
        session_id: str,
    ) -> str:

        expire = datetime.now(timezone.utc) + timedelta(
            days=self.config.refresh_token_expire_days
        )

        payload = {
            "sub": user_id,
            "session_id": session_id,
            "type": "refresh",
            "jti": self.generate_jti(),
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "iat": datetime.now(timezone.utc),
            "exp": expire,
        }

        return jwt.encode(
            payload,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

    # ------------------------------------------------------
    # Decode
    # ------------------------------------------------------

    def decode_token(
        self,
        token: str,
    ) -> Dict[str, Any]:

        try:

            return jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                issuer=self.config.issuer,
                audience=self.config.audience,
            )

        except ExpiredSignatureError as exc:

            raise JWTError("Token has expired.") from exc

        except InvalidTokenError as exc:

            raise JWTError("Invalid token.") from exc

    # ------------------------------------------------------
    # Verify
    # ------------------------------------------------------

    def verify_token(
        self,
        token: str,
        token_type: Optional[str] = None,
    ) -> Dict[str, Any]:

        payload = self.decode_token(token)

        if token_type:

            if payload.get("type") != token_type:

                raise JWTError(
                    f"Expected {token_type} token."
                )

        return payload

    # ------------------------------------------------------
    # Refresh Access Token
    # ------------------------------------------------------

    def refresh_access_token(
        self,
        refresh_token: str,
        email: str,
        role: str,
        permissions: Optional[list[str]] = None,
    ) -> str:

        payload = self.verify_token(
            refresh_token,
            "refresh",
        )

        return self.create_access_token(

            user_id=payload["sub"],

            email=email,

            role=role,

            session_id=payload["session_id"],

            permissions=permissions,

        )

    # ------------------------------------------------------
    # Claims
    # ------------------------------------------------------

    def extract_claims(
        self,
        token: str,
    ) -> Dict[str, Any]:

        return self.decode_token(token)

    # ------------------------------------------------------
    # Remaining Lifetime
    # ------------------------------------------------------

    def token_remaining_time(
        self,
        token: str,
    ) -> int:

        payload = self.decode_token(token)

        exp = datetime.fromtimestamp(
            payload["exp"],
            tz=timezone.utc,
        )

        remaining = exp - datetime.now(timezone.utc)

        return max(
            0,
            int(remaining.total_seconds()),
        )

    # ------------------------------------------------------
    # JTI
    # ------------------------------------------------------

    @staticmethod
    def generate_jti() -> str:

        return uuid.uuid4().hex
