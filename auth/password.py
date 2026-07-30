from __future__ import annotations

import hmac
import re
import secrets
import string
from dataclasses import dataclass
from typing import List

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash


# ==========================================================
# Exceptions
# ==========================================================

class PasswordError(Exception):
    """Base password exception."""


# ==========================================================
# Password Policy
# ==========================================================

@dataclass(slots=True)
class PasswordPolicy:
    min_length: int = 8
    max_length: int = 128

    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_special: bool = True

    special_characters: str = "!@#$%^&*()_-+=[]{}|\\:;\"'<>,.?/~`"


# ==========================================================
# Password Manager
# ==========================================================

class PasswordManager:

    def __init__(
        self,
        policy: PasswordPolicy | None = None,
    ) -> None:

        self.policy = policy or PasswordPolicy()

        self.hasher = PasswordHasher()

    # ------------------------------------------------------
    # Hash
    # ------------------------------------------------------

    def hash_password(
        self,
        password: str,
    ) -> str:

        self.validate_strength(password)

        return self.hasher.hash(password)

    # ------------------------------------------------------
    # Verify
    # ------------------------------------------------------

    def verify_password(
        self,
        password: str,
        hashed_password: str,
    ) -> bool:

        try:

            return self.hasher.verify(
                hashed_password,
                password,
            )

        except (
            VerifyMismatchError,
            InvalidHash,
        ):

            return False

    # ------------------------------------------------------
    # Rehash
    # ------------------------------------------------------

    def needs_rehash(
        self,
        hashed_password: str,
    ) -> bool:

        return self.hasher.check_needs_rehash(
            hashed_password
        )

    # ------------------------------------------------------
    # Strength Validation
    # ------------------------------------------------------

    def validate_strength(
        self,
        password: str,
    ) -> None:

        p = self.policy

        if len(password) < p.min_length:
            raise PasswordError(
                f"Password must contain at least "
                f"{p.min_length} characters."
            )

        if len(password) > p.max_length:
            raise PasswordError(
                "Password is too long."
            )

        if (
            p.require_uppercase
            and not re.search(r"[A-Z]", password)
        ):
            raise PasswordError(
                "Password requires an uppercase letter."
            )

        if (
            p.require_lowercase
            and not re.search(r"[a-z]", password)
        ):
            raise PasswordError(
                "Password requires a lowercase letter."
            )

        if (
            p.require_digit
            and not re.search(r"\d", password)
        ):
            raise PasswordError(
                "Password requires a digit."
            )

        if (
            p.require_special
            and not any(
                c in p.special_characters
                for c in password
            )
        ):
            raise PasswordError(
                "Password requires a special character."
            )

    # ------------------------------------------------------
    # Constant Time Comparison
    # ------------------------------------------------------

    @staticmethod
    def compare_constant_time(
        value1: str,
        value2: str,
    ) -> bool:

        return hmac.compare_digest(
            value1,
            value2,
        )

    # ------------------------------------------------------
    # Random Password
    # ------------------------------------------------------

    def generate_password(
        self,
        length: int = 16,
    ) -> str:

        alphabet = (
            string.ascii_letters
            + string.digits
            + self.policy.special_characters
        )

        while True:

            password = "".join(

                secrets.choice(alphabet)

                for _ in range(length)

            )

            try:

                self.validate_strength(password)

                return password

            except PasswordError:

                continue

    # ------------------------------------------------------
    # Password History
    # ------------------------------------------------------

    def in_password_history(
        self,
        password: str,
        history: List[str],
    ) -> bool:

        for hashed in history:

            if self.verify_password(
                password,
                hashed,
            ):
                return True

        return False

    # ------------------------------------------------------
    # Password Expiry Placeholder
    # ------------------------------------------------------

    @staticmethod
    def password_expired(
        days_since_change: int,
        expiry_days: int = 90,
    ) -> bool:

        return days_since_change >= expiry_days
