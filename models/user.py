from __future__ import annotations

import json
import logging
import re
import uuid

from dataclasses import (
    dataclass,
    field,
)

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class UserError(Exception):
    """
    User model exception.
    """
    pass


# ==========================================================
# User Role
# ==========================================================

class UserRole(str, Enum):

    ADMIN = "admin"

    USER = "user"

    EDITOR = "editor"

    VIEWER = "viewer"

    SYSTEM = "system"


# ==========================================================
# User Status
# ==========================================================

class UserStatus(str, Enum):

    ACTIVE = "active"

    INACTIVE = "inactive"

    BLOCKED = "blocked"

    PENDING = "pending"

    DELETED = "deleted"


# ==========================================================
# User Preferences
# ==========================================================

@dataclass(slots=True)
class UserPreferences:
    """
    User preferences.
    """

    theme: str = "light"

    language: str = "en"

    timezone: str = "UTC"

    embedding_model: str = ""

    llm_provider: str = ""

    notifications: bool = True

    auto_save: bool = True

    dark_mode: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {

            "theme":

                self.theme,

            "language":

                self.language,

            "timezone":

                self.timezone,

            "embedding_model":

                self.embedding_model,

            "llm_provider":

                self.llm_provider,

            "notifications":

                self.notifications,

            "auto_save":

                self.auto_save,

            "dark_mode":

                self.dark_mode,

            "metadata":

                self.metadata,

        }


# ==========================================================
# User Profile
# ==========================================================

@dataclass(slots=True)
class UserProfile:
    """
    User profile.
    """

    first_name: str = ""

    last_name: str = ""

    phone: str = ""

    organization: str = ""

    bio: str = ""

    avatar: str = ""

    location: str = ""

    website: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def full_name(
        self,
    ) -> str:

        return (

            f"{self.first_name} "

            f"{self.last_name}"

        ).strip()

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {

            "first_name":

                self.first_name,

            "last_name":

                self.last_name,

            "full_name":

                self.full_name,

            "phone":

                self.phone,

            "organization":

                self.organization,

            "bio":

                self.bio,

            "avatar":

                self.avatar,

            "location":

                self.location,

            "website":

                self.website,

            "metadata":

                self.metadata,

        }


# ==========================================================
# User
# ==========================================================

@dataclass(slots=True)
class User:
    """
    Production User model.
    """

    username: str

    email: str

    password_hash: str

    role: UserRole = UserRole.USER

    status: UserStatus = UserStatus.ACTIVE

    preferences: UserPreferences = field(
        default_factory=UserPreferences
    )

    profile: UserProfile = field(
        default_factory=UserProfile
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    id: str = field(
        default_factory=lambda: str(
            uuid.uuid4()
        )
    )

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    updated_at: datetime = field(
        default_factory=datetime.utcnow
    )

    last_login: Optional[
        datetime
    ] = None

    login_count: int = 0

    is_verified: bool = False

    is_superuser: bool = False

    # ======================================================
    # Initialization
    # ======================================================

    def __post_init__(
        self,
    ):

        self.validate_username()

        self.validate_email()

    # ======================================================
    # Username Validation
    # ======================================================

    def validate_username(
        self,
    ) -> None:
        """
        Validate username.
        """

        if len(

            self.username.strip()

        ) < 3:

            raise UserError(

                "Username must contain "

                "at least 3 characters."

            )

    # ======================================================
    # Email Validation
    # ======================================================

    def validate_email(
        self,
    ) -> None:
        """
        Validate email.
        """

        pattern = (

            r"^[A-Za-z0-9._%+-]+"

            r"@[A-Za-z0-9.-]+"

            r"\.[A-Za-z]{2,}$"

        )

        if not re.match(

            pattern,

            self.email,

        ):

            raise UserError(

                "Invalid email address."

            )

    # ======================================================
    # Update Timestamp
    # ======================================================

    def touch(
        self,
    ) -> None:
        """
        Update modification timestamp.
        """

        self.updated_at = (

            datetime.utcnow()

        )

    # ======================================================
    # Serialize
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize user.
        """

        return {

            "id":

                self.id,

            "username":

                self.username,

            "email":

                self.email,

            "password_hash":

                self.password_hash,

            "role":

                self.role.value,

            "status":

                self.status.value,

            "preferences":

                self.preferences.to_dict(),

            "profile":

                self.profile.to_dict(),

            "metadata":

                self.metadata,

            "created_at":

                self.created_at.isoformat(),

            "updated_at":

                self.updated_at.isoformat(),

            "last_login":

                self.last_login.isoformat()

                if self.last_login

                else None,

            "login_count":

                self.login_count,

            "is_verified":

                self.is_verified,

            "is_superuser":

                self.is_superuser,

        }

# ======================================================
# Update Password
# ======================================================

def update_password(
    self,
    password_hash: str,
) -> None:
    """
    Update password hash.
    """

    self.password_hash = password_hash

    self.touch()


# ======================================================
# Login
# ======================================================

def login(
    self,
) -> None:
    """
    Record successful login.
    """

    self.last_login = datetime.utcnow()

    self.login_count += 1

    self.status = UserStatus.ACTIVE

    self.touch()

    logger.info(

        f"{self.username} logged in."

    )


# ======================================================
# Logout
# ======================================================

def logout(
    self,
) -> None:
    """
    Logout user.
    """

    self.touch()

    logger.info(

        f"{self.username} logged out."

    )


# ======================================================
# Verify Account
# ======================================================

def verify(
    self,
) -> None:
    """
    Verify user.
    """

    self.is_verified = True

    self.touch()


# ======================================================
# Activate
# ======================================================

def activate(
    self,
) -> None:
    """
    Activate user.
    """

    self.status = UserStatus.ACTIVE

    self.touch()


# ======================================================
# Deactivate
# ======================================================

def deactivate(
    self,
) -> None:
    """
    Deactivate user.
    """

    self.status = UserStatus.INACTIVE

    self.touch()


# ======================================================
# Block User
# ======================================================

def block(
    self,
) -> None:
    """
    Block user.
    """

    self.status = UserStatus.BLOCKED

    self.touch()


# ======================================================
# Delete User
# ======================================================

def delete(
    self,
) -> None:
    """
    Soft delete user.
    """

    self.status = UserStatus.DELETED

    self.touch()


# ======================================================
# Role Check
# ======================================================

def has_role(
    self,
    role: UserRole,
) -> bool:
    """
    Check user role.
    """

    return self.role == role


# ======================================================
# Permission Check
# ======================================================

def has_permission(
    self,
    permission: str,
) -> bool:
    """
    Simple RBAC.
    """

    permissions = {

        UserRole.ADMIN: {

            "*",

        },

        UserRole.SYSTEM: {

            "*",

        },

        UserRole.EDITOR: {

            "read",

            "write",

            "update",

        },

        UserRole.USER: {

            "read",

            "write",

        },

        UserRole.VIEWER: {

            "read",

        },

    }

    allowed = permissions.get(

        self.role,

        set(),

    )

    return (

        "*"

        in allowed

        or

        permission

        in allowed

    )


# ======================================================
# Promote Role
# ======================================================

def promote(
    self,
    role: UserRole,
) -> None:
    """
    Promote user role.
    """

    self.role = role

    self.touch()


# ======================================================
# Update Preferences
# ======================================================

def update_preferences(
    self,
    **kwargs,
) -> None:
    """
    Update preferences.
    """

    for key, value in kwargs.items():

        if hasattr(

            self.preferences,

            key,

        ):

            setattr(

                self.preferences,

                key,

                value,

            )

    self.touch()


# ======================================================
# Update Profile
# ======================================================

def update_profile(
    self,
    **kwargs,
) -> None:
    """
    Update profile.
    """

    for key, value in kwargs.items():

        if hasattr(

            self.profile,

            key,

        ):

            setattr(

                self.profile,

                key,

                value,

            )

    self.touch()


# ======================================================
# Add Metadata
# ======================================================

def add_metadata(
    self,
    key: str,
    value: Any,
) -> None:
    """
    Add metadata entry.
    """

    self.metadata[key] = value

    self.touch()


# ======================================================
# Remove Metadata
# ======================================================

def remove_metadata(
    self,
    key: str,
) -> None:
    """
    Remove metadata entry.
    """

    self.metadata.pop(

        key,

        None,

    )

    self.touch()


# ======================================================
# User Statistics
# ======================================================

def statistics(
    self,
) -> Dict[str, Any]:
    """
    User statistics.
    """

    return {

        "id":

            self.id,

        "username":

            self.username,

        "role":

            self.role.value,

        "status":

            self.status.value,

        "verified":

            self.is_verified,

        "superuser":

            self.is_superuser,

        "login_count":

            self.login_count,

        "last_login":

            self.last_login.isoformat()

            if self.last_login

            else None,

        "member_since":

            self.created_at.isoformat(),

        "profile_completed":

            bool(

                self.profile.first_name

                or

                self.profile.last_name

            ),

        "metadata_entries":

            len(

                self.metadata

            ),

    }


# ======================================================
# Account Age
# ======================================================

@property
def account_age_days(
    self,
) -> int:
    """
    Account age in days.
    """

    return (

        datetime.utcnow()

        -

        self.created_at

    ).days
