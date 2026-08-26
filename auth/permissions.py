from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set


class PermissionError(Exception):
    """Raised when permission checks fail."""

# Permission Names
class Permissions:

    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"

    CHAT = "chat"
    RAG_QUERY = "rag:query"
    EMBEDDING_CREATE = "embedding:create"
    ANALYTICS_READ = "analytics:read"
    ADMIN = "admin"


# Role Definition
@dataclass(slots=True)
class Role:
    name: str
    permissions: Set[str] = field(default_factory=set)

# RBAC Manager
class PermissionManager:

    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self._initialize_roles()

    # Default Roles
    def _initialize_roles(self):
        self.roles["viewer"] = Role(
            name="viewer",
            permissions={
                Permissions.DOCUMENT_READ,
                Permissions.MEMORY_READ,
                Permissions.CHAT,
                Permissions.RAG_QUERY,

            },

        )

        self.roles["user"] = Role(
            name="user",
            permissions={
                Permissions.DOCUMENT_READ
                Permissions.DOCUMENT_WRITE,
                Permissions.MEMORY_READ,
                Permissions.MEMORY_WRITE,
                Permissions.CHAT,
                Permissions.RAG_QUERY,

            },

        )

        self.roles["editor"] = Role(

            name="editor",

            permissions={

                Permissions.DOCUMENT_READ,

                Permissions.DOCUMENT_WRITE,

                Permissions.DOCUMENT_DELETE,

                Permissions.MEMORY_READ,

                Permissions.MEMORY_WRITE,

                Permissions.CHAT,

                Permissions.RAG_QUERY,

                Permissions.EMBEDDING_CREATE,

            },

        )

        self.roles["admin"] = Role(

            name="admin",

            permissions={

                Permissions.ADMIN,

                Permissions.USERS_READ,

                Permissions.USERS_WRITE,

                Permissions.USERS_DELETE,

                Permissions.DOCUMENT_READ,

                Permissions.DOCUMENT_WRITE,

                Permissions.DOCUMENT_DELETE,

                Permissions.MEMORY_READ,

                Permissions.MEMORY_WRITE,

                Permissions.CHAT,

                Permissions.RAG_QUERY,

                Permissions.EMBEDDING_CREATE,

                Permissions.ANALYTICS_READ,

            },

        )

    # ------------------------------------------------------
    # Role Management
    # ------------------------------------------------------

    def add_role(
        self,
        role: Role,
    ):

        self.roles[role.name] = role

    def remove_role(
        self,
        role_name: str,
    ):

        self.roles.pop(role_name, None)

    def get_role(
        self,
        role_name: str,
    ) -> Role | None:

        return self.roles.get(role_name)

    # ------------------------------------------------------
    # Permission Checks
    # ------------------------------------------------------

    def has_permission(
        self,
        role_name: str,
        permission: str,
    ) -> bool:

        role = self.roles.get(role_name)

        if role is None:

            return False

        return (

            permission in role.permissions

            or Permissions.ADMIN in role.permissions

        )

    def require_permission(
        self,
        role_name: str,
        permission: str,
    ):

        if not self.has_permission(

            role_name,

            permission,

        ):

            raise PermissionError(

                f"'{role_name}' lacks '{permission}' permission."

            )

    # ------------------------------------------------------
    # Bulk Operations
    # ------------------------------------------------------

    def permissions_for_role(
        self,
        role_name: str,
    ) -> List[str]:

        role = self.roles.get(role_name)

        if role is None:

            return []

        return sorted(role.permissions)

    def assign_permission(
        self,
        role_name: str,
        permission: str,
    ):

        role = self.get_role(role_name)

        if role:

            role.permissions.add(permission)

    def revoke_permission(
        self,
        role_name: str,
        permission: str,
    ):

        role = self.get_role(role_name)

        if role:

            role.permissions.discard(permission)

    def list_roles(self) -> List[str]:

        return sorted(self.roles.keys())
