from __future__ import annotations

import json
import logging
import uuid

from dataclasses import (
    dataclass,
    field,
)

from datetime import datetime

from enum import Enum

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from .message import Message
from .token_usage import TokenUsage


logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class SessionError(Exception):
    """
    Session model exception.
    """
    pass


# ==========================================================
# Session Status
# ==========================================================

class SessionStatus(str, Enum):

    ACTIVE = "active"

    PAUSED = "paused"

    ARCHIVED = "archived"

    CLOSED = "closed"

    DELETED = "deleted"


# ==========================================================
# Session Type
# ==========================================================

class SessionType(str, Enum):

    CHAT = "chat"

    RAG = "rag"

    AGENT = "agent"

    MEMORY = "memory"

    WORKFLOW = "workflow"

    SYSTEM = "system"


# ==========================================================
# Session
# ==========================================================

@dataclass(slots=True)
class Session:
    """
    Production Session Model.
    """

    user_id: str

    title: str

    session_type: SessionType = (

        SessionType.CHAT

    )

    status: SessionStatus = (

        SessionStatus.ACTIVE

    )

    messages: List[Message] = field(

        default_factory=list

    )

    context: Dict[str, Any] = field(

        default_factory=dict

    )

    memory: Dict[str, Any] = field(

        default_factory=dict

    )

    metadata: Dict[str, Any] = field(

        default_factory=dict

    )

    token_usage: List[TokenUsage] = field(

        default_factory=list

    )

    id: str = field(

        default_factory=lambda:

        str(uuid.uuid4())

    )

    created_at: datetime = field(

        default_factory=datetime.utcnow

    )

    updated_at: datetime = field(

        default_factory=datetime.utcnow

    )

    last_activity: datetime = field(

        default_factory=datetime.utcnow

    )

    is_pinned: bool = False

    is_favorite: bool = False

    archived: bool = False


    # ======================================================
    # Initialization
    # ======================================================

    def __post_init__(
        self,
    ):

        self.validate()


    # ======================================================
    # Validation
    # ======================================================

    def validate(
        self,
    ) -> None:
        """
        Validate session.
        """

        if not self.user_id.strip():

            raise SessionError(

                "User ID cannot be empty."

            )

        if not self.title.strip():

            raise SessionError(

                "Session title cannot "

                "be empty."

            )


    # ======================================================
    # Update Timestamp
    # ======================================================

    def touch(
        self,
    ) -> None:
        """
        Update timestamps.
        """

        now = datetime.utcnow()

        self.updated_at = now

        self.last_activity = now


    # ======================================================
    # Rename Session
    # ======================================================

    def rename(
        self,
        title: str,
    ) -> None:
        """
        Rename session.
        """

        if not title.strip():

            raise SessionError(

                "Title cannot be empty."

            )

        self.title = title

        self.touch()


    # ======================================================
    # Update Metadata
    # ======================================================

    def update_metadata(
        self,
        **kwargs,
    ) -> None:
        """
        Update metadata.
        """

        self.metadata.update(

            kwargs

        )

        self.touch()


    # ======================================================
    # Add Context
    # ======================================================

    def add_context(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Add context.
        """

        self.context[key] = value

        self.touch()


    # ======================================================
    # Add Memory
    # ======================================================

    def add_memory(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Add memory.
        """

        self.memory[key] = value

        self.touch()


    # ======================================================
    # Basic Statistics
    # ======================================================

    def statistics(
        self,
    ) -> Dict[str, Any]:
        """
        Session statistics.
        """

        return {

            "id":

                self.id,

            "title":

                self.title,

            "user_id":

                self.user_id,

            "status":

                self.status.value,

            "type":

                self.session_type.value,

            "messages":

                len(

                    self.messages

                ),

            "context_items":

                len(

                    self.context

                ),

            "memory_items":

                len(

                    self.memory

                ),

            "token_records":

                len(

                    self.token_usage

                ),

            "created_at":

                self.created_at.isoformat(),

            "updated_at":

                self.updated_at.isoformat(),

            "last_activity":

                self.last_activity.isoformat(),

        }


    # ======================================================
    # Serialization
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize session.
        """

        return {

            "id":

                self.id,

            "user_id":

                self.user_id,

            "title":

                self.title,

            "session_type":

                self.session_type.value,

            "status":

                self.status.value,

            "messages":

                [

                    message.to_dict()

                    for message

                    in self.messages

                ],

            "context":

                self.context,

            "memory":

                self.memory,

            "metadata":

                self.metadata,

            "token_usage":

                [

                    usage.to_dict()

                    for usage

                    in self.token_usage

                ],

            "created_at":

                self.created_at.isoformat(),

            "updated_at":

                self.updated_at.isoformat(),

            "last_activity":

                self.last_activity.isoformat(),

            "is_pinned":

                self.is_pinned,

            "is_favorite":

                self.is_favorite,

            "archived":

                self.archived,

        }
