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

# ======================================================
# Add Message
# ======================================================

def add_message(
    self,
    message: Message,
) -> None:
    """
    Add a message to the session.
    """

    if not isinstance(

        message,

        Message,

    ):

        raise SessionError(

            "Expected Message object."

        )

    self.messages.append(

        message

    )

    self.touch()


# ======================================================
# Remove Message
# ======================================================

def remove_message(
    self,
    message_id: str,
) -> bool:
    """
    Remove a message by ID.
    """

    for index, message in enumerate(

        self.messages

    ):

        if message.id == message_id:

            del self.messages[index]

            self.touch()

            return True

    return False


# ======================================================
# Get Message
# ======================================================

def get_message(
    self,
    message_id: str,
) -> Optional[Message]:
    """
    Retrieve message by ID.
    """

    for message in self.messages:

        if message.id == message_id:

            return message

    return None


# ======================================================
# Last Message
# ======================================================

def last_message(
    self,
) -> Optional[Message]:
    """
    Return latest message.
    """

    if not self.messages:

        return None

    return self.messages[-1]


# ======================================================
# First Message
# ======================================================

def first_message(
    self,
) -> Optional[Message]:
    """
    Return first message.
    """

    if not self.messages:

        return None

    return self.messages[0]


# ======================================================
# Clear Messages
# ======================================================

def clear_messages(
    self,
) -> None:
    """
    Remove all messages.
    """

    self.messages.clear()

    self.touch()


# ======================================================
# Message Count
# ======================================================

@property
def message_count(
    self,
) -> int:
    """
    Total messages.
    """

    return len(

        self.messages

    )


# ======================================================
# Add Token Usage
# ======================================================

def add_token_usage(
    self,
    usage: TokenUsage,
) -> None:
    """
    Add token usage.
    """

    if not isinstance(

        usage,

        TokenUsage,

    ):

        raise SessionError(

            "Expected TokenUsage."

        )

    self.token_usage.append(

        usage

    )

    self.touch()


# ======================================================
# Clear Token Usage
# ======================================================

def clear_token_usage(
    self,
) -> None:
    """
    Remove all token records.
    """

    self.token_usage.clear()

    self.touch()


# ======================================================
# Total Tokens
# ======================================================

@property
def total_tokens(
    self,
) -> int:
    """
    Aggregate total tokens.
    """

    return sum(

        usage.total_tokens

        for usage

        in self.token_usage

    )


# ======================================================
# Total Cost
# ======================================================

@property
def total_cost(
    self,
) -> float:
    """
    Aggregate total cost.
    """

    return round(

        sum(

            usage.total_cost

            for usage

            in self.token_usage

        ),

        6,

    )


# ======================================================
# Conversation Statistics
# ======================================================

def conversation_statistics(
    self,
) -> Dict[str, Any]:
    """
    Conversation statistics.
    """

    user_messages = 0

    assistant_messages = 0

    tool_messages = 0

    system_messages = 0

    for message in self.messages:

        role = getattr(

            message,

            "role",

            None,

        )

        role_name = (

            role.value

            if hasattr(

                role,

                "value",

            )

            else str(role)

        )

        role_name = role_name.lower()

        if role_name == "user":

            user_messages += 1

        elif role_name == "assistant":

            assistant_messages += 1

        elif role_name == "tool":

            tool_messages += 1

        elif role_name == "system":

            system_messages += 1

    return {

        "total_messages":

            self.message_count,

        "user_messages":

            user_messages,

        "assistant_messages":

            assistant_messages,

        "tool_messages":

            tool_messages,

        "system_messages":

            system_messages,

    }


# ======================================================
# Token Statistics
# ======================================================

def token_statistics(
    self,
) -> Dict[str, Any]:
    """
    Aggregate token statistics.
    """

    prompt = sum(

        usage.prompt_tokens

        for usage

        in self.token_usage

    )

    completion = sum(

        usage.completion_tokens

        for usage

        in self.token_usage

    )

    embedding = sum(

        usage.embedding_tokens

        for usage

        in self.token_usage

    )

    cached = sum(

        usage.cached_tokens

        for usage

        in self.token_usage

    )

    return {

        "prompt_tokens":

            prompt,

        "completion_tokens":

            completion,

        "embedding_tokens":

            embedding,

        "cached_tokens":

            cached,

        "total_tokens":

            self.total_tokens,

        "total_cost":

            self.total_cost,

    }


# ======================================================
# Search Messages
# ======================================================

def search_messages(
    self,
    keyword: str,
) -> List[Message]:
    """
    Search messages containing keyword.
    """

    keyword = keyword.lower()

    results = []

    for message in self.messages:

        content = getattr(

            message,

            "content",

            "",

        )

        if keyword in content.lower():

            results.append(

                message

            )

    return results


# ======================================================
# Get Messages by Role
# ======================================================

def messages_by_role(
    self,
    role: str,
) -> List[Message]:
    """
    Filter messages by role.
    """

    role = role.lower()

    results = []

    for message in self.messages:

        message_role = getattr(

            message,

            "role",

            None,

        )

        value = (

            message_role.value

            if hasattr(

                message_role,

                "value",

            )

            else str(

                message_role

            )

        )

        if value.lower() == role:

            results.append(

                message

            )

    return results

# ======================================================
# Update Context
# ======================================================

def update_context(
    self,
    context: Dict[str, Any],
) -> None:
    """
    Merge new context.
    """

    self.context.update(

        context

    )

    self.touch()


# ======================================================
# Clear Context
# ======================================================

def clear_context(
    self,
) -> None:
    """
    Remove all context.
    """

    self.context.clear()

    self.touch()


# ======================================================
# Get Context
# ======================================================

def get_context(
    self,
    key: str,
    default: Any = None,
) -> Any:
    """
    Retrieve context value.
    """

    return self.context.get(

        key,

        default,

    )


# ======================================================
# Update Memory
# ======================================================

def update_memory(
    self,
    memory: Dict[str, Any],
) -> None:
    """
    Merge memory.
    """

    self.memory.update(

        memory

    )

    self.touch()


# ======================================================
# Clear Memory
# ======================================================

def clear_memory(
    self,
) -> None:
    """
    Remove all memory.
    """

    self.memory.clear()

    self.touch()


# ======================================================
# Get Memory
# ======================================================

def get_memory(
    self,
    key: str,
    default: Any = None,
) -> Any:
    """
    Retrieve memory value.
    """

    return self.memory.get(

        key,

        default,

    )


# ======================================================
# Summary
# ======================================================

def summary(
    self,
) -> Dict[str, Any]:
    """
    Human-readable session summary.
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

        "session_type":

            self.session_type.value,

        "messages":

            self.message_count,

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

        "total_tokens":

            self.total_tokens,

        "total_cost":

            self.total_cost,

        "last_activity":

            self.last_activity.isoformat(),

    }


# ======================================================
# Analytics
# ======================================================

def analytics(
    self,
) -> Dict[str, Any]:
    """
    Complete analytics.
    """

    return {

        "conversation":

            self.conversation_statistics(),

        "tokens":

            self.token_statistics(),

        "summary":

            self.summary(),

    }


# ======================================================
# Diagnostics
# ======================================================

def diagnostics(
    self,
) -> Dict[str, Any]:
    """
    Session diagnostics.
    """

    return {

        "model":

            self.__class__.__name__,

        "id":

            self.id,

        "user_id":

            self.user_id,

        "title":

            self.title,

        "status":

            self.status.value,

        "type":

            self.session_type.value,

        "created_at":

            self.created_at.isoformat(),

        "updated_at":

            self.updated_at.isoformat(),

        "last_activity":

            self.last_activity.isoformat(),

        "statistics":

            self.statistics(),

    }


# ======================================================
# Export
# ======================================================

def export(
    self,
) -> Dict[str, Any]:
    """
    Export complete session.
    """

    return {

        "session":

            self.to_dict(),

        "summary":

            self.summary(),

        "analytics":

            self.analytics(),

        "diagnostics":

            self.diagnostics(),

    }


# ======================================================
# JSON Serialization
# ======================================================

def to_json(
    self,
    indent: int = 4,
) -> str:
    """
    Serialize session to JSON.
    """

    return json.dumps(

        self.to_dict(),

        indent=indent,

        ensure_ascii=False,

    )


# ======================================================
# Create From Dictionary
# ======================================================

@classmethod
def from_dict(
    cls,
    data: Dict[str, Any],
) -> "Session":
    """
    Create Session from dictionary.
    """

    return cls(

        user_id=data["user_id"],

        title=data["title"],

        session_type=SessionType(

            data.get(

                "session_type",

                SessionType.CHAT,

            )

        ),

        status=SessionStatus(

            data.get(

                "status",

                SessionStatus.ACTIVE,

            )

        ),

        messages=[

            Message.from_dict(

                item

            )

            for item

            in data.get(

                "messages",

                [],

            )

        ],

        context=data.get(

            "context",

            {},

        ),

        memory=data.get(

            "memory",

            {},

        ),

        metadata=data.get(

            "metadata",

            {},

        ),

        token_usage=[

            TokenUsage.from_dict(

                item

            )

            for item

            in data.get(

                "token_usage",

                [],

            )

        ],

        id=data.get(

            "id",

            str(

                uuid.uuid4()

            ),

        ),

        created_at=datetime.fromisoformat(

            data.get(

                "created_at",

                datetime.utcnow().isoformat(),

            )

        ),

        updated_at=datetime.fromisoformat(

            data.get(

                "updated_at",

                datetime.utcnow().isoformat(),

            )

        ),

        last_activity=datetime.fromisoformat(

            data.get(

                "last_activity",

                datetime.utcnow().isoformat(),

            )

        ),

        is_pinned=data.get(

            "is_pinned",

            False,

        ),

        is_favorite=data.get(

            "is_favorite",

            False,

        ),

        archived=data.get(

            "archived",

            False,

        ),

    )


# ======================================================
# Create From JSON
# ======================================================

@classmethod
def from_json(
    cls,
    json_string: str,
) -> "Session":
    """
    Create Session from JSON.
    """

    return cls.from_dict(

        json.loads(

            json_string

        )

    )


# ======================================================
# Clone
# ======================================================

def clone(
    self,
) -> "Session":
    """
    Deep copy session.
    """

    return Session.from_dict(

        self.to_dict()

    )

# ======================================================
# Cleanup
# ======================================================

def cleanup(
    self,
) -> None:
    """
    Cleanup session resources.
    """

    logger.info(

        f"Cleaning Session "

        f"{self.id}"

    )


# ======================================================
# Context Manager
# ======================================================

def __enter__(
    self,
):
    """
    Context manager entry.
    """

    logger.info(

        f"Entering session "

        f"{self.id}"

    )

    return self


def __exit__(
    self,
    exc_type,
    exc_value,
    traceback,
):
    """
    Context manager exit.
    """

    self.cleanup()


# ======================================================
# Archive Session
# ======================================================

def archive(
    self,
) -> None:
    """
    Archive session.
    """

    self.archived = True

    self.status = SessionStatus.ARCHIVED

    self.touch()


# ======================================================
# Restore Session
# ======================================================

def restore(
    self,
) -> None:
    """
    Restore archived session.
    """

    self.archived = False

    self.status = SessionStatus.ACTIVE

    self.touch()


# ======================================================
# Close Session
# ======================================================

def close(
    self,
) -> None:
    """
    Close session.
    """

    self.status = SessionStatus.CLOSED

    self.touch()


# ======================================================
# Pin Session
# ======================================================

def pin(
    self,
) -> None:
    """
    Pin session.
    """

    self.is_pinned = True

    self.touch()


# ======================================================
# Unpin Session
# ======================================================

def unpin(
    self,
) -> None:
    """
    Unpin session.
    """

    self.is_pinned = False

    self.touch()


# ======================================================
# Favorite Session
# ======================================================

def favorite(
    self,
) -> None:
    """
    Mark as favorite.
    """

    self.is_favorite = True

    self.touch()


# ======================================================
# Unfavorite Session
# ======================================================

def unfavorite(
    self,
) -> None:
    """
    Remove favorite.
    """

    self.is_favorite = False

    self.touch()


# ======================================================
# Refresh
# ======================================================

def refresh(
    self,
) -> None:
    """
    Refresh timestamps.
    """

    self.touch()

    logger.info(

        f"Session "

        f"{self.id} refreshed."

    )


# ======================================================
# Copy
# ======================================================

def copy(
    self,
) -> "Session":
    """
    Alias for clone().
    """

    return self.clone()


# ======================================================
# String Representation
# ======================================================

def __repr__(
    self,
):
    """
    Developer representation.
    """

    return (

        "Session("

        f"id='{self.id}', "

        f"title='{self.title}', "

        f"messages={len(self.messages)}, "

        f"status='{self.status.value}'"

        ")"

    )


# ======================================================
# Human Readable
# ======================================================

def __str__(
    self,
):
    """
    Human-readable string.
    """

    return (

        f"{self.title} "

        f"({self.message_count} messages)"

    )


# ======================================================
# Length
# ======================================================

def __len__(
    self,
):
    """
    Number of messages.
    """

    return len(

        self.messages

    )


# ======================================================
# Iterator
# ======================================================

def __iter__(
    self,
):
    """
    Iterate over messages.
    """

    return iter(

        self.messages

    )


# ======================================================
# Get Item
# ======================================================

def __getitem__(
    self,
    index: int,
):
    """
    Get message by index.
    """

    return self.messages[

        index

    ]


# ======================================================
# Contains
# ======================================================

def __contains__(
    self,
    item,
):
    """
    Check whether message exists.
    """

    return item in self.messages


# ======================================================
# Equality
# ======================================================

def __eq__(
    self,
    other: object,
) -> bool:
    """
    Compare sessions.
    """

    if not isinstance(

        other,

        Session,

    ):

        return False

    return self.id == other.id


# ======================================================
# Hash
# ======================================================

def __hash__(
    self,
):
    """
    Hash by UUID.
    """

    return hash(

        self.id

    )


# ======================================================
# Boolean
# ======================================================

def __bool__(
    self,
):
    """
    Check whether session is active.
    """

    return (

        self.status

        ==

        SessionStatus.ACTIVE

    )


# ======================================================
# Convenience Properties
# ======================================================

@property
def is_active(
    self,
) -> bool:
    """
    Active session.
    """

    return (

        self.status

        ==

        SessionStatus.ACTIVE

    )


@property
def has_messages(
    self,
) -> bool:
    """
    Whether session has messages.
    """

    return len(

        self.messages

    ) > 0


@property
def has_memory(
    self,
) -> bool:
    """
    Whether memory exists.
    """

    return len(

        self.memory

    ) > 0


@property
def has_context(
    self,
) -> bool:
    """
    Whether context exists.
    """

    return len(

        self.context

    ) > 0
