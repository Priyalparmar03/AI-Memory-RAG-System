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

from .token_usage import TokenUsage


logger = logging.getLogger(__name__)


# ==========================================================
# Exception
# ==========================================================

class MessageError(Exception):
    """
    Message model exception.
    """
    pass


# ==========================================================
# Message Role
# ==========================================================

class MessageRole(str, Enum):

    SYSTEM = "system"

    USER = "user"

    ASSISTANT = "assistant"

    TOOL = "tool"

    FUNCTION = "function"

    DEVELOPER = "developer"


# ==========================================================
# Message Status
# ==========================================================

class MessageStatus(str, Enum):

    PENDING = "pending"

    SENT = "sent"

    DELIVERED = "delivered"

    READ = "read"

    FAILED = "failed"

    DELETED = "deleted"


# ==========================================================
# Message Type
# ==========================================================

class MessageType(str, Enum):

    TEXT = "text"

    IMAGE = "image"

    DOCUMENT = "document"

    AUDIO = "audio"

    VIDEO = "video"

    CODE = "code"

    TOOL_CALL = "tool_call"

    TOOL_RESULT = "tool_result"

    SYSTEM_EVENT = "system_event"


# ==========================================================
# Attachment
# ==========================================================

@dataclass(slots=True)
class Attachment:
    """
    Message attachment.
    """

    name: str

    file_type: str

    file_path: str

    size: int = 0

    mime_type: str = ""

    metadata: Dict[str, Any] = field(

        default_factory=dict

    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {

            "name":

                self.name,

            "file_type":

                self.file_type,

            "file_path":

                self.file_path,

            "size":

                self.size,

            "mime_type":

                self.mime_type,

            "metadata":

                self.metadata,

        }


# ==========================================================
# Message
# ==========================================================

@dataclass(slots=True)
class Message:
    """
    Production Message Model.
    """

    role: MessageRole

    content: str

    message_type: MessageType = (

        MessageType.TEXT

    )

    status: MessageStatus = (

        MessageStatus.PENDING

    )

    attachments: List[Attachment] = field(

        default_factory=list

    )

    metadata: Dict[str, Any] = field(

        default_factory=dict

    )

    token_usage: Optional[TokenUsage] = None

    parent_message_id: Optional[str] = None

    thread_id: Optional[str] = None

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

    edited: bool = False

    edited_at: Optional[

        datetime

    ] = None


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
        Validate message.
        """

        if not self.content.strip():

            raise MessageError(

                "Message content "

                "cannot be empty."

            )


    # ======================================================
    # Update Timestamp
    # ======================================================

    def touch(
        self,
    ) -> None:
        """
        Update timestamp.
        """

        self.updated_at = (

            datetime.utcnow()

        )


    # ======================================================
    # Edit Content
    # ======================================================

    def edit(
        self,
        content: str,
    ) -> None:
        """
        Edit message.
        """

        if not content.strip():

            raise MessageError(

                "Content cannot "

                "be empty."

            )

        self.content = content

        self.edited = True

        self.edited_at = (

            datetime.utcnow()

        )

        self.touch()


    # ======================================================
    # Serialization
    # ======================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize message.
        """

        return {

            "id":

                self.id,

            "role":

                self.role.value,

            "content":

                self.content,

            "message_type":

                self.message_type.value,

            "status":

                self.status.value,

            "attachments":

                [

                    attachment.to_dict()

                    for attachment

                    in self.attachments

                ],

            "metadata":

                self.metadata,

            "token_usage":

                self.token_usage.to_dict()

                if self.token_usage

                else None,

            "parent_message_id":

                self.parent_message_id,

            "thread_id":

                self.thread_id,

            "created_at":

                self.created_at.isoformat(),

            "updated_at":

                self.updated_at.isoformat(),

            "edited":

                self.edited,

            "edited_at":

                self.edited_at.isoformat()

                if self.edited_at

                else None,

        }


# ======================================================
# Update Status
# ======================================================

def update_status(
    self,
    status: MessageStatus,
) -> None:
    """
    Update message status.
    """

    self.status = status

    self.touch()


# ======================================================
# Mark Delivered
# ======================================================

def mark_delivered(
    self,
) -> None:
    """
    Mark message as delivered.
    """

    self.status = MessageStatus.DELIVERED

    self.touch()


# ======================================================
# Mark Read
# ======================================================

def mark_read(
    self,
) -> None:
    """
    Mark message as read.
    """

    self.status = MessageStatus.READ

    self.touch()


# ======================================================
# Mark Failed
# ======================================================

def mark_failed(
    self,
) -> None:
    """
    Mark message as failed.
    """

    self.status = MessageStatus.FAILED

    self.touch()


# ======================================================
# Delete Message
# ======================================================

def delete(
    self,
) -> None:
    """
    Soft delete message.
    """

    self.status = MessageStatus.DELETED

    self.touch()


# ======================================================
# Add Attachment
# ======================================================

def add_attachment(
    self,
    attachment: Attachment,
) -> None:
    """
    Add attachment.
    """

    if not isinstance(

        attachment,

        Attachment,

    ):

        raise MessageError(

            "Expected Attachment."

        )

    self.attachments.append(

        attachment

    )

    self.touch()


# ======================================================
# Remove Attachment
# ======================================================

def remove_attachment(
    self,
    name: str,
) -> bool:
    """
    Remove attachment by name.
    """

    for index, attachment in enumerate(

        self.attachments

    ):

        if attachment.name == name:

            del self.attachments[index]

            self.touch()

            return True

    return False


# ======================================================
# Clear Attachments
# ======================================================

def clear_attachments(
    self,
) -> None:
    """
    Remove all attachments.
    """

    self.attachments.clear()

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
    Add metadata.
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
    Remove metadata.
    """

    self.metadata.pop(

        key,

        None,

    )

    self.touch()


# ======================================================
# Clear Metadata
# ======================================================

def clear_metadata(
    self,
) -> None:
    """
    Remove all metadata.
    """

    self.metadata.clear()

    self.touch()


# ======================================================
# Set Token Usage
# ======================================================

def set_token_usage(
    self,
    usage: TokenUsage,
) -> None:
    """
    Assign token usage.
    """

    if not isinstance(

        usage,

        TokenUsage,

    ):

        raise MessageError(

            "Expected TokenUsage."

        )

    self.token_usage = usage

    self.touch()


# ======================================================
# Remove Token Usage
# ======================================================

def clear_token_usage(
    self,
) -> None:
    """
    Remove token usage.
    """

    self.token_usage = None

    self.touch()


# ======================================================
# Statistics
# ======================================================

def statistics(
    self,
) -> Dict[str, Any]:
    """
    Message statistics.
    """

    return {

        "id":

            self.id,

        "role":

            self.role.value,

        "status":

            self.status.value,

        "message_type":

            self.message_type.value,

        "characters":

            len(

                self.content

            ),

        "words":

            len(

                self.content.split()

            ),

        "attachments":

            len(

                self.attachments

            ),

        "metadata_entries":

            len(

                self.metadata

            ),

        "edited":

            self.edited,

        "has_token_usage":

            self.token_usage

            is not None,

    }


# ======================================================
# Attachment Count
# ======================================================

@property
def attachment_count(
    self,
) -> int:
    """
    Number of attachments.
    """

    return len(

        self.attachments

    )


# ======================================================
# Has Attachments
# ======================================================

@property
def has_attachments(
    self,
) -> bool:
    """
    Whether attachments exist.
    """

    return (

        len(

            self.attachments

        )

        >

        0

    )


# ======================================================
# Word Count
# ======================================================

@property
def word_count(
    self,
) -> int:
    """
    Number of words.
    """

    return len(

        self.content.split()

    )


# ======================================================
# Character Count
# ======================================================

@property
def character_count(
    self,
) -> int:
    """
    Number of characters.
    """

    return len(

        self.content

    )
