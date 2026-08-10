from __future__ import annotations
from collections import deque
from typing import Any
from typing import Tuple
from services.embedding_service import EmbeddingService
import logging
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Production Memory Service.

    This service stores and retrieves
    conversations independently from ChatService.
    """

    DATABASE_NAME = "memory.db"

    def __init__(
        self,
        database_path: str = "./database",
    ):

        self.database_path = Path(database_path)
        self.database_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.database = (
            self.database_path /
            self.DATABASE_NAME
        )

        self.connection = sqlite3.connect(
            self.database,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self._initialize_database()

        logger.info(
            "MemoryService initialized."
        )

    # =====================================================
    # Database Initialization
    # =====================================================

    def _initialize_database(self):

        cursor = self.connection.cursor()

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS conversations(

            conversation_id TEXT PRIMARY KEY,

            user_id TEXT,

            title TEXT,

            created_at TEXT,

            updated_at TEXT

        )

        """)

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS messages(

            message_id TEXT PRIMARY KEY,

            conversation_id TEXT,

            role TEXT,

            content TEXT,

            created_at TEXT,

            FOREIGN KEY(conversation_id)

            REFERENCES conversations(conversation_id)

        )

        """)

        self.connection.commit()

    # =====================================================
    # Conversation
    # =====================================================

    def create_conversation(
        self,
        user_id: str,
        title: str = "New Conversation",
    ) -> str:

        conversation_id = str(uuid.uuid4())

        timestamp = datetime.utcnow().isoformat()

        cursor = self.connection.cursor()

        cursor.execute(

            """

            INSERT INTO conversations

            VALUES (?, ?, ?, ?, ?)

            """,

            (

                conversation_id,

                user_id,

                title,

                timestamp,

                timestamp,

            ),

        )

        self.connection.commit()

        logger.info(
            "Conversation created."
        )

        return conversation_id

    # =====================================================
    # Get Conversation
    # =====================================================

    def get_conversation(
        self,
        conversation_id: str,
    ) -> Optional[Dict]:

        cursor = self.connection.cursor()

        cursor.execute(

            """

            SELECT *

            FROM conversations

            WHERE conversation_id=?

            """,

            (conversation_id,),

        )

        row = cursor.fetchone()

        if row is None:

            return None

        return dict(row)

    # =====================================================
    # List Conversations
    # =====================================================

    def list_conversations(
        self,
        user_id: str,
    ) -> List[Dict]:

        cursor = self.connection.cursor()

        cursor.execute(

            """

            SELECT *

            FROM conversations

            WHERE user_id=?

            ORDER BY updated_at DESC

            """,

            (user_id,),

        )

        rows = cursor.fetchall()

        return [

            dict(row)

            for row in rows

        ]

    # =====================================================
    # Rename Conversation
    # =====================================================

    def rename_conversation(
        self,
        conversation_id: str,
        title: str,
    ):

        cursor = self.connection.cursor()

        cursor.execute(

            """

            UPDATE conversations

            SET title=?,

                updated_at=?

            WHERE conversation_id=?

            """,

            (

                title,

                datetime.utcnow().isoformat(),

                conversation_id,

            ),

        )

        self.connection.commit()

    # =====================================================
    # Delete Conversation
    # =====================================================

    def delete_conversation(
        self,
        conversation_id: str,
    ):

        cursor = self.connection.cursor()

        cursor.execute(

            """

            DELETE FROM messages

            WHERE conversation_id=?

            """,

            (conversation_id,),

        )

        cursor.execute(

            """

            DELETE FROM conversations

            WHERE conversation_id=?

            """,

            (conversation_id,),

        )

        self.connection.commit()

        logger.info(
            "Conversation deleted."
        )

    # =====================================================
    # Add Message
    # =====================================================

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> str:

        message_id = str(uuid.uuid4())

        timestamp = datetime.utcnow().isoformat()

        cursor = self.connection.cursor()

        cursor.execute(

            """

            INSERT INTO messages

            VALUES (?, ?, ?, ?, ?)

            """,

            (

                message_id,

                conversation_id,

                role,

                content,

                timestamp,

            ),

        )

        cursor.execute(

            """

            UPDATE conversations

            SET updated_at=?

            WHERE conversation_id=?

            """,

            (

                timestamp,

                conversation_id,

            ),

        )

        self.connection.commit()

        return message_id

    # =====================================================
    # Get Messages
    # =====================================================

    def get_messages(
        self,
        conversation_id: str,
    ) -> List[Dict]:

        cursor = self.connection.cursor()

        cursor.execute(

            """

            SELECT *

            FROM messages

            WHERE conversation_id=?

            ORDER BY created_at

            """,

            (conversation_id,),

        )

        rows = cursor.fetchall()

        return [

            dict(row)

            for row in rows

        ]

    # =====================================================
    # Delete Message
    # =====================================================

    def delete_message(
        self,
        message_id: str,
    ):

        cursor = self.connection.cursor()

        cursor.execute(

            """

            DELETE FROM messages

            WHERE message_id=?

            """,

            (message_id,),

        )

        self.connection.commit()

    # =====================================================
    # Health
    # =====================================================

    def health(self) -> Dict:

        cursor = self.connection.cursor()

        cursor.execute(

            """

            SELECT COUNT(*)

            FROM conversations

            """

        )

        conversations = cursor.fetchone()[0]

        cursor.execute(

            """

            SELECT COUNT(*)

            FROM messages

            """

        )

        messages = cursor.fetchone()[0]

        return {

            "status": "healthy",

            "database": str(self.database),

            "conversations": conversations,

            "messages": messages,

        }

    # =====================================================
    # Close
    # =====================================================

    def close(self):

        self.connection.close()

        logger.info(
            "MemoryService closed."
        )

  # =====================================================
# Recent Messages
# =====================================================

def recent_messages(
    self,
    conversation_id: str,
    limit: int = 10,
) -> List[Dict]:
    """
    Return the most recent messages.
    """

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM messages
        WHERE conversation_id=?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (
            conversation_id,
            limit,
        ),
    )

    rows = cursor.fetchall()

    return [
        dict(row)
        for row in reversed(rows)
    ]

# =====================================================
# Build Context
# =====================================================

def build_context(
    self,
    conversation_id: str,
    limit: int = 10,
) -> List[Dict]:
    """
    Build conversation history
    for the LLM.
    """

    messages = self.recent_messages(
        conversation_id,
        limit,
    )

    history = []

    for message in messages:

        history.append(

            {

                "role": message["role"],

                "content": message["content"],

            }

        )

    return history

# =====================================================
# Estimate Tokens
# =====================================================

@staticmethod
def estimate_tokens(
    text: str,
) -> int:
    """
    Approximate token count.

    Rough estimate:
    1 token ≈ 4 characters
    """

    if not text:

        return 0

    return max(

        1,

        len(text) // 4,

    )

# =====================================================
# Conversation Tokens
# =====================================================

def conversation_tokens(
    self,
    conversation_id: str,
) -> int:

    messages = self.get_messages(
        conversation_id
    )

    total = 0

    for message in messages:

        total += self.estimate_tokens(

            message["content"]

        )

    return total

# =====================================================
# Context Window
# =====================================================

def context_window(
    self,
    conversation_id: str,
    max_tokens: int = 4000,
) -> List[Dict]:
    """
    Keep newest messages
    within token budget.
    """

    messages = self.get_messages(
        conversation_id
    )

    window = deque()

    total = 0

    for message in reversed(messages):

        tokens = self.estimate_tokens(

            message["content"]

        )

        if total + tokens > max_tokens:

            break

        total += tokens

        window.appendleft(

            {

                "role":

                    message["role"],

                "content":

                    message["content"],

            }

        )

    return list(window)

# =====================================================
# Conversation Summary
# =====================================================

def summarize_conversation(
    self,
    conversation_id: str,
    max_messages: int = 5,
) -> str:
    """
    Lightweight summary of
    the latest messages.

    Later this can be replaced
    by an LLM-generated summary.
    """

    messages = self.recent_messages(

        conversation_id,

        max_messages,

    )

    lines = []

    for message in messages:

        content = message["content"]

        if len(content) > 120:

            content = content[:120] + "..."

        lines.append(

            f"{message['role']}: {content}"

        )

    return "\n".join(lines)

# =====================================================
# Prune Conversation
# =====================================================

def prune_history(
    self,
    conversation_id: str,
    keep_last: int = 100,
):
    """
    Delete very old messages.
    """

    messages = self.get_messages(
        conversation_id
    )

    if len(messages) <= keep_last:

        return

    remove = messages[:-keep_last]

    cursor = self.connection.cursor()

    for message in remove:

        cursor.execute(

            """
            DELETE FROM messages
            WHERE message_id=?
            """,

            (

                message["message_id"],

            ),

        )

    self.connection.commit()

    logger.info(

        "Conversation pruned."

    )

# =====================================================
# Prompt Memory
# =====================================================

def prompt_memory(
    self,
    conversation_id: str,
    max_tokens: int = 3000,
) -> Dict:
    """
    Return memory ready for
    PromptService.
    """

    context = self.context_window(

        conversation_id,

        max_tokens,

    )

    return {

        "messages":

            context,

        "summary":

            self.summarize_conversation(

                conversation_id

            ),

        "token_count":

            self.conversation_tokens(

                conversation_id

            ),

    }

# =====================================================
# Statistics
# =====================================================

def statistics(self):

    cursor = self.connection.cursor()

    cursor.execute(

        """

        SELECT COUNT(*)

        FROM conversations

        """

    )

    conversations = cursor.fetchone()[0]

    cursor.execute(

        """

        SELECT COUNT(*)

        FROM messages

        """

    )

    messages = cursor.fetchone()[0]

    return {

        "conversations":

            conversations,

        "messages":

            messages,

        "database":

            str(self.database),

    }

cursor.execute("""

CREATE TABLE IF NOT EXISTS memory(

    memory_id TEXT PRIMARY KEY,

    user_id TEXT,

    conversation_id TEXT,

    content TEXT,

    embedding_id TEXT,

    importance REAL DEFAULT 0.5,

    created_at TEXT,

    last_accessed TEXT

)

""")

# =====================================================
# Store Memory
# =====================================================

def store_memory(
    self,
    user_id: str,
    conversation_id: str,
    content: str,
    importance: float = 0.5,
) -> str:
    """
    Store a memory record.
    The embedding should also be indexed in the vector store.
    """

    memory_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    embedding = self.embedding_service.embed_query(content)

    # TODO:
    # vector_store.add(
    #     id=memory_id,
    #     embedding=embedding,
    #     metadata={
    #         "memory_id": memory_id,
    #         "user_id": user_id,
    #         "conversation_id": conversation_id,
    #         "type": "memory",
    #     },
    #     document=content,
    # )

    cursor = self.connection.cursor()

    cursor.execute(
        """
        INSERT INTO memory
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory_id,
            user_id,
            conversation_id,
            content,
            memory_id,
            importance,
            timestamp,
            timestamp,
        ),
    )

    self.connection.commit()

    return memory_id

# =====================================================
# List Memories
# =====================================================

def list_memories(
    self,
    user_id: str,
) -> List[Dict]:

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM memory
        WHERE user_id=?
        ORDER BY created_at DESC
        """,
        (user_id,),
    )

    return [
        dict(row)
        for row in cursor.fetchall()
    ]

# =====================================================
# Semantic Memory Search
# =====================================================

def search_memory(
    self,
    query: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Search the vector store for related memories.
    """

    query_embedding = self.embedding_service.embed_query(query)

    # Example placeholder:
    # results = self.vector_store.search(
    #     embedding=query_embedding,
    #     top_k=top_k,
    #     filter={"type": "memory"},
    # )

    return []

# =====================================================
# Memory Context
# =====================================================

def memory_context(
    self,
    conversation_id: str,
    query: str,
    max_tokens: int = 3000,
) -> Dict:
    """
    Build combined context for ChatService.
    """

    short_term = self.context_window(
        conversation_id,
        max_tokens=max_tokens,
    )

    long_term = self.search_memory(
        query=query,
        top_k=5,
    )

    return {

        "conversation": short_term,

        "long_term": long_term,

    }

# =====================================================
# Touch Memory
# =====================================================

def touch_memory(
    self,
    memory_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        UPDATE memory
        SET last_accessed=?
        WHERE memory_id=?
        """,
        (
            datetime.utcnow().isoformat(),
            memory_id,
        ),
    )

    self.connection.commit()

# =====================================================
# Delete Memory
# =====================================================

def delete_memory(
    self,
    memory_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        DELETE FROM memory
        WHERE memory_id=?
        """,
        (memory_id,),
    )

    self.connection.commit()

 # Also remove from vector store.
# =====================================================
# User Memory Profile
# =====================================================

def memory_profile(
    self,
    user_id: str,
) -> Dict:

    memories = self.list_memories(user_id)

    return {

        "user_id": user_id,

        "memory_count": len(memories),

        "recent": memories[:10],

    }

# =====================================================
# Memory Statistics
# =====================================================

def memory_statistics(
    self,
) -> Dict:

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM memory
        """
    )

    total = cursor.fetchone()[0]

    return {

        "stored_memories": total,

    }

# =====================================================
# Export Conversation
# =====================================================

def export_conversation(
    self,
    conversation_id: str,
) -> Dict:
    """
    Export a conversation with all messages.
    """

    conversation = self.get_conversation(
        conversation_id
    )

    if conversation is None:
        return {}

    messages = self.get_messages(
        conversation_id
    )

    return {

        "conversation": conversation,

        "messages": messages,

    }


# =====================================================
# Export User Memories
# =====================================================

def export_memories(
    self,
    user_id: str,
) -> Dict:

    return {

        "user_id": user_id,

        "memories": self.list_memories(
            user_id
        ),

    }


# =====================================================
# Cleanup Old Conversations
# =====================================================

def cleanup_conversations(
    self,
    days: int = 365,
):
    """
    Remove conversations older than N days.
    """

    from datetime import timedelta

    cutoff = (
        datetime.utcnow() -
        timedelta(days=days)
    ).isoformat()

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT conversation_id
        FROM conversations
        WHERE updated_at < ?
        """,
        (cutoff,),
    )

    rows = cursor.fetchall()

    for row in rows:

        self.delete_conversation(
            row["conversation_id"]
        )

    logger.info(
        "Old conversations cleaned."
    )


# =====================================================
# Cleanup Low Importance Memories
# =====================================================

def cleanup_memories(
    self,
    minimum_importance: float = 0.2,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        DELETE FROM memory
        WHERE importance < ?
        """,
        (minimum_importance,),
    )

    self.connection.commit()

    logger.info(
        "Low importance memories removed."
    )


# =====================================================
# Database Size
# =====================================================

def database_size(self) -> int:
    """
    Database size in bytes.
    """

    return self.database.stat().st_size


# =====================================================
# Diagnostics
# =====================================================

def diagnostics(self) -> Dict:

    return {

        "health": self.health(),

        "statistics": self.statistics(),

        "memory_statistics":
            self.memory_statistics(),

        "database_size":
            self.database_size(),

    }


# =====================================================
# Reset Memory Database
# =====================================================

def reset(self):
    """
    Delete all stored data.
    """

    cursor = self.connection.cursor()

    cursor.execute(
        "DELETE FROM messages"
    )

    cursor.execute(
        "DELETE FROM conversations"
    )

    cursor.execute(
        "DELETE FROM memory"
    )

    self.connection.commit()

    logger.warning(
        "Memory database reset."
    )


# =====================================================
# Service Version
# =====================================================

@staticmethod
def version():

    return "1.0.0"


# =====================================================
# Service Info
# =====================================================

def info(self):

    return {

        "service": "MemoryService",

        "version": self.version(),

        "database": str(self.database),

        "statistics": self.statistics(),

        "memory": self.memory_statistics(),

    }


# =====================================================
# Shutdown
# =====================================================

def close(self):

    try:

        self.connection.close()

        logger.info(
            "MemoryService shutdown."
        )

    except Exception as e:

        logger.exception(e)
