from __future__ import annotations
from datetime import datetime
from typing import Dict, List
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional
import time
from typing import Generator
from services.llm_service import LLMService
from services.memory_service import MemoryService
from services.prompt_service import PromptService
from services.rag_service import RagService
from decimal import Decimal
from typing import List


logger = logging.getLogger(__name__)

MAX_RETRIES = 3
DEFAULT_MAX_TOKENS = 4000
DEFAULT_HISTORY_LIMIT = 10

class ChatService:
    """
    Main orchestration service.

    Flow

        User
          │
          ▼
    Memory Service
          │
          ▼
     RAG Retrieval
          │
          ▼
    Prompt Builder
          │
          ▼
      LLM Service
          │
          ▼
     Save Response
          │
          ▼
        Return
    """

    def __init__(
        self,
        llm_service: LLMService,
        memory_service: MemoryService,
        rag_service: RagService,
        prompt_service: PromptService,
    ):

        self.llm = llm_service
        self.memory = memory_service
        self.rag = rag_service
        self.prompt = prompt_service

        logger.info("ChatService initialized.")

    # =====================================================
    # Conversation
    # =====================================================

    def create_conversation(
        self,
        user_id: str,
        title: str = "New Conversation",
    ) -> str:

        return self.memory.create_conversation(
            user_id=user_id,
            title=title,
        )

    # =====================================================
    # Conversation Exists
    # =====================================================

    def conversation_exists(
        self,
        conversation_id: str,
    ) -> bool:

        return (
            self.memory.get_conversation(
                conversation_id
            )
            is not None
        )

    # =====================================================
    # Chat
    # =====================================================

    def chat(
        self,
        user_id: str,
        message: str,
        conversation_id: Optional[str] = None,
        use_rag: bool = True,
    ) -> Dict:
        """
        Main chat pipeline.
        """

        # ----------------------------------------
        # Create conversation if needed
        # ----------------------------------------

        if conversation_id is None:

            conversation_id = self.create_conversation(
                user_id=user_id
            )

        # ----------------------------------------
        # Save user message
        # ----------------------------------------

        self.memory.add_message(
            conversation_id,
            "user",
            message,
        )

        # ----------------------------------------
        # Retrieve memory
        # ----------------------------------------

        memory = self.memory.prompt_memory(
            conversation_id
        )

        # ----------------------------------------
        # Retrieve RAG context
        # ----------------------------------------

        rag_context = None

        if use_rag:

            rag_context = self.rag.retrieve(
                query=message,
            )

        # ----------------------------------------
        # Build prompt
        # ----------------------------------------

        prompt = self.prompt.build_chat_prompt(
            query=message,
            memory=memory,
            rag=rag_context,
        )

        # ----------------------------------------
        # Generate response
        # ----------------------------------------

        response = self.llm.generate(
            prompt
        )

        # ----------------------------------------
        # Save assistant response
        # ----------------------------------------

        self.memory.add_message(
            conversation_id,
            "assistant",
            response,
        )

        # ----------------------------------------
        # Store long-term memory
        # ----------------------------------------

        self.memory.store_memory(
            user_id=user_id,
            conversation_id=conversation_id,
            content=message,
        )

        return {

            "conversation_id": conversation_id,

            "response": response,

            "memory": memory,

            "rag": rag_context,

            "timestamp": datetime.utcnow().isoformat(),

        }

    # =====================================================
    # Get Conversation
    # =====================================================

    def conversation(
        self,
        conversation_id: str,
    ) -> Dict:

        return {

            "conversation":

                self.memory.get_conversation(
                    conversation_id
                ),

            "messages":

                self.memory.get_messages(
                    conversation_id
                ),

        }

    # =====================================================
    # Delete Conversation
    # =====================================================

    def delete_conversation(
        self,
        conversation_id: str,
    ):

        self.memory.delete_conversation(
            conversation_id
        )

    # =====================================================
    # Health
    # =====================================================

    def health(self):

        return {

            "chat": "healthy",

            "llm":

                self.llm.health(),

            "memory":

                self.memory.health(),

            "rag":

                self.rag.health(),

        }

    # =====================================================
    # Info
    # =====================================================

    def info(self):

        return {

            "service": "ChatService",

            "version": "1.0.0",

            "llm":

                self.llm.info(),

            "memory":

                self.memory.info(),

            "rag":

                self.rag.info(),

        }
# =====================================================
# Generate with Retry
# =====================================================

def _generate_with_retry(
    self,
    prompt: str,
    retries: int = MAX_RETRIES,
) -> str:
    """
    Generate a response with retry logic.
    """

    last_exception = None

    for attempt in range(retries):

        try:

            return self.llm.generate(prompt)

        except Exception as exc:

            last_exception = exc

            logger.warning(
                "LLM generation failed (attempt %d/%d): %s",
                attempt + 1,
                retries,
                exc,
            )

            time.sleep(2 ** attempt)

    raise RuntimeError(
        "Failed to generate response after retries."
    ) from last_exception


# =====================================================
# Stream Chat
# =====================================================

def stream_chat(
    self,
    user_id: str,
    message: str,
    conversation_id: Optional[str] = None,
    use_rag: bool = True,
) -> Generator[str, None, None]:
    """
    Stream tokens from the LLM.
    """

    if conversation_id is None:

        conversation_id = self.create_conversation(
            user_id=user_id
        )

    self.memory.add_message(
        conversation_id,
        "user",
        message,
    )

    memory = self.memory.prompt_memory(
        conversation_id
    )

    rag = None

    if use_rag:

        rag = self.rag.retrieve(message)

    prompt = self.prompt.build_chat_prompt(
        query=message,
        memory=memory,
        rag=rag,
    )

    full_response = []

    for chunk in self.llm.stream(prompt):

        full_response.append(chunk)

        yield chunk

    assistant_reply = "".join(full_response)

    self.memory.add_message(
        conversation_id,
        "assistant",
        assistant_reply,
    )

# =====================================================
# Generate Conversation Title
# =====================================================

def generate_title(
    self,
    conversation_id: str,
):
    """
    Generate a short conversation title.
    """

    messages = self.memory.recent_messages(
        conversation_id,
        limit=3,
    )

    if not messages:

        return

    text = "\n".join(

        m["content"]

        for m in messages

    )

    prompt = f"""
Generate a concise title (maximum 6 words)
for this conversation.

{text}
"""

    title = self._generate_with_retry(
        prompt
    )

    self.memory.rename_conversation(
        conversation_id,
        title.strip(),
    )

# =====================================================
# Optimize Context
# =====================================================

def optimize_context(
    self,
    conversation_id: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):

    return self.memory.context_window(
        conversation_id,
        max_tokens=max_tokens,
    )

# =====================================================
# Conversation Summary
# =====================================================

def summarize(
    self,
    conversation_id: str,
):

    history = self.memory.build_context(
        conversation_id,
        limit=20,
    )

    prompt = self.prompt.build_summary_prompt(
        history
    )

    return self._generate_with_retry(
        prompt
    )

# =====================================================
# Prune Conversation
# =====================================================

def prune(
    self,
    conversation_id: str,
):

    self.memory.prune_history(
        conversation_id
    )


# =====================================================
# Token Usage
# =====================================================

def token_usage(
    self,
    conversation_id: str,
):

    return {

        "tokens":

            self.memory.conversation_tokens(
                conversation_id
            ),

        "budget":

            DEFAULT_MAX_TOKENS,

    }


# =====================================================
# Conversation Overview
# =====================================================

def conversation_overview(
    self,
    conversation_id: str,
):

    return {

        "conversation":

            self.memory.get_conversation(
                conversation_id
            ),

        "summary":

            self.summarize(
                conversation_id
            ),

        "token_usage":

            self.token_usage(
                conversation_id
            ),

    }


# =====================================================
# Conversation Analytics
# =====================================================

def analytics(
    self,
    conversation_id: str,
):

    messages = self.memory.get_messages(
        conversation_id
    )

    users = 0
    assistants = 0

    characters = 0

    for message in messages:

        characters += len(message["content"])

        if message["role"] == "user":

            users += 1

        elif message["role"] == "assistant":

            assistants += 1

    tokens = self.memory.conversation_tokens(
        conversation_id
    )

    return {

        "conversation_id": conversation_id,

        "messages": len(messages),

        "user_messages": users,

        "assistant_messages": assistants,

        "characters": characters,

        "tokens": tokens,

    }

# =====================================================
# Cost Estimation
# =====================================================

def estimate_cost(
    self,
    conversation_id: str,
    price_per_1k_tokens: float = 0.01,
):

    tokens = self.memory.conversation_tokens(
        conversation_id
    )

    cost = (

        Decimal(tokens)

        / Decimal(1000)

    ) * Decimal(price_per_1k_tokens)

    return {

        "tokens": tokens,

        "estimated_cost": float(

            round(cost, 4)

        ),

    }


# =====================================================
# Search Conversation
# =====================================================

def search_messages(
    self,
    conversation_id: str,
    keyword: str,
):

    messages = self.memory.get_messages(
        conversation_id
    )

    results = []

    keyword = keyword.lower()

    for message in messages:

        if keyword in message["content"].lower():

            results.append(message)

    return results

# =====================================================
# JSON Chat
# =====================================================

def json_chat(
    self,
    user_id: str,
    message: str,
    conversation_id=None,
):

    response = self.chat(

        user_id=user_id,

        message=message,

        conversation_id=conversation_id,

    )

    return {

        "status": "success",

        "conversation_id":

            response["conversation_id"],

        "response":

            response["response"],

        "timestamp":

            response["timestamp"],

    }

# =====================================================
# Conversation Feedback
# =====================================================

def feedback(
    self,
    conversation_id: str,
    rating: int,
    comment: str = "",
):

    return {

        "conversation_id":

            conversation_id,

        "rating":

            rating,

        "comment":

            comment,

        "saved": True,

    }

# =====================================================
# Switch Model
# =====================================================

def switch_model(
    self,
    provider: str,
    api_key: str,
    model: str,
):

    self.llm = LLMService(

        provider=provider,

        api_key=api_key,

        model=model,

    )

    logger.info(

        "Switched LLM model to %s",

        model,

    )

# =====================================================
# Available Models
# =====================================================

def models(self):

    return self.llm.info()

# =====================================================
# Status
# =====================================================

def status(self):

    return {

        "llm":

            self.llm.health(),

        "memory":

            self.memory.health(),

        "rag":

            self.rag.health(),

        "chat":

            "healthy",

    }


# =====================================================
# Tool Dispatcher
# =====================================================

def execute_tool(
    self,
    tool_name: str,
    parameters: dict,
):
    """
    Placeholder for tool execution.

    Extend this method by registering
    tools in a mapping instead of
    using long if/elif chains.
    """

    raise NotImplementedError(
        "Tool execution is not implemented yet."
    )

# =====================================================
# Export Conversation
# =====================================================

def export_conversation(
    self,
    conversation_id: str,
) -> Dict:
    """
    Export a conversation with metadata.
    """

    conversation = self.memory.get_conversation(
        conversation_id
    )

    messages = self.memory.get_messages(
        conversation_id
    )

    analytics = self.analytics(
        conversation_id
    )

    return {

        "conversation": conversation,

        "messages": messages,

        "analytics": analytics,

        "exported_at": datetime.utcnow().isoformat(),

    }
# =====================================================
# Export User Conversations
# =====================================================

def export_user(
    self,
    user_id: str,
) -> Dict:

    conversations = self.memory.list_conversations(
        user_id
    )

    data = []

    for conversation in conversations:

        data.append(

            self.export_conversation(

                conversation["conversation_id"]

            )

        )

    return {

        "user_id": user_id,

        "conversation_count": len(data),

        "conversations": data,

    }

# =====================================================
# Clear Conversation
# =====================================================

def clear_conversation(
    self,
    conversation_id: str,
):

    self.memory.delete_conversation(
        conversation_id
    )

    logger.info(

        "Conversation cleared."

    )

# =====================================================
# Conversation Summary
# =====================================================

def conversation_summary(
    self,
    conversation_id: str,
):

    summary = self.summarize(
        conversation_id
    )

    analytics = self.analytics(
        conversation_id
    )

    return {

        "summary": summary,

        "analytics": analytics,

    }

# =====================================================
# Diagnostics
# =====================================================

def diagnostics(self):

    return {

        "chat": "healthy",

        "llm":

            self.llm.info(),

        "memory":

            self.memory.info(),

        "rag":

            self.rag.info(),

    }

# =====================================================
# Service Information
# =====================================================

def info(self):

    return {

        "service":

            "ChatService",

        "version":

            "1.0.0",

        "llm":

            self.llm.info(),

        "memory":

            self.memory.info(),

        "rag":

            self.rag.info(),

        "started":

            datetime.utcnow().isoformat(),

    }

# =====================================================
# Health
# =====================================================

def health(self):

    return {

        "status":

            "healthy",

        "services": {

            "llm":

                self.llm.health(),

            "memory":

                self.memory.health(),

            "rag":

                self.rag.health(),

        },

    }

# =====================================================
# Cleanup
# =====================================================

def cleanup(self):

    self.memory.cleanup_conversations()

    self.memory.cleanup_memories()

    logger.info(

        "Cleanup complete."

    )

# =====================================================
# Reset Chat Service
# =====================================================

def reset(self):

    self.memory.reset()

    self.rag.clear()

    logger.warning(

        "ChatService reset."

    )

# =====================================================
# Shutdown
# =====================================================

def close(self):

    try:

        self.memory.close()

    finally:

        logger.info(

            "ChatService shutdown."

        )
