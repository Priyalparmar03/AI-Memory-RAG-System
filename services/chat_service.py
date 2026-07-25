"""
services/chat_service.py

Production Chat Service

Responsibilities
----------------
- Chat orchestration
- Conversation lifecycle
- Memory retrieval
- RAG retrieval
- Prompt construction
- LLM interaction
- Response persistence
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Dict, Optional

from services.llm_service import LLMService
from services.memory_service import MemoryService
from services.prompt_service import PromptService
from services.rag_service import RagService

logger = logging.getLogger(__name__)


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
