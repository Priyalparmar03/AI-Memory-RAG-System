"""
services/llm_service.py

Unified LLM Service

Supports:
- Google Gemini
- OpenAI
- Future Providers

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional

import google.generativeai as genai
from openai import OpenAI

from app.config import Config

logger = logging.getLogger(__name__)


# ============================================================
# Base Provider
# ============================================================

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> str:
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        pass


# ============================================================
# Gemini Provider
# ============================================================

class GeminiProvider(BaseLLMProvider):

    def __init__(self):

        genai.configure(api_key=Config.GEMINI_API_KEY)

        self.model = genai.GenerativeModel(
            Config.DEFAULT_MODEL
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> str:

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature
            }
        )

        return response.text

    def stream(
        self,
        prompt: str,
        temperature: float = 0.3,
    ):

        response = self.model.generate_content(
            prompt,
            stream=True,
            generation_config={
                "temperature": temperature
            }
        )

        for chunk in response:

            if chunk.text:

                yield chunk.text


# ============================================================
# OpenAI Provider
# ============================================================

class OpenAIProvider(BaseLLMProvider):

    def __init__(self):

        self.client = OpenAI(
            api_key=Config.OPENAI_API_KEY
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> str:

        response = self.client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content

    def stream(
        self,
        prompt: str,
        temperature: float = 0.3,
    ):

        response = self.client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            stream=True,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        for chunk in response:

            delta = chunk.choices[0].delta.content

            if delta:

                yield delta


# ============================================================
# Main LLM Service
# ============================================================

class LLMService:
    """
    Unified interface for all providers.
    """

    PROVIDERS = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
    }

    @classmethod
    def get_provider(
        cls,
        provider: str,
    ) -> BaseLLMProvider:

        provider = provider.lower()

        if provider not in cls.PROVIDERS:

            raise ValueError(
                f"Unsupported provider: {provider}"
            )

        return cls.PROVIDERS[provider]()

    @classmethod
    def generate(
        cls,
        prompt: str,
        provider: str = "gemini",
        temperature: float = 0.3,
    ) -> str:

        logger.info(
            "Calling %s model",
            provider,
        )

        llm = cls.get_provider(provider)

        return llm.generate(
            prompt=prompt,
            temperature=temperature,
        )

    @classmethod
    def stream(
        cls,
        prompt: str,
        provider: str = "gemini",
        temperature: float = 0.3,
    ):

        llm = cls.get_provider(provider)

        yield from llm.stream(
            prompt=prompt,
            temperature=temperature,
        )