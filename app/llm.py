"""Shared Claude client wrapper for the generation stage of the RAG pipeline."""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from config import settings

_llm: ChatAnthropic | None = None


def get_llm() -> ChatAnthropic:
    """Return the shared Claude client and fail fast when the API key is absent."""
    global _llm
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured. Add it to the .env file referenced by .env.example."
        )
    if _llm is None:
        _llm = ChatAnthropic(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            api_key=settings.anthropic_api_key,
        )
    return _llm
