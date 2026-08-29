"""Khởi tạo client Claude (qua LangChain)."""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from config import settings

_llm: ChatAnthropic | None = None


def get_llm() -> ChatAnthropic:
    """Trả về client Claude dùng chung. Báo lỗi rõ ràng nếu thiếu API key."""
    global _llm
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "Chưa có ANTHROPIC_API_KEY. Tạo file .env (xem .env.example) và điền key."
        )
    if _llm is None:
        _llm = ChatAnthropic(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            api_key=settings.anthropic_api_key,
        )
    return _llm
