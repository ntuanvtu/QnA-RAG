"""Cấu hình tập trung cho toàn bộ ứng dụng.

Giá trị đọc theo thứ tự ưu tiên: biến môi trường > file .env > mặc định ở đây.
Khi tune hệ thống (đổi ngưỡng, số chunk, model...) chỉ cần sửa ở một chỗ này.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM (Anthropic Claude) ---
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0

    # --- Embedding: chạy local bằng sentence-transformers, KHÔNG cần API key ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval ---
    top_k: int = 4
    # Điểm similarity nằm trong [0, 1] (1 = giống nhất).
    # Nếu điểm cao nhất của top-k < ngưỡng này -> trả lời fallback, không gọi LLM.
    confidence_threshold: float = 0.35

    # --- Đường dẫn ---
    data_dir: Path = BASE_DIR / "data"
    vector_dir: Path = BASE_DIR / "storage" / "chroma"
    collection_name: str = "documents"


settings = Settings()
