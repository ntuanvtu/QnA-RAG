"""Central configuration for the application.

Values are resolved in this order: environment variables > .env file > defaults
defined here. This keeps optimization and runtime settings in one place.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Project-wide runtime settings for retrieval, summary, and LLM behavior."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM configuration
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0

    # Local embeddings run without paid API calls.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking strategy
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # Sparse slide decks produce very short pages; merging nearby pages keeps the
    # chunk size meaningful and prevents noisy retrieval from tiny fragments.
    sparse_median_threshold: int = 700
    sparse_block_chars: int = 600

    # Retrieval tuning
    top_k: int = 4
    # A small query rewrite adds a second LLM call but can improve matching on
    # non-English or instruction-heavy questions.
    enable_query_rewrite: bool = True

    # Whole-document summary mode
    summary_batch_chars: int = 6000
    summary_max_batches: int = 10
    # Retrieval confidence is normalized to [0, 1]. Falling back early prevents
    # hallucinated answers when the evidence is weak.
    confidence_threshold: float = 0.35

    # Storage and runtime paths
    data_dir: Path = BASE_DIR / "data"
    vector_dir: Path = BASE_DIR / "storage" / "chroma"
    static_dir: Path = BASE_DIR / "app" / "static"
    collection_name: str = "documents"


settings = Settings()
