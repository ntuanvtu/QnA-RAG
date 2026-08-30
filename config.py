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
    # Tài liệu kiểu trình chiếu (slide): text thưa, rời rạc theo trang.
    # Nếu số ký tự TRUNG VỊ mỗi trang < ngưỡng -> gộp các trang liền nhau thành
    # khối >= sparse_block_chars trước khi chunk (đỡ rời rạc, embed có nghĩa hơn).
    sparse_median_threshold: int = 700
    sparse_block_chars: int = 600

    # --- Retrieval ---
    top_k: int = 4
    # Trước khi tìm: nhờ Claude viết lại câu hỏi thành truy vấn tìm kiếm
    # (bỏ cụm chỉ thị, dịch sang tiếng Anh). Tốn thêm 1 lần gọi LLM mỗi câu.
    enable_query_rewrite: bool = True

    # --- Chế độ tóm tắt / thao tác toàn tài liệu (map-reduce) ---
    summary_batch_chars: int = 6000     # gộp chunk thành lô ~ngần này ký tự để "map"
    summary_max_batches: int = 10       # tài liệu dài hơn -> lấy mẫu đều tối đa ngần này lô
    # Điểm similarity nằm trong [0, 1] (1 = giống nhất).
    # Nếu điểm cao nhất của top-k < ngưỡng này -> trả lời fallback, không gọi LLM.
    confidence_threshold: float = 0.35

    # --- Đường dẫn ---
    data_dir: Path = BASE_DIR / "data"
    vector_dir: Path = BASE_DIR / "storage" / "chroma"
    static_dir: Path = BASE_DIR / "app" / "static"
    collection_name: str = "documents"


settings = Settings()
