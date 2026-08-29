"""Embedding model chạy local (sentence-transformers).

Model được tải về máy lần đầu chạy (~90MB cho all-MiniLM-L6-v2), sau đó cache lại.
Không tốn tiền, không cần API key.
"""
from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Trả về instance embedding dùng chung (khởi tạo 1 lần)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            # Chuẩn hoá vector -> khoảng cách cosine ổn định trong [0, 1]
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings
