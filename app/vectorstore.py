"""Vector database (Chroma) - lưu embedding của các chunk xuống đĩa."""
from __future__ import annotations

from langchain_chroma import Chroma

from app.embeddings import get_embeddings
from config import settings


def get_vectorstore() -> Chroma:
    """Mở (hoặc tạo mới) collection Chroma đã persist trong storage/chroma."""
    settings.vector_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.vector_dir),
        # Dùng cosine để điểm relevance quy về [0, 1] (1 - cosine_distance)
        collection_metadata={"hnsw:space": "cosine"},
    )
