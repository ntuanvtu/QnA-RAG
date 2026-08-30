"""Local sentence-transformer embeddings used for document indexing and retrieval.

The model is downloaded once on first use and then reused, which keeps indexing
free and avoids repeated initialization overhead.
"""
from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding client, creating it once and reusing it."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings
