"""Retrieve the most relevant chunks and score how confident the match is."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from app.vectorstore import get_vectorstore
from config import settings


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    score: float  # 0..1, where a higher value means a stronger match.


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]

    @property
    def confidence(self) -> float:
        """Return the strongest relevance score in the current top-k list."""
        return max((c.score for c in self.chunks), default=0.0)

    @property
    def is_confident(self) -> bool:
        """Return True only when the retrieval quality is high enough to trust the answer."""
        return len(self.chunks) > 0 and self.confidence >= settings.confidence_threshold


def retrieve(
    question: str, k: int | None = None, source: str | None = None
) -> RetrievalResult:
    """Return the top-k matching chunks, optionally restricted to one source file."""
    k = k or settings.top_k
    vs = get_vectorstore()
    pairs: list[tuple[Document, float]] = vs.similarity_search_with_relevance_scores(
        question, k=k, filter={"source": source} if source else None
    )
    chunks = [
        RetrievedChunk(
            text=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            score=max(0.0, min(1.0, float(score))),
        )
        for doc, score in pairs
    ]
    return RetrievalResult(chunks=chunks)
