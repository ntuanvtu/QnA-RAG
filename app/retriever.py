"""Truy hồi chunk liên quan + tính điểm confidence cho câu hỏi."""
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
    score: float  # 0..1, càng cao càng liên quan


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]

    @property
    def confidence(self) -> float:
        """Điểm liên quan cao nhất trong top-k (0 nếu không có chunk nào)."""
        return max((c.score for c in self.chunks), default=0.0)

    @property
    def is_confident(self) -> bool:
        """Đủ tự tin để gọi LLM trả lời hay không."""
        return len(self.chunks) > 0 and self.confidence >= settings.confidence_threshold


def retrieve(question: str, k: int | None = None) -> RetrievalResult:
    k = k or settings.top_k
    vs = get_vectorstore()
    # relevance score đã quy về [0, 1] (1 = giống nhất) nhờ cosine space.
    pairs: list[tuple[Document, float]] = vs.similarity_search_with_relevance_scores(
        question, k=k
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
