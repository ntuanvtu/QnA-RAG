"""Pipeline RAG: retrieve -> (kiểm tra fallback) -> sinh câu trả lời kèm trích dẫn.

Chạy thử nhanh trên terminal:
    python -m app.rag "Câu hỏi của bạn ở đây"
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_llm
from app.retriever import RetrievalResult, RetrievedChunk, retrieve

FALLBACK_MESSAGE = "Không tìm thấy thông tin trong tài liệu để trả lời câu hỏi này."

SYSTEM_PROMPT = """Bạn là trợ lý hỏi-đáp trên tài liệu kỹ thuật.
Chỉ được trả lời DỰA HOÀN TOÀN vào phần "NGỮ CẢNH" bên dưới.
Nếu ngữ cảnh không chứa đủ thông tin để trả lời, hãy trả lời CHÍNH XÁC câu sau và không thêm gì khác:
"{fallback}"
Khi dùng thông tin từ ngữ cảnh, trích dẫn nguồn ngay sau ý đó theo dạng [nguồn: <tên file> tr.<số trang>].
Trả lời ngắn gọn, chính xác, bằng tiếng Việt."""

USER_PROMPT = """NGỮ CẢNH:
{context}

CÂU HỎI: {question}

Trả lời:"""


@dataclass
class Answer:
    question: str
    answer: str
    is_fallback: bool
    confidence: float
    chunks: list[RetrievedChunk]
    latency_s: float
    sources: list[str] = field(default_factory=list)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        loc = c.source + (f" tr.{c.page}" if c.page else "")
        blocks.append(f"[{i}] ({loc})\n{c.text}")
    return "\n\n".join(blocks)


def _unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    seen: list[str] = []
    for c in chunks:
        s = c.source + (f" tr.{c.page}" if c.page else "")
        if s not in seen:
            seen.append(s)
    return seen


def answer_question(question: str, k: int | None = None) -> Answer:
    t0 = time.perf_counter()
    result: RetrievalResult = retrieve(question, k=k)

    # --- Fallback tầng 1: retrieval không đủ tự tin -> không gọi LLM, khỏi tốn tiền ---
    if not result.is_confident:
        return Answer(
            question=question,
            answer=FALLBACK_MESSAGE,
            is_fallback=True,
            confidence=result.confidence,
            chunks=result.chunks,
            latency_s=time.perf_counter() - t0,
        )

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
    )
    messages = prompt.format_messages(
        fallback=FALLBACK_MESSAGE,
        context=_format_context(result.chunks),
        question=question,
    )
    response = get_llm().invoke(messages)
    text = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    ).strip()

    # --- Fallback tầng 2: LLM tự nhận không đủ thông tin ---
    is_fb = FALLBACK_MESSAGE[:35].lower() in text.lower()

    return Answer(
        question=question,
        answer=text,
        is_fallback=is_fb,
        confidence=result.confidence,
        chunks=result.chunks,
        latency_s=time.perf_counter() - t0,
        sources=[] if is_fb else _unique_sources(result.chunks),
    )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Câu hỏi: ")
    a = answer_question(q)
    print("\n" + a.answer + "\n" + "-" * 40)
    print(
        f"confidence={a.confidence:.2f}  fallback={a.is_fallback}  "
        f"latency={a.latency_s:.2f}s"
    )
    if a.sources:
        print("Nguồn:", "; ".join(a.sources))
