"""Pipeline RAG: retrieve -> (kiểm tra fallback) -> sinh câu trả lời kèm trích dẫn.

Chạy thử nhanh trên terminal:
    python -m app.rag "Câu hỏi của bạn ở đây"
"""
from __future__ import annotations

import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_llm
from app.retriever import RetrievalResult, RetrievedChunk, retrieve
from config import settings

FALLBACK_MESSAGE = "Không tìm thấy thông tin trong tài liệu để trả lời câu hỏi này."

REWRITE_PROMPT = """Viết lại tin nhắn của người dùng thành MỘT truy vấn tìm kiếm ngắn để tra cứu trong tài liệu kỹ thuật.
- Bỏ cụm chỉ thị / xã giao ("hãy", "giải thích dễ hiểu", "tóm tắt giúp", "cho tôi biết"...), chỉ giữ chủ đề + từ khoá kỹ thuật.
- Dịch sang tiếng Anh (tài liệu kỹ thuật thường bằng tiếng Anh).
- Nếu tin nhắn đã là truy vấn tốt thì giữ gần như nguyên.
Chỉ trả về truy vấn, không thêm gì khác.

Tin nhắn: {q}
Truy vấn:"""

SYSTEM_PROMPT = """Bạn là trợ lý hỏi-đáp trên tài liệu kỹ thuật.
Chỉ được trả lời DỰA HOÀN TOÀN vào phần "NGỮ CẢNH" bên dưới.
Nếu ngữ cảnh không chứa đủ thông tin để trả lời, hãy trả lời CHÍNH XÁC câu sau và không thêm gì khác:
"{fallback}"
Khi dùng thông tin từ ngữ cảnh, trích dẫn nguồn ngay sau ý đó theo dạng [nguồn: <tên file> tr.<số trang>].
Trả lời chính xác, bằng tiếng Việt, không lan man.

Định dạng (giao diện hiển thị văn bản thuần, KHÔNG render markdown):
- Nội dung có nhiều bước / quy trình / phân loại / liệt kê -> trình bày dạng danh sách
  đánh số hoặc gạch đầu dòng, có mục con thụt lề khi cần.
- Câu hỏi định nghĩa hoặc hỏi một ý ngắn -> trả lời gọn 1-3 câu, KHÔNG ép thành danh sách.
- KHÔNG dùng ký hiệu markdown (**, ##, `). Muốn nhấn mạnh thì viết hoa hoặc dùng dấu hai chấm."""

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
    search_query: str = ""  # truy vấn thực tế dùng để retrieve (sau khi viết lại)


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


_INSTRUCTION_RE = re.compile(
    r"\b(hãy|giải thích|diễn giải|tóm tắt|liệt kê|trình bày|cho tôi biết|mô tả|"
    r"so sánh|khái quát|tổng hợp|summar|explain|describe|list all|overview)\b",
    re.IGNORECASE,
)


def _needs_rewrite(question: str) -> bool:
    """Chỉ viết lại khi cần: câu không phải tiếng Anh, hoặc có kèm cụm chỉ thị.

    Câu hỏi tiếng Anh gọn -> tìm thẳng, tránh làm lệch retrieval (giảm groundedness).
    """
    return not question.isascii() or bool(_INSTRUCTION_RE.search(question))


def _rewrite_query(question: str) -> str:
    """Chuẩn hoá câu hỏi thành truy vấn tìm kiếm (bỏ chỉ thị, dịch sang tiếng Anh).

    Lỗi hoặc kết quả rỗng -> dùng lại câu gốc.
    """
    try:
        out = get_llm().invoke(REWRITE_PROMPT.format(q=question)).content
        out = (out if isinstance(out, str) else str(out)).strip()
        return out or question
    except Exception:  # noqa: BLE001 - rewrite hỏng thì lùi về câu gốc, không chặn luồng
        return question


_SUMMARY_RE = re.compile(
    r"(tóm tắt|tóm lược|summar|mục lục|khái quát|tổng quan|tổng quát|overview|"
    r"nội dung chính|ý chính|liệt kê (toàn bộ|tất cả|hết)|toàn bộ (tài liệu|nội dung)|"
    r"cả (tài liệu|file|quyển|cuốn))",
    re.IGNORECASE,
)


def _summary_intent(question: str) -> bool:
    """Câu hỏi có phải yêu cầu tóm tắt / thao tác trên toàn bộ tài liệu không."""
    return bool(_SUMMARY_RE.search(question))


def answer_question(
    question: str,
    k: int | None = None,
    source: str | None = None,
    force_summary: bool = False,
) -> Answer:
    t0 = time.perf_counter()
    # Chuẩn hoá Unicode: input từ HTTP có thể ở dạng NFD, làm hỏng match regex tiếng Việt.
    question = unicodedata.normalize("NFC", question)

    # --- Chế độ tóm tắt toàn tài liệu (không đi qua retrieval top-k) ---
    if force_summary or _summary_intent(question):
        from app.summarize import summarize_document
        from app.vectorstore import list_sources

        target = source or None
        if target is None:
            names = [s["source"] for s in list_sources()]
            if len(names) == 1:
                target = names[0]
            elif not names:
                return Answer(
                    question=question,
                    answer="Chưa có tài liệu nào trong hệ thống. Hãy đính kèm một file PDF trước.",
                    is_fallback=True,
                    confidence=0.0,
                    chunks=[],
                    latency_s=time.perf_counter() - t0,
                )
            else:
                return Answer(
                    question=question,
                    answer=(
                        "Có nhiều tài liệu — hãy chọn một tài liệu ở ô \"Phạm vi\" "
                        "rồi hỏi lại để tôi tóm tắt đúng tài liệu đó."
                    ),
                    is_fallback=True,
                    confidence=0.0,
                    chunks=[],
                    latency_s=time.perf_counter() - t0,
                )
        return summarize_document(target, question)

    search_query = (
        _rewrite_query(question)
        if settings.enable_query_rewrite and _needs_rewrite(question)
        else question
    )
    result: RetrievalResult = retrieve(search_query, k=k, source=source)

    # --- Fallback tầng 1: retrieval không đủ tự tin -> không gọi LLM, khỏi tốn tiền ---
    if not result.is_confident:
        return Answer(
            question=question,
            answer=FALLBACK_MESSAGE,
            is_fallback=True,
            confidence=result.confidence,
            chunks=result.chunks,
            latency_s=time.perf_counter() - t0,
            search_query=search_query,
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
        search_query=search_query,
    )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Câu hỏi: ")
    a = answer_question(q)
    print("\n" + a.answer + "\n" + "-" * 40)
    if a.search_query and a.search_query.lower() != q.lower():
        print(f"tìm với: {a.search_query}")
    print(
        f"confidence={a.confidence:.2f}  fallback={a.is_fallback}  "
        f"latency={a.latency_s:.2f}s"
    )
    if a.sources:
        print("Nguồn:", "; ".join(a.sources))
