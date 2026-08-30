"""Chế độ tóm tắt / thao tác trên TOÀN BỘ một tài liệu (map-reduce).

Khác với RAG thường (chỉ lấy top-k chunk), ở đây lấy hết chunk của 1 file:
  map    : tóm tắt từng lô chunk
  reduce : gộp các bản tóm tắt lô thành một bản cuối

Dùng khi người dùng hỏi kiểu "tóm tắt tài liệu", "liệt kê toàn bộ...", "mục lục".
"""
from __future__ import annotations

import time

from app.llm import get_llm
from app.retriever import RetrievedChunk
from app.vectorstore import get_vectorstore
from config import settings

_MAP_PROMPT = """Tóm tắt các đoạn trích sau (theo thứ tự) từ tài liệu "{source}".
Giữ ý chính, thuật ngữ, số liệu quan trọng. Viết tiếng Việt, gọn, KHÔNG dùng markdown.
{focus}
ĐOẠN TRÍCH:
{text}

Tóm tắt phần này:"""

_REDUCE_PROMPT = """Dưới đây là các bản tóm tắt từng phần của tài liệu "{source}" (theo thứ tự đọc).
Tổng hợp thành MỘT bản tóm tắt mạch lạc, đủ ý, không lặp. Viết tiếng Việt, KHÔNG dùng markdown;
nội dung có tính quy trình / phân loại thì trình bày dạng danh sách đánh số, mục con thụt lề.
{focus}{note}
CÁC BẢN TÓM TẮT PHẦN:
{parts}

Bản tóm tắt tổng hợp:"""


def _chunks_of_source(source: str) -> list[tuple[int, str]]:
    """(page, text) của mọi chunk thuộc file, sắp theo trang."""
    got = get_vectorstore().get(where={"source": source}, include=["metadatas", "documents"])
    pairs = [
        (int(m.get("page", 0) or 0), doc)
        for m, doc in zip(got["metadatas"], got["documents"])
    ]
    return sorted(pairs, key=lambda p: p[0])


def _batches(pairs: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Gộp chunk liền nhau thành lô ~summary_batch_chars ký tự."""
    out: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    size = 0
    for page, text in pairs:
        cur.append((page, text))
        size += len(text)
        if size >= settings.summary_batch_chars:
            out.append(cur)
            cur, size = [], 0
    if cur:
        out.append(cur)
    return out


def _evenly_sample(items: list, n: int) -> list:
    """Lấy n phần tử trải đều trên danh sách (giữ nguyên thứ tự)."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _focus_line(question: str | None) -> str:
    q = (question or "").strip()
    return f'Người dùng hỏi cụ thể: "{q}" — ưu tiên làm rõ trọng tâm đó.\n' if q else ""


def summarize_document(source: str, question: str | None = None):
    """Tóm tắt toàn bộ `source`. Trả về app.rag.Answer."""
    from app.rag import Answer  # tránh vòng import

    t0 = time.perf_counter()
    pairs = _chunks_of_source(source)
    if not pairs:
        return Answer(
            question=question or f"Tóm tắt {source}",
            answer=f'Không có nội dung nào của "{source}" trong hệ thống.',
            is_fallback=True,
            confidence=0.0,
            chunks=[],
            latency_s=time.perf_counter() - t0,
        )

    batches = _batches(pairs)
    note = ""
    if len(batches) > settings.summary_max_batches:
        batches = _evenly_sample(batches, settings.summary_max_batches)
        note = (
            f"\nLƯU Ý: tài liệu dài, bản tóm tắt dựa trên {len(batches)} phần trải đều "
            "trên toàn tài liệu (không phải toàn văn)."
        )

    llm = get_llm()
    focus = _focus_line(question)
    partials: list[str] = []
    for batch in batches:
        text = "\n\n".join(t for _, t in batch)
        resp = llm.invoke(
            _MAP_PROMPT.format(source=source, focus=focus, text=text)
        ).content
        partials.append((resp if isinstance(resp, str) else str(resp)).strip())

    if len(partials) == 1:
        final = partials[0]
    else:
        resp = llm.invoke(
            _REDUCE_PROMPT.format(
                source=source,
                focus=focus,
                note=note,
                parts="\n\n---\n\n".join(partials),
            )
        ).content
        final = (resp if isinstance(resp, str) else str(resp)).strip()

    if note and note.strip() not in final:
        final = f"{final}\n\n{note.strip()}"

    pages = sorted({p for p, _ in pairs})
    return Answer(
        question=question or f"Tóm tắt {source}",
        answer=final,
        is_fallback=False,
        confidence=1.0,
        chunks=[
            RetrievedChunk(text=t, source=source, page=p, score=1.0)
            for p, t in pairs[:1]
        ],
        latency_s=time.perf_counter() - t0,
        sources=[f"{source} (toàn tài liệu, tr.{pages[0]}–{pages[-1]})"] if pages else [source],
    )
