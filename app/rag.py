"""RAG pipeline: retrieve evidence, enforce fallback rules, and generate grounded answers.

This module is the single entry point for CLI, API, and evaluation flows.
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

FALLBACK_MESSAGE = "No relevant information was found in the document to answer this question."

REWRITE_PROMPT = """Rewrite the user's message into a short search query for technical-document retrieval.
- Remove conversational filler and instruction phrases such as "please", "explain slowly", "summarize", "tell me about".
- Keep the technical topic and keywords.
- Translate to English if the original question is in another language.
- Return only the rewritten query.

User message: {q}
Search query:"""

SYSTEM_PROMPT = """You are a technical-document Q&A assistant.
Only answer from the CONTEXT below.
If the context does not contain enough information, reply with exactly this fallback:
"{fallback}"
When you cite information from the context, append the source immediately after the statement in the format [source: <file> p.<page>].
Respond accurately and concisely in Vietnamese.

Formatting rules:
- For multi-step or procedural answers, use numbered lists or bullets.
- For short factual answers, keep it to 1-3 sentences.
- Do not use markdown formatting symbols such as **, ##, or `.
"""

USER_PROMPT = """CONTEXT:
{context}

QUESTION: {question}

Answer:"""


@dataclass
class Answer:
    question: str
    answer: str
    is_fallback: bool
    confidence: float
    chunks: list[RetrievedChunk]
    latency_s: float
    sources: list[str] = field(default_factory=list)
    search_query: str = ""  # The actual retrieval query after any rewrite step.


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a compact context block for the LLM prompt."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        loc = c.source + (f" p.{c.page}" if c.page else "")
        blocks.append(f"[{i}] ({loc})\n{c.text}")
    return "\n\n".join(blocks)


def _unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Deduplicate citations while preserving the first-seen source order."""
    seen: list[str] = []
    for c in chunks:
        s = c.source + (f" p.{c.page}" if c.page else "")
        if s not in seen:
            seen.append(s)
    return seen


_INSTRUCTION_RE = re.compile(
    r"\b(hãy|giải thích|diễn giải|tóm tắt|liệt kê|trình bày|cho tôi biết|mô tả|"
    r"so sánh|khái quát|tổng hợp|summar|explain|describe|list all|overview)\b",
    re.IGNORECASE,
)


def _needs_rewrite(question: str) -> bool:
    """Rewrite only when the input is likely to hurt retrieval quality.

    Clean English questions usually work better without rewriting because rewrite
    steps can distort the original semantic target and reduce groundedness.
    """
    return not question.isascii() or bool(_INSTRUCTION_RE.search(question))


def _rewrite_query(question: str) -> str:
    """Normalize a question into a short retrieval query.

    If the rewrite step fails, we fall back to the original question rather than
    blocking the request entirely.
    """
    try:
        out = get_llm().invoke(REWRITE_PROMPT.format(q=question)).content
        out = (out if isinstance(out, str) else str(out)).strip()
        return out or question
    except Exception:  # noqa: BLE001 - failed rewrite should never block the retrieval path
        return question


_SUMMARY_RE = re.compile(
    r"(tóm tắt|tóm lược|summar|mục lục|khái quát|tổng quan|tổng quát|overview|"
    r"nội dung chính|ý chính|liệt kê (toàn bộ|tất cả|hết)|toàn bộ (tài liệu|nội dung)|"
    r"cả (tài liệu|file|quyển|cuốn))",
    re.IGNORECASE,
)


def _summary_intent(question: str) -> bool:
    """Detect requests that need an entire-document summary rather than top-k retrieval."""
    return bool(_SUMMARY_RE.search(question))


def answer_question(
    question: str,
    k: int | None = None,
    source: str | None = None,
    force_summary: bool = False,
) -> Answer:
    """Entry point for the full RAG flow: normalize input, route the query, and generate a grounded answer.

    The function is intentionally the single orchestration point so the CLI, API, and
    evaluation pipeline all share the same fallback and grounding behavior.
    """
    t0 = time.perf_counter()
    question = unicodedata.normalize("NFC", question)

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
                    answer="There are no documents in the system yet. Please upload a PDF first.",
                    is_fallback=True,
                    confidence=0.0,
                    chunks=[],
                    latency_s=time.perf_counter() - t0,
                )
            else:
                return Answer(
                    question=question,
                    answer=(
                        "Multiple documents are available — please choose a source in the scope dropdown and ask again."
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
    q = " ".join(sys.argv[1:]) or input("Question: ")
    a = answer_question(q)
    print("\n" + a.answer + "\n" + "-" * 40)
    if a.search_query and a.search_query.lower() != q.lower():
        print(f"search query: {a.search_query}")
    print(
        f"confidence={a.confidence:.2f}  fallback={a.is_fallback}  "
        f"latency={a.latency_s:.2f}s"
    )
    if a.sources:
        print("Sources:", "; ".join(a.sources))
