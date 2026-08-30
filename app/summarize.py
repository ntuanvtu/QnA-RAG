"""Whole-document summary mode for one file using a map-reduce pattern.

Unlike standard retrieval, this path reads every stored chunk for a source file,
then summarizes each batch and merges them into a final answer.
"""
from __future__ import annotations

import time

from app.llm import get_llm
from app.retriever import RetrievedChunk
from app.vectorstore import get_vectorstore
from config import settings

_MAP_PROMPT = """Summarize the excerpts below from document "{source}" in order.
Keep the key ideas, technical terms, and important numbers. Write in Vietnamese, keep it concise, and avoid markdown.
{focus}
EXCERPTS:
{text}

Summary:"""

_REDUCE_PROMPT = """You are given several partial summaries of document "{source}" in reading order.
Combine them into one coherent summary with no repetition. Write in Vietnamese and avoid markdown; if the content is procedural or classificatory, use numbered lists with nested subpoints.
{focus}{note}
PARTIAL SUMMARIES:
{parts}

Combined summary:"""


def _chunks_of_source(source: str) -> list[tuple[int, str]]:
    """Return every chunk belonging to a source, sorted by page number."""
    got = get_vectorstore().get(where={"source": source}, include=["metadatas", "documents"])
    pairs = [
        (int(m.get("page", 0) or 0), doc)
        for m, doc in zip(got["metadatas"], got["documents"])
    ]
    return sorted(pairs, key=lambda p: p[0])


def _batches(pairs: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Group nearby chunks into batches sized roughly to summary_batch_chars."""
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
    """Keep the ordering while sampling evenly across a long list."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _focus_line(question: str | None) -> str:
    """Add a short focus hint to keep the summary aligned with the user's request."""
    q = (question or "").strip()
    return f'User asked specifically: "{q}" — prioritize that emphasis.\n' if q else ""


def summarize_document(source: str, question: str | None = None):
    """Create a full-document summary for a given source and return the standard Answer object."""
    from app.rag import Answer  # Avoid a circular import at module import time.

    t0 = time.perf_counter()
    pairs = _chunks_of_source(source)
    if not pairs:
        return Answer(
            question=question or f"Summary of {source}",
            answer=f'No content for "{source}" is available in the system yet.',
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
            f"\nNOTE: This document is long, so the summary is based on {len(batches)} evenly sampled sections "
            "instead of the entire text."
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
        question=question or f"Summary of {source}",
        answer=final,
        is_fallback=False,
        confidence=1.0,
        chunks=[
            RetrievedChunk(text=t, source=source, page=p, score=1.0)
            for p, t in pairs[:1]
        ],
        latency_s=time.perf_counter() - t0,
        sources=[f"{source} (full document, p.{pages[0]}–p.{pages[-1]})"] if pages else [source],
    )
