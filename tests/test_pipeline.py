"""Smoke test - chạy được KHÔNG cần API key.

Chỉ kiểm tra phần ingest + retrieve (embedding local). Bỏ qua nếu chưa có PDF.
Chạy:  pytest -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.ingest import ingest_pdf
from app.retriever import retrieve

_PDFS = sorted(Path("data").glob("*.pdf"))


@pytest.mark.skipif(not _PDFS, reason="Chưa có PDF nào trong data/")
def test_ingest_and_retrieve() -> None:
    pdf_names = {p.name for p in _PDFS}

    n_chunks = ingest_pdf(_PDFS[0])
    assert n_chunks > 0

    result = retrieve("nội dung chính của tài liệu là gì", k=3)
    assert len(result.chunks) <= 3
    assert 0.0 <= result.confidence <= 1.0
    for c in result.chunks:
        assert c.text
        assert c.source in pdf_names
        assert c.page is None or c.page >= 1


@pytest.mark.skipif(len(_PDFS) < 2, reason="Cần ít nhất 2 PDF trong data/")
def test_retrieve_scoped_to_one_source() -> None:
    ingest_pdf(_PDFS[0])
    ingest_pdf(_PDFS[1])

    target = _PDFS[0].name
    result = retrieve("nội dung chính của tài liệu là gì", k=5, source=target)
    assert result.chunks  # vẫn tìm được trong file được chỉ định
    assert all(c.source == target for c in result.chunks)


def test_summary_intent() -> None:
    from app.rag import _summary_intent

    assert _summary_intent("Tóm tắt tài liệu này")
    assert _summary_intent("cho tôi cái nhìn tổng quan về cả tài liệu")
    assert _summary_intent("liệt kê toàn bộ các quy tắc")
    assert not _summary_intent("What is ASIC?")
    assert not _summary_intent("Liệt kê các loại sản phẩm bán dẫn")


def test_summary_batching() -> None:
    from app.summarize import _batches, _evenly_sample

    pairs = [(i, "x" * 2500) for i in range(1, 11)]  # 10 chunk, mỗi cái 2500 ký tự
    batches = _batches(pairs)  # summary_batch_chars = 6000 -> ~3 chunk / lô
    assert sum(len(b) for b in batches) == 10
    assert all(len(b) >= 1 for b in batches)

    assert len(_evenly_sample(list(range(100)), 10)) == 10
    assert _evenly_sample([1, 2], 10) == [1, 2]


def test_fallback_threshold_logic() -> None:
    """RetrievalResult.is_confident phản ánh đúng ngưỡng."""
    from app.retriever import RetrievalResult, RetrievedChunk

    empty = RetrievalResult(chunks=[])
    assert empty.confidence == 0.0
    assert empty.is_confident is False

    high = RetrievalResult(
        chunks=[RetrievedChunk(text="x", source="s", page=1, score=0.9)]
    )
    assert high.is_confident is True
