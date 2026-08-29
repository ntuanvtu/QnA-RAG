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
