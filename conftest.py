# pytest tự thêm thư mục chứa file này vào sys.path, nhờ đó `import app` /
# `import config` chạy được khi gõ `pytest` từ gốc repo.
"""Fixture chung cho test."""
from __future__ import annotations

import pytest

from config import settings


@pytest.fixture(autouse=True)
def _isolate_vector_store(tmp_path, monkeypatch):
    # Trỏ vector store sang thư mục tạm: test không đụng storage/chroma/ thật
    # (tránh nhân đôi chunk và phụ thuộc vào dữ liệu đã ingest sẵn).
    monkeypatch.setattr(settings, "vector_dir", tmp_path / "chroma")
