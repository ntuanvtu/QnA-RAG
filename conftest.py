"""Common pytest configuration for isolated test runs.

This file sits at the repository root so pytest resolves the project modules
without requiring an editable install step.
"""
from __future__ import annotations

import pytest

from config import settings


@pytest.fixture(autouse=True)
def _isolate_vector_store(tmp_path, monkeypatch):
    """Use a temporary Chroma directory so tests cannot pollute real data."""
    monkeypatch.setattr(settings, "vector_dir", tmp_path / "chroma")
