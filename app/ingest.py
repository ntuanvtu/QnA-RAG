"""Ingest PDF: đọc -> chia chunk -> embed -> lưu vào Chroma.

Chạy trực tiếp:
    python -m app.ingest data/tai-lieu.pdf     # nạp 1 file
    python -m app.ingest                        # nạp mọi PDF trong data/
"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.vectorstore import get_vectorstore
from config import settings


def _splitter() -> RecursiveCharacterTextSplitter:
    # Ưu tiên cắt ở ranh giới đoạn/câu để không cắt giữa câu khi tránh được.
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )


def _load_pages(pdf_path: Path) -> list[Document]:
    """Đọc PDF -> mỗi trang là 1 Document. Bỏ qua trang không trích được chữ."""
    reader = PdfReader(str(pdf_path))
    pages: list[Document] = []
    for idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            # page đánh số từ 0; +1 khi hiển thị cho người đọc (làm ở dưới)
            pages.append(Document(page_content=text, metadata={"page": idx}))
    return pages


def ingest_pdf(pdf_path: str | Path) -> int:
    """Nạp 1 file PDF vào vector store. Trả về số chunk đã thêm."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không thấy file: {pdf_path}")

    pages = _load_pages(pdf_path)
    if not pages:
        raise ValueError(
            f"Không trích được chữ từ {pdf_path.name} "
            "(PDF scan ảnh cần OCR — ngoài phạm vi project này)."
        )

    chunks = _splitter().split_documents(pages)
    for c in chunks:
        c.metadata["source"] = pdf_path.name
        c.metadata["page"] = int(c.metadata.get("page", 0)) + 1

    get_vectorstore().add_documents(chunks)
    return len(chunks)


def ingest_dir(data_dir: str | Path | None = None) -> dict[str, int]:
    """Nạp mọi file *.pdf trong thư mục. Trả về {tên_file: số_chunk}."""
    data_dir = Path(data_dir or settings.data_dir)
    return {pdf.name: ingest_pdf(pdf) for pdf in sorted(data_dir.glob("*.pdf"))}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = ingest_pdf(sys.argv[1])
        print(f"Đã nạp {n} chunk từ {sys.argv[1]}")
    else:
        results = ingest_dir()
        if not results:
            print(f"Không tìm thấy PDF nào trong {settings.data_dir}")
        for name, n in results.items():
            print(f"{name}: {n} chunk")
