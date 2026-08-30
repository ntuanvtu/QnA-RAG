"""Ingest PDF: đọc -> chia chunk -> embed -> lưu vào Chroma.

Chạy trực tiếp:
    python -m app.ingest data/tai-lieu.pdf     # nạp 1 file
    python -m app.ingest                        # nạp mọi PDF trong data/
"""
from __future__ import annotations

import hashlib
import statistics
import sys
from pathlib import Path

import pypdfium2 as pdfium
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    """Đọc PDF -> mỗi trang là 1 Document. Bỏ qua trang không trích được chữ.

    Dùng pypdfium2 (PDFium — engine PDF của Chromium): trích text ổn định hơn
    với layout nhiều cột / slide. Chữ nằm trong ảnh raster (sơ đồ, ảnh chụp) vẫn
    không đọc được — cần OCR, ngoài phạm vi project (xem README).
    """
    pages: list[Document] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for idx in range(len(pdf)):
            page = pdf[idx]
            textpage = page.get_textpage()
            text = (textpage.get_text_range() or "").replace("\r\n", "\n").strip()
            textpage.close()
            page.close()
            if text:
                # page đánh số từ 0; +1 khi hiển thị cho người đọc (làm ở dưới)
                pages.append(Document(page_content=text, metadata={"page": idx}))
    finally:
        pdf.close()
    return pages


def _merge_sparse_pages(pages: list[Document]) -> list[Document]:
    """Gộp các trang liền nhau tới khi khối đủ 'dày' (>= sparse_block_chars).

    Dùng cho PDF trình chiếu: mỗi slide chỉ vài dòng rời rạc; để nguyên từng trang
    thì chunk quá ngắn, embedding nhiễu, retrieve trượt. `page` của khối lấy theo
    trang đầu tiên trong khối (trích dẫn sẽ trỏ tới slide đầu của cụm liên quan).
    """
    blocks: list[Document] = []
    buf: list[str] = []
    start_page: int | None = None
    for d in pages:
        if start_page is None:
            start_page = d.metadata["page"]
        buf.append(d.page_content)
        if sum(len(t) for t in buf) >= settings.sparse_block_chars:
            blocks.append(
                Document(page_content="\n\n".join(buf), metadata={"page": start_page})
            )
            buf, start_page = [], None
    if buf:
        blocks.append(
            Document(page_content="\n\n".join(buf), metadata={"page": start_page})
        )
    return blocks


def ingest_pdf(pdf_path: str | Path) -> int:
    """Nạp 1 file PDF vào vector store. Trả về số chunk đã thêm (0 = file y hệt đã có sẵn).

    Chống chunk trùng khi UI tự động ingest mỗi lần submit:
      - Nội dung file (SHA-256) đã có trong store  -> bỏ qua, không embed lại.
      - Cùng tên file nhưng nội dung khác (bản mới) -> xoá chunk cũ rồi nạp lại.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không thấy file: {pdf_path}")

    content_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    vs = get_vectorstore()
    if vs.get(where={"content_hash": content_hash}, limit=1)["ids"]:
        return 0

    pages = _load_pages(pdf_path)
    if not pages:
        raise ValueError(
            f"Không trích được chữ từ {pdf_path.name} "
            "(PDF scan ảnh cần OCR — ngoài phạm vi project này)."
        )

    # Tài liệu trình chiếu (text thưa) -> gộp slide liền nhau trước khi chunk.
    median_chars = statistics.median(len(p.page_content) for p in pages)
    if median_chars < settings.sparse_median_threshold:
        pages = _merge_sparse_pages(pages)

    chunks = _splitter().split_documents(pages)
    for c in chunks:
        c.metadata["source"] = pdf_path.name
        c.metadata["page"] = int(c.metadata.get("page", 0)) + 1
        c.metadata["content_hash"] = content_hash

    # Dọn bản cũ cùng tên (nếu có) trước khi nạp bản mới.
    if vs.get(where={"source": pdf_path.name}, limit=1)["ids"]:
        vs.delete(where={"source": pdf_path.name})

    vs.add_documents(chunks)
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
