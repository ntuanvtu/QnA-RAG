"""PDF ingestion pipeline: extract text, split it into chunks, embed it, and store it in Chroma.

Run directly:
    python -m app.ingest data/example.pdf
    python -m app.ingest
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
    """Prefer structural boundaries so chunks do not break mid-sentence or mid-section."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )


def _load_pages(pdf_path: Path) -> list[Document]:
    """Extract one document per page while skipping blank pages.

    We rely on pypdfium2 because it handles technical layouts and multi-column PDF
    text much more reliably than text-only extraction. Raster-image PDFs still
    fall outside the project scope unless OCR is added later.
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
                pages.append(Document(page_content=text, metadata={"page": idx}))
    finally:
        pdf.close()
    return pages


def _merge_sparse_pages(pages: list[Document]) -> list[Document]:
    """Merge consecutive pages for slide-like PDFs so chunks remain informative.

    Sparse decks often produce tiny, noisy chunks when each page is processed alone.
    The block page number keeps the first page of the merged run, which is a trade-off
    that preserves retrieval quality while keeping citations close to the relevant cluster.
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
    """Add a single PDF to the vector store and return the number of chunks inserted.

    This avoids duplicate indexing when the same file is re-uploaded and lets the UI
    refresh a file by name even if content changed, without leaving stale chunks behind.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    content_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    vs = get_vectorstore()
    if vs.get(where={"content_hash": content_hash}, limit=1)["ids"]:
        return 0

    pages = _load_pages(pdf_path)
    if not pages:
        raise ValueError(
            f"No extractable text was found in {pdf_path.name}. "
            "Image-only PDFs require OCR and are outside the current scope."
        )

    median_chars = statistics.median(len(p.page_content) for p in pages)
    if median_chars < settings.sparse_median_threshold:
        pages = _merge_sparse_pages(pages)

    chunks = _splitter().split_documents(pages)
    for c in chunks:
        c.metadata["source"] = pdf_path.name
        c.metadata["page"] = int(c.metadata.get("page", 0)) + 1
        c.metadata["content_hash"] = content_hash

    if vs.get(where={"source": pdf_path.name}, limit=1)["ids"]:
        vs.delete(where={"source": pdf_path.name})

    vs.add_documents(chunks)
    return len(chunks)


def ingest_dir(data_dir: str | Path | None = None) -> dict[str, int]:
    """Ingest every PDF in a directory and return a file-to-chunk count map."""
    data_dir = Path(data_dir or settings.data_dir)
    return {pdf.name: ingest_pdf(pdf) for pdf in sorted(data_dir.glob("*.pdf"))}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = ingest_pdf(sys.argv[1])
        print(f"Inserted {n} chunks from {sys.argv[1]}")
    else:
        results = ingest_dir()
        if not results:
            print(f"No PDF files were found in {settings.data_dir}")
        for name, n in results.items():
            print(f"{name}: {n} chunks")
