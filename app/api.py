"""FastAPI: giao diện chatbot + API JSON cho hỏi đáp RAG.

Chạy:
    uvicorn app.api:app --reload
rồi mở http://127.0.0.1:8000

Frontend là một trang tĩnh (app/static/index.html) gọi các endpoint JSON dưới đây
bằng fetch — không dùng template engine, không có state ngoài vector store trên đĩa.
"""
from __future__ import annotations

import shutil

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.ingest import ingest_pdf
from app.rag import answer_question
from app.vectorstore import get_vectorstore, list_sources
from config import settings

app = FastAPI(title="Technical Document Q&A (RAG)")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.get("/api/info")
def info() -> JSONResponse:
    return JSONResponse({"model": settings.llm_model})


@app.get("/api/files")
def files() -> JSONResponse:
    """Danh sách tài liệu đã nạp (cho dropdown chọn phạm vi)."""
    return JSONResponse(list_sources())


@app.post("/api/chat")
async def chat(
    question: str = Form(""),
    source: str = Form(""),
    summarize: str = Form(""),
    file: UploadFile | None = File(None),
) -> JSONResponse:
    """Một lượt chat: nếu có file đính kèm thì tự nạp trước, rồi trả lời câu hỏi.

    `source` != "" -> chỉ tìm câu trả lời trong file đó.
    `summarize` = "1" -> ép chế độ tóm tắt toàn bộ tài liệu.
    """
    ingested: str | None = None
    try:
        if file is not None and file.filename:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            dest = settings.data_dir / file.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            n = ingest_pdf(dest)
            ingested = (
                f"Đã nạp {n} đoạn từ {dest.name}."
                if n
                else f"{dest.name} đã có sẵn trong hệ thống."
            )

        if not question.strip():
            if ingested:
                return JSONResponse({"answer": None, "ingested": ingested})
            return JSONResponse(
                {"error": "Nhập câu hỏi hoặc đính kèm tài liệu."}, status_code=400
            )

        ans = answer_question(
            question, source=source or None, force_summary=summarize == "1"
        )
    except Exception as e:  # noqa: BLE001 - endpoint không được sập vì 1 request lỗi
        return JSONResponse({"error": str(e)}, status_code=400)

    return JSONResponse(
        {
            "answer": ans.answer,
            "is_fallback": ans.is_fallback,
            "confidence": round(ans.confidence, 3),
            "latency_s": round(ans.latency_s, 2),
            "sources": ans.sources,
            "search_query": ans.search_query,
            "ingested": ingested,
        }
    )


@app.post("/api/reset")
def reset() -> JSONResponse:
    """Xoá toàn bộ tài liệu đã nạp khỏi vector store."""
    get_vectorstore().reset_collection()
    return JSONResponse({"ok": True})
