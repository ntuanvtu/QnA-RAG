"""FastAPI app that serves the chat UI and JSON endpoints for the RAG workflow.

Run with:
    uvicorn app.api:app --reload
Then open http://127.0.0.1:8000 in the browser.
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
    """Serve the static chat UI from the frontend folder."""
    return FileResponse(settings.static_dir / "index.html")


@app.get("/api/info")
def info() -> JSONResponse:
    """Return the active LLM model name for the frontend."""
    return JSONResponse({"model": settings.llm_model})


@app.get("/api/files")
def files() -> JSONResponse:
    """Return all ingested documents with chunk counts for the scope dropdown."""
    return JSONResponse(list_sources())


@app.post("/api/chat")
async def chat(
    question: str = Form(""),
    source: str = Form(""),
    summarize: str = Form(""),
    file: UploadFile | None = File(None),
) -> JSONResponse:
    """Handle one chat turn and auto-ingest an uploaded PDF when needed.

    This keeps the upload flow simple: the user can submit a file and a question in
    the same request, and the answer logic sees the refreshed corpus immediately.
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
                f"Ingested {n} chunks from {dest.name}."
                if n
                else f"{dest.name} is already present in the system."
            )

        if not question.strip():
            if ingested:
                return JSONResponse({"answer": None, "ingested": ingested})
            return JSONResponse(
                {"error": "Please enter a question or upload a document."}, status_code=400
            )

        ans = answer_question(
            question, source=source or None, force_summary=summarize == "1"
        )
    except Exception as e:  # noqa: BLE001 - a single bad request should not crash the endpoint
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
    """Clear the entire in-memory vector collection and start from an empty corpus."""
    get_vectorstore().reset_collection()
    return JSONResponse({"ok": True})
