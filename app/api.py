"""FastAPI: giao diện web tối giản (form HTML) để upload PDF và hỏi đáp.

Chạy:
    uvicorn app.api:app --reload
rồi mở http://127.0.0.1:8000
"""
from __future__ import annotations

import html
import shutil

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

from app.ingest import ingest_pdf
from app.rag import answer_question
from config import settings

app = FastAPI(title="Technical Document Q&A (RAG)")

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Document Q&A</title>
<style>
 body{{font-family:system-ui,-apple-system,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#1a1a1a}}
 textarea,input[type=text]{{width:100%;padding:.55rem;font:inherit;box-sizing:border-box}}
 button{{padding:.55rem 1.1rem;font:inherit;margin-top:.6rem;cursor:pointer;border-radius:6px;border:1px solid #888}}
 .card{{border:1px solid #ddd;border-radius:10px;padding:1rem 1.2rem;margin:1.1rem 0}}
 .muted{{color:#666;font-size:.9rem}}
 .fallback{{color:#b00020}}
 pre{{white-space:pre-wrap;background:#f5f5f5;padding:.75rem;border-radius:6px;font-size:.85rem}}
 summary{{cursor:pointer}}
</style></head><body>
<h1>Technical Document Q&A</h1>
<p class="muted">RAG + fallback chống hallucination. Model: {model}</p>

<div class="card">
 <h3>1. Nạp tài liệu (PDF)</h3>
 <form action="/ingest" method="post" enctype="multipart/form-data">
  <input type="file" name="file" accept="application/pdf" required>
  <button type="submit">Nạp vào hệ thống</button>
 </form>
 {ingest_msg}
</div>

<div class="card">
 <h3>2. Đặt câu hỏi</h3>
 <form action="/ask" method="post">
  <textarea name="question" rows="3" placeholder="Nhập câu hỏi về nội dung tài liệu..." required>{question}</textarea>
  <button type="submit">Hỏi</button>
 </form>
</div>

{answer_block}
</body></html>"""


def _render(ingest_msg: str = "", question: str = "", answer_block: str = "") -> str:
    return PAGE.format(
        model=html.escape(settings.llm_model),
        ingest_msg=ingest_msg,
        question=html.escape(question),
        answer_block=answer_block,
    )


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _render()


@app.post("/ingest", response_class=HTMLResponse)
async def ingest(file: UploadFile = File(...)) -> str:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.data_dir / (file.filename or "upload.pdf")
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        n = ingest_pdf(dest)
        msg = f'<p class="muted">✅ Đã nạp <b>{n}</b> chunk từ {html.escape(dest.name)}.</p>'
    except Exception as e:  # noqa: BLE001 - hiển thị lỗi cho người dùng
        msg = f'<p class="fallback">Lỗi khi nạp: {html.escape(str(e))}</p>'
    return _render(ingest_msg=msg)


@app.post("/ask", response_class=HTMLResponse)
async def ask(question: str = Form(...)) -> str:
    try:
        ans = answer_question(question)
    except Exception as e:  # noqa: BLE001
        block = f'<div class="card fallback">Lỗi: {html.escape(str(e))}</div>'
        return _render(question=question, answer_block=block)

    cls = "fallback" if ans.is_fallback else ""
    sources = (
        f'<p class="muted">Nguồn: {html.escape("; ".join(ans.sources))}</p>'
        if ans.sources
        else ""
    )
    ctx = "\n\n".join(
        f"[{i}] ({c.source} tr.{c.page}) score={c.score:.2f}\n{c.text[:400]}..."
        for i, c in enumerate(ans.chunks, 1)
    )
    block = f"""<div class="card">
 <h3 class="{cls}">Trả lời</h3>
 <p>{html.escape(ans.answer)}</p>
 {sources}
 <p class="muted">Confidence: {ans.confidence:.2f} &middot; Thời gian: {ans.latency_s:.2f}s
   &middot; Fallback: {"có" if ans.is_fallback else "không"}</p>
 <details><summary class="muted">Xem {len(ans.chunks)} chunk đã truy hồi</summary>
   <pre>{html.escape(ctx)}</pre></details>
</div>"""
    return _render(question=question, answer_block=block)
