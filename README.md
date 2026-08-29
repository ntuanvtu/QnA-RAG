# Technical Document Q&A Assistant (RAG + Fallback + Eval)

Hệ thống hỏi–đáp trên tài liệu kỹ thuật (PDF) dùng kiến trúc **RAG**, có cơ chế
**fallback chống hallucination** (biết khi nào không đủ thông tin để trả lời) và
một **bộ đánh giá định lượng** để đo hệ thống tốt tới đâu — không chỉ "chạy được"
mà còn "đo được".

## Kiến trúc

```
        PDF ──► ingest.py ──► chunk (RecursiveCharacterTextSplitter)
                                  │
                                  ▼
                      embeddings.py (sentence-transformers, local)
                                  │
                                  ▼
                          Chroma vector DB  (storage/chroma)
                                  │
   câu hỏi ──► retriever.py ──► top-k chunk + similarity score
                                  │
                     ┌────────────┴─────────────┐
             score < ngưỡng?                score ≥ ngưỡng
                     │                            │
                     ▼                            ▼
           FALLBACK ("không tìm thấy")   rag.py ──► Claude ──► câu trả lời + [nguồn: file tr.X]
                                                        │
                                              LLM tự nhận thiếu thông tin? ──► FALLBACK (tầng 2)
```

| Thành phần | Lựa chọn | File |
|---|---|---|
| API / UI | FastAPI + form HTML | [app/api.py](app/api.py) |
| Đọc & chunk PDF | `pypdf` + LangChain `RecursiveCharacterTextSplitter` | [app/ingest.py](app/ingest.py) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (chạy local, miễn phí) | [app/embeddings.py](app/embeddings.py) |
| Vector DB | Chroma (persist ra đĩa) | [app/vectorstore.py](app/vectorstore.py) |
| Retrieval + confidence | cosine similarity, ngưỡng cấu hình được | [app/retriever.py](app/retriever.py) |
| LLM | Claude (`claude-haiku-4-5`) qua `langchain-anthropic` | [app/llm.py](app/llm.py) |
| Fallback | 2 tầng: (1) ngưỡng similarity, (2) LLM tự nhận thiếu thông tin | [app/rag.py](app/rag.py) |
| Eval | bộ test tự viết + Claude làm giám khảo groundedness | [eval/run_eval.py](eval/run_eval.py) |

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env            # rồi mở .env, điền ANTHROPIC_API_KEY
```

> Lấy API key tại <https://console.anthropic.com>. Phần embedding chạy local nên
> **ingest + retrieve không cần key**; chỉ bước sinh câu trả lời mới cần.

## Cách dùng

### 1. Nạp tài liệu

```bash
# đặt file PDF vào thư mục data/ rồi:
python -m app.ingest data/ten-tai-lieu.pdf     # 1 file
python -m app.ingest                           # tất cả PDF trong data/
```

### 2. Hỏi nhanh trên terminal

```bash
python -m app.rag "Thuật toán X gồm mấy bước?"
```

### 3. Chạy giao diện web

```bash
uvicorn app.api:app --reload
# mở http://127.0.0.1:8000
```

### 4. Chạy đánh giá

```bash
# sửa eval/testset.json: thay câu hỏi VÍ DỤ bằng câu hỏi thật về PDF của bạn
python -m eval.run_eval
```

Kết quả in ra: `Retrieval accuracy@k`, `Groundedness`, `Fallback đúng`, `Latency (TB / p95)`.

### Test khói (không cần key)

```bash
pytest -q
```

## Tinh chỉnh

Mọi tham số nằm trong [config.py](config.py) (ghi đè được qua `.env`):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.35` | similarity tối thiểu để KHÔNG fallback |
| `TOP_K` | `4` | số chunk truy hồi |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | kích thước chunk |
| `LLM_MODEL` | `claude-haiku-4-5` | đổi sang `claude-sonnet-5` khi chạy eval cuối để tăng chất lượng |

Cách tune ngưỡng: chạy `run_eval` vài lần với các giá trị `CONFIDENCE_THRESHOLD`
khác nhau, chọn giá trị mà câu out-of-scope bị fallback hết **mà** câu in-scope
vẫn được trả lời.

## Kết quả eval

Bộ test tự xây: **35 câu** (25 in-scope trên 2 tài liệu + 10 out-of-scope).
Model `claude-haiku-4-5`, embedding `all-MiniLM-L6-v2`, `top_k=4`, ngưỡng `0.35`.
Chi tiết từng câu: [eval/results.json](eval/results.json).

| Chỉ số | Kết quả |
|---|---|
| Retrieval accuracy@4 | **96%** (24/25) |
| Groundedness (faithfulness, Claude judge) | **100%** (23/23 câu được trả lời) |
| Chống hallucination — câu ngoài phạm vi bị từ chối | **10/10** |
| In-scope được trả lời (không fallback nhầm) | **92%** (23/25) |
| Latency | TB **3.1s** · p95 **6.1s** |

> 2/25 câu in-scope bị fallback do retriever không xếp đúng chunk chứa đáp án lên
> top-4 (MiniLM + chunk cố định). Hệ thống từ chối thay vì đoán bừa — đúng thiết
> kế. Cải thiện được bằng hybrid search / re-ranker (ngoài scope).

## Ngoài scope (có chủ đích)

Agent / tool-calling, OCR, hybrid search, re-ranker, deploy quy mô lớn, load testing.
