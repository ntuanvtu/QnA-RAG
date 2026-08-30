# Technical Document Q&A Assistant

**Hỏi–đáp bằng tiếng Việt trên tài liệu kỹ thuật PDF của bạn, có trích dẫn nguồn và biết từ chối khi không đủ thông tin.**

Đây là một chatbot kiểu RAG (*Retrieval-Augmented Generation*): thay vì để mô hình ngôn ngữ
trả lời "từ trí nhớ" (dễ bịa), hệ thống **tìm các đoạn văn liên quan trong tài liệu trước**,
rồi mới yêu cầu Claude trả lời **chỉ dựa trên các đoạn đó** và ghi rõ trang nguồn.

Điểm nhấn của project không chỉ là "chạy được" mà là **"đo được"**: có sẵn một bộ đánh giá
định lượng tự xây (35 câu hỏi) đo độ chính xác truy hồi, độ bám sát nguồn (groundedness),
khả năng từ chối câu hỏi ngoài phạm vi, và độ trễ.

---

## Tính năng chính

- **Giao diện chatbot** – đính kèm PDF và đặt câu hỏi trong cùng một khung chat.
- **Tự động nạp tài liệu** – tài liệu được xử lý ngay khi bạn gửi câu hỏi, không có bước "nạp" riêng.
  Gửi lại đúng file cũ sẽ được bỏ qua (chống trùng theo mã băm nội dung).
- **Trích dẫn nguồn** – mỗi ý trong câu trả lời kèm `[nguồn: tên_file tr.X]`.
- **Chống bịa đặt (2 tầng)**:
  1. Nếu không tìm được đoạn nào đủ liên quan → trả lời "không tìm thấy thông tin", **không gọi LLM** (tiết kiệm chi phí).
  2. Nếu LLM tự thấy ngữ cảnh không đủ → cũng trả về câu từ chối thay vì đoán bừa.
- **Hiểu câu hỏi tiếng Việt trên tài liệu tiếng Anh** – trước khi tìm, Claude viết lại câu hỏi
  thành truy vấn tìm kiếm (bỏ chữ chỉ thị kiểu "hãy giải thích dễ hiểu…", dịch sang tiếng Anh).
  Câu hỏi tiếng Anh gọn thì bỏ qua bước này.
- **Tóm tắt / thao tác toàn tài liệu** – hỏi "tóm tắt tài liệu", "liệt kê toàn bộ…" (hoặc tick ô
  *Tóm tắt cả tài liệu*) → hệ thống đọc **hết** các đoạn của một file và tóm tắt kiểu map-reduce,
  thay vì chỉ nhìn top-4 đoạn.
- **Chọn phạm vi** – dropdown giới hạn câu hỏi trong một tài liệu cụ thể.
- **Bộ đánh giá định lượng** – `python -m eval.run_eval` cho ra bảng số liệu, dùng chính Claude làm giám khảo mức độ bám sát nguồn.
- **Embedding chạy local, miễn phí** – bước nạp và truy hồi tài liệu không cần API key; chỉ bước sinh câu trả lời mới cần.

---

## Giao diện

Chạy `uvicorn app.api:app --reload` rồi mở <http://127.0.0.1:8000>:

- Nút **📎** để đính kèm một file PDF.
- **Phạm vi** – chọn hỏi trong tất cả tài liệu hay một tài liệu cụ thể.
- Ô **Tóm tắt cả tài liệu** – ép chế độ tóm tắt (nếu câu hỏi không rõ ý định).
- Ô nhập câu hỏi, nhấn **Gửi** (hoặc phím Enter; Shift+Enter để xuống dòng).
- Câu trả lời hiện dưới dạng bong bóng chat, kèm nguồn trích dẫn, độ tự tin (confidence) và thời gian xử lý.
  Nếu câu hỏi được viết lại, dòng meta hiện "tìm với: …".
- Nút **Xóa tài liệu** để dọn sạch toàn bộ tài liệu đã nạp.

> *(Gợi ý: chụp màn hình phần hỏi–đáp và một câu bị từ chối, lưu vào `docs/` rồi chèn ảnh vào đây.)*

---

## Kiến trúc

```
                        ┌──────────────────────────────────────────────┐
   PDF  ──────────────► │  app/ingest.py                                │
                        │   • pypdfium2 đọc text từng trang             │
                        │   • slide thưa chữ → gộp các trang liền nhau  │
                        │   • RecursiveCharacterTextSplitter → chunk    │
                        └───────────────────────┬──────────────────────┘
                                                ▼
                        app/embeddings.py  (sentence-transformers, chạy local)
                                                ▼
                        app/vectorstore.py (Chroma, lưu ra đĩa storage/chroma/, không gian cosine)

  câu hỏi ────────────► app/rag.py  answer_question()
                          │
              ┌───────────┴────────────┐
     "tóm tắt / toàn bộ…"        câu hỏi thường
              │                         │
              ▼                         ▼
   app/summarize.py           (nếu cần) viết lại câu hỏi bằng Claude
   map-reduce trên TẤT CẢ            → app/retriever.py
   đoạn của 1 file                     • lấy top-k đoạn + điểm similarity (0..1)
                                       • confidence = điểm cao nhất trong top-k
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
              confidence < ngưỡng (0.35)                    confidence ≥ ngưỡng
                     │                                              │
                     ▼                                              ▼
        FALLBACK tầng 1                              app/llm.py (Claude)
        "Không tìm thấy thông tin…"                    • prompt: chỉ trả lời dựa trên ngữ cảnh
        (không gọi LLM)                                • LLM tự nhận thiếu thông tin? → FALLBACK tầng 2
                                                       • ngược lại → câu trả lời + [nguồn: file tr.X]
```

| Thành phần | Lựa chọn | File |
|---|---|---|
| API + giao diện | FastAPI + trang chat tĩnh (vanilla JS, không framework) | [app/api.py](app/api.py), [app/static/index.html](app/static/index.html) |
| Đọc & chunk PDF | `pypdfium2` (PDFium – engine PDF của Chromium) + LangChain `RecursiveCharacterTextSplitter` | [app/ingest.py](app/ingest.py) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` – chạy local, miễn phí | [app/embeddings.py](app/embeddings.py) |
| Vector DB | Chroma, persist ra đĩa, không gian cosine | [app/vectorstore.py](app/vectorstore.py) |
| Truy hồi + confidence | cosine similarity, ngưỡng cấu hình được, lọc theo tài liệu | [app/retriever.py](app/retriever.py) |
| Viết lại câu hỏi + fallback | 2 tầng fallback; viết lại truy vấn cho câu tiếng Việt / có chỉ thị | [app/rag.py](app/rag.py) |
| Tóm tắt toàn tài liệu | map-reduce trên mọi đoạn của một file | [app/summarize.py](app/summarize.py) |
| LLM | Claude `claude-haiku-4-5` qua `langchain-anthropic` | [app/llm.py](app/llm.py) |
| Đánh giá | bộ test tự viết + Claude làm giám khảo groundedness | [eval/run_eval.py](eval/run_eval.py) |

**Vì sao chọn như vậy:** embedding local để nạp/tìm tài liệu không tốn tiền và không rò rỉ dữ liệu ra ngoài;
Chroma để không phải dựng thêm dịch vụ; Claude Haiku vì nhanh và rẻ cho khối lượng câu hỏi Q&A;
LangChain chỉ dùng như lớp keo mỏng (chunk, gọi LLM), không phụ thuộc sâu.

---

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

copy .env.example .env              # rồi mở .env, điền ANTHROPIC_API_KEY
```

> Lấy API key tại <https://console.anthropic.com>.
> Lần chạy đầu tiên sẽ tải model embedding (~90 MB) về máy, sau đó dùng lại từ cache.

---

## Cách dùng

### 1. Giao diện web (khuyến nghị)

```bash
uvicorn app.api:app --reload
# mở http://127.0.0.1:8000
```

Đính kèm PDF bằng nút 📎, gõ câu hỏi, nhấn Gửi. Tài liệu được nạp tự động trong cùng lượt gửi.

### 2. Nạp tài liệu bằng dòng lệnh (không cần API key)

```bash
# đặt file PDF vào thư mục data/ rồi:
python -m app.ingest data/ten-tai-lieu.pdf     # nạp 1 file
python -m app.ingest                           # nạp mọi PDF trong data/
```

### 3. Hỏi nhanh trên terminal

```bash
python -m app.rag "Thuật toán X gồm mấy bước?"
python -m app.rag "tóm tắt tài liệu"          # chế độ tóm tắt (nếu chỉ có 1 file đã nạp)
```

### 4. Chạy bộ đánh giá (cần API key)

```bash
python -m eval.run_eval        # đọc eval/testset.json, in bảng số liệu, ghi eval/results.json
```

Muốn đánh giá trên tài liệu của bạn: thay các câu hỏi trong [eval/testset.json](eval/testset.json).

### 5. Test khói (không cần API key)

```bash
pytest -q
```

---

## Cấu hình

Mọi tham số nằm trong [config.py](config.py), ghi đè được qua biến môi trường hoặc file `.env`
(thứ tự ưu tiên: biến môi trường > `.env` > mặc định).

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Bắt buộc để sinh câu trả lời và chạy eval |
| `LLM_MODEL` | `claude-haiku-4-5` | Đổi sang `claude-sonnet-5` nếu cần chất lượng cao hơn (chậm & đắt hơn) |
| `CONFIDENCE_THRESHOLD` | `0.35` | Điểm similarity tối thiểu để **không** fallback tầng 1 |
| `TOP_K` | `4` | Số đoạn văn truy hồi cho mỗi câu hỏi |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Kích thước đoạn khi cắt tài liệu |
| `SPARSE_MEDIAN_THRESHOLD` | `700` | Nếu số ký tự trung vị mỗi trang thấp hơn mức này → coi là PDF trình chiếu, gộp slide trước khi chunk |
| `SPARSE_BLOCK_CHARS` | `600` | Kích thước tối thiểu của một khối slide sau khi gộp |
| `ENABLE_QUERY_REWRITE` | `true` | Cho Claude viết lại câu hỏi (tiếng Việt / có chỉ thị) thành truy vấn tìm kiếm trước khi retrieve |
| `SUMMARY_BATCH_CHARS` / `SUMMARY_MAX_BATCHES` | `6000` / `10` | Chế độ tóm tắt: kích thước mỗi lô "map" và số lô tối đa (tài liệu dài hơn → lấy mẫu đều) |

**Lưu ý khi đổi cấu hình liên quan tới truy hồi** (`CHUNK_SIZE`, `EMBEDDING_MODEL`, cách chunk…):
phải xóa thư mục `storage/chroma/` và chạy lại `python -m app.ingest` — dữ liệu cũ trong vector DB
không tự cập nhật theo cấu hình mới.

---

## Kết quả eval

Bộ test tự xây: **35 câu hỏi** — 25 câu *in-scope* (có đáp án trong 2 tài liệu mẫu) + 10 câu
*out-of-scope* (hỏi những thứ tài liệu không đề cập, để kiểm tra khả năng từ chối).
Cấu hình: model `claude-haiku-4-5`, embedding `all-MiniLM-L6-v2`, `top_k = 4`, ngưỡng `0.35`.
Chi tiết từng câu: [eval/results.json](eval/results.json).

| Chỉ số | Kết quả | Ý nghĩa |
|---|---|---|
| **Retrieval accuracy@4** | **96%** (24/25) | Đoạn chứa đáp án có nằm trong top-4 truy hồi không |
| **Groundedness** | **100%** (23/23) | Trong số câu được trả lời, câu nào cũng bám đúng ngữ cảnh (Claude chấm faithfulness kiểu RAGAS) |
| **Chống hallucination** | **10/10** | Câu ngoài phạm vi đều bị từ chối, không bịa |
| **In-scope được trả lời** | **92%** (23/25) | Câu có đáp án mà hệ thống thực sự trả lời (không từ chối nhầm) |
| **Latency** | TB **2.9s** · p95 **6.7s** | Thời gian mỗi câu (bao gồm truy hồi + gọi Claude) |

> **Về 2/25 câu in-scope bị từ chối:** retriever (MiniLM + chunk cố định) không xếp được đoạn
> chứa đáp án lên top-4. Hệ thống chọn **từ chối thay vì đoán bừa** — đúng thiết kế. Có thể cải
> thiện bằng hybrid search hoặc thêm re-ranker (nằm ngoài phạm vi project này).
>
> Bộ eval gồm câu hỏi tiếng Anh trên 2 tài liệu tiếng Anh, nên tính năng viết lại câu hỏi
> (tiếng Việt / có chỉ thị) không tác động ở đây — hiệu quả của nó được kiểm bằng các ca thử riêng.

---

## Giới hạn đã biết

- **PDF trình chiếu (slide):** khi số ký tự trung vị mỗi trang thấp hơn `SPARSE_MEDIAN_THRESHOLD`,
  hệ thống tự gộp các slide liền nhau thành khối lớn hơn trước khi chunk (nếu để nguyên từng
  slide thì chunk quá ngắn, embedding nhiễu, truy hồi trượt). **Đánh đổi:** trích dẫn sẽ trỏ tới
  slide đầu tiên của cụm liên quan, không phải slide chính xác.
- **Chữ nằm trong ảnh** (sơ đồ mạch, ảnh chụp màn hình, biểu đồ được xuất ra dạng ảnh): không đọc
  được — cần OCR hoặc mô hình thị giác, **ngoài phạm vi**. Với câu hỏi chỉ trả lời được từ nội
  dung dạng ảnh, hệ thống sẽ từ chối thay vì đoán.
- **Tóm tắt tài liệu dài:** với file rất dài (vượt `SUMMARY_MAX_BATCHES` lô), bản tóm tắt dựa
  trên các phần **lấy mẫu trải đều** khắp tài liệu, không phải toàn văn (có ghi chú trong câu trả lời).
- **Không có bộ nhớ hội thoại:** mỗi câu hỏi được xử lý độc lập, không hiểu câu hỏi nối tiếp kiểu
  "giải thích thêm ý số 2".
- **Chỉ nhận PDF có text trích được** (không nhận ảnh scan, Word, HTML…).

---

## Ngoài phạm vi (có chủ đích)

Agent / tool-calling, suy luận nhiều bước, OCR, hybrid search, re-ranker, xử lý domain nhạy cảm
(pháp lý / y tế / tài chính), deploy quy mô lớn, load testing.

---

## Cấu trúc thư mục

```
app/
  ingest.py       đọc PDF → (gộp slide thưa) → chunk → embed → lưu Chroma (chống trùng theo hash)
  embeddings.py   model embedding local (singleton)
  vectorstore.py  kết nối Chroma + liệt kê tài liệu đã nạp
  retriever.py    truy hồi top-k + tính confidence + lọc theo tài liệu + fallback tầng 1
  rag.py          điều phối: viết lại câu hỏi, định tuyến tóm tắt, gọi Claude, fallback tầng 2
  summarize.py    tóm tắt toàn tài liệu bằng map-reduce
  llm.py          khởi tạo client Claude (singleton)
  api.py          FastAPI: trang chat + /api/chat, /api/reset, /api/files, /api/info
  static/         giao diện chatbot (index.html)
eval/
  testset.json    35 câu hỏi (in-scope + out-of-scope)
  run_eval.py     chạy toàn bộ testset, tự chấm điểm, ghi results.json
tests/            test khói (ingest / retrieve / intent), không cần API key
config.py         toàn bộ tham số cấu hình
```
