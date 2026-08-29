# Project: Technical Document Q&A Assistant (RAG + Fallback + Eval)

## 1. Mục tiêu project

Xây dựng một hệ thống hỏi-đáp trên tài liệu kỹ thuật (PDF), sử dụng kiến trúc RAG, có khả năng nhận biết khi nào không có đủ thông tin để trả lời (tránh hallucination), và có bộ đánh giá định lượng để chứng minh hệ thống hoạt động tốt tới đâu — không chỉ "chạy được" mà còn "đo được".

## 2. Phạm vi (Scope)

**Có trong scope:**
- Ingest một hoặc nhiều file PDF
- Pipeline RAG đầy đủ: chunk → embed → lưu vector DB → retrieve → generate câu trả lời kèm trích dẫn nguồn (trang/đoạn)
- Cơ chế fallback: phát hiện câu hỏi ngoài phạm vi tài liệu, trả lời "không tìm thấy thông tin" thay vì bịa
- Bộ eval tự xây: đo retrieval accuracy và groundedness của câu trả lời
- Giao diện tối giản (Streamlit hoặc form HTML qua FastAPI)

**Ngoài scope:**
- Agent/tool-calling, multi-step reasoning
- Xử lý dữ liệu scan/OCR, hybrid search, re-ranker, domain phức tạp (legal/medical/finance)
- Deploy production quy mô lớn, load testing

## 3. Yêu cầu chức năng (Functional Requirements)

1. Upload và parse được PDF, chia thành chunk hợp lý (có overlap, không cắt giữa câu nếu tránh được)
2. Convert chunk → vector embedding, lưu vào vector database
3. Nhận câu hỏi, retrieve top-k chunk liên quan nhất
4. Sinh câu trả lời từ LLM dựa trên context retrieve được, kèm trích dẫn nguồn
5. Tính điểm confidence cho câu trả lời (dựa trên độ tương đồng similarity score của top-k chunk)
6. Nếu confidence dưới ngưỡng hoặc không tìm thấy chunk liên quan → trả lời fallback ("không tìm thấy thông tin trong tài liệu"), không bịa
7. Tự xây bộ test set 15-20 câu hỏi (gồm cả câu có trong tài liệu và câu ngoài phạm vi)
8. Đo retrieval accuracy@k trên bộ test
9. Đo groundedness (câu trả lời có bám sát context retrieve được không)
10. Đo latency trung bình mỗi câu trả lời

## 4. Tech Stack

- **Backend/API:** FastAPI
- **Orchestration:** LangChain
- **Vector DB:** Chroma hoặc FAISS
- **Embedding model:** OpenAI `text-embedding-3-small`, hoặc `sentence-transformers`
- **LLM:** OpenAI API hoặc Anthropic API
- **Confidence/Fallback logic:** tự viết dựa trên similarity score threshold
- **Eval:** bộ test tự viết (chấm tay/script so sánh) hoặc RAGAS
- **Giao diện:** Streamlit
- **(Tùy chọn) Container hóa:** Docker

## 5. Kết quả cần đạt (Outcome / Deliverables)

- Repo GitHub public, README rõ ràng (mô tả kiến trúc, cách chạy, kết quả eval)
- Demo chạy được (video ngắn hoặc link deploy)
- Số liệu cụ thể cho CV:
  - Retrieval accuracy@3: X% trên bộ test tự tạo
  - Groundedness: Y% câu trả lời bám đúng context
  - Fallback hoạt động đúng trên Z/Z câu hỏi ngoài phạm vi tài liệu (không hallucination)
  - Latency trung bình: X giây/câu trả lời
- Bullet CV mẫu (viết sau khi có số liệu thật): "Built a RAG-based document Q&A system with confidence-based fallback to prevent hallucination; achieved X% retrieval accuracy and Y% groundedness on a self-built 20-question eval set"

## 6. Thời gian ước tính

~7-10 ngày:
- Nền RAG: 3-5 ngày
- Fallback logic: 2-3 ngày
- Eval + đo số liệu + README: 2-3 ngày
