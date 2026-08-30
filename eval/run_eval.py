"""Chạy bộ đánh giá định lượng cho hệ thống RAG.

Đo:
  1. Retrieval accuracy@k     - top-k chunk có chứa thông tin liên quan không
  2. Groundedness             - câu trả lời có bám sát ngữ cảnh (Claude làm giám khảo faithfulness)
  3. Chống hallucination      - câu ngoài phạm vi có bị từ chối đúng không
  4. In-scope answered        - câu trong tài liệu có được trả lời (không fallback nhầm)
  5. Latency                  - thời gian trung bình & p95 mỗi câu

Kết quả chi tiết ghi ra eval/results.json.
Yêu cầu: đã ingest PDF + có ANTHROPIC_API_KEY trong .env.
Chạy:  python -m eval.run_eval
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from app.llm import get_llm
from app.rag import RetrievedChunk, answer_question
from config import settings

TESTSET = Path(__file__).parent / "testset.json"


def _retrieval_hit(
    chunks: list[RetrievedChunk], keywords: list[str], pages: list[int]
) -> bool:
    """Đoạn chứa đáp án có nằm trong top-k đã retrieve không (khớp keyword hoặc trang).

    Nhận thẳng `chunks` từ `answer_question` để phản ánh đúng truy vấn thật
    (đã qua bước viết lại câu hỏi), không retrieve lại bằng câu gốc.
    """
    for c in chunks:
        text = c.text.lower()
        if any(kw.lower() in text for kw in keywords):
            return True
        if c.page is not None and c.page in pages:
            return True
    return False


def _is_grounded(question: str, answer: str, context: str) -> bool:
    """Chấm faithfulness (kiểu RAGAS): mọi khẳng định trong câu trả lời có
    suy ra được từ NGỮ CẢNH không. Cho phép diễn đạt lại / tóm tắt / sắp xếp lại;
    chỉ chấm NOT_GROUNDED khi có khẳng định bịa thêm hoặc mâu thuẫn với ngữ cảnh.
    """
    verdict = str(
        get_llm().invoke(
            "Bạn đánh giá tính TRUNG THỰC (faithfulness) của một câu trả lời RAG.\n"
            "GROUNDED nếu MỌI khẳng định thực tế trong câu trả lời đều suy ra được từ "
            "NGỮ CẢNH (được phép diễn đạt lại, tóm tắt, thêm cấu trúc/đánh số).\n"
            "NOT_GROUNDED nếu có ít nhất một khẳng định mâu thuẫn với ngữ cảnh, hoặc "
            "không thể tìm thấy cơ sở nào trong ngữ cảnh (bịa thêm).\n"
            "Bỏ qua mọi khác biệt về cách diễn đạt, thứ tự, định dạng, ngôn ngữ.\n"
            "Trả lời DUY NHẤT một từ: GROUNDED hoặc NOT_GROUNDED.\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\nCÂU TRẢ LỜI: {answer}\n\nPhán quyết:"
        ).content
    ).lower()
    return "not_grounded" not in verdict and "grounded" in verdict


def main() -> None:
    questions = json.loads(TESTSET.read_text(encoding="utf-8"))["questions"]
    k = settings.top_k

    retr_hits = retr_total = 0
    grnd_hits = grnd_total = 0
    oos_total = oos_refused = 0       # out-of-scope: từ chối đúng (chống hallucination)
    in_total = in_answered = 0        # in-scope: trả lời (không fallback nhầm)
    latencies: list[float] = []

    rows: list[dict] = []
    for q in questions:
        ans = answer_question(q["question"])
        latencies.append(ans.latency_s)
        out_of_scope = q["type"] == "out_of_scope"
        retr_hit = grounded = None

        if out_of_scope:
            oos_total += 1
            oos_refused += ans.is_fallback
        else:
            in_total += 1
            in_answered += not ans.is_fallback

            # 1. Retrieval accuracy
            retr_total += 1
            retr_hit = _retrieval_hit(
                ans.chunks,
                q.get("relevant_keywords", []),
                q.get("relevant_pages", []),
            )
            retr_hits += retr_hit

            # 2. Groundedness (chỉ tính khi thực sự có trả lời)
            if not ans.is_fallback:
                grnd_total += 1
                ctx = "\n\n".join(c.text for c in ans.chunks)
                grounded = _is_grounded(q["question"], ans.answer, ctx)
                grnd_hits += grounded

        tag = "FB" if ans.is_fallback else "OK"
        print(f"[{tag}] {ans.latency_s:5.1f}s  {q['question'][:66]}")
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "is_fallback": ans.is_fallback,
                "confidence": round(ans.confidence, 3),
                "latency_s": round(ans.latency_s, 2),
                "retrieval_hit": retr_hit,
                "grounded": grounded,
                "answer": ans.answer,
                "sources": ans.sources,
            }
        )

    p95 = sorted(latencies)[max(0, round(len(latencies) * 0.95) - 1)]
    fb_correct = oos_refused + in_answered

    print("\n===================== KẾT QUẢ =====================")
    if retr_total:
        print(f"Retrieval accuracy@{k}        : {retr_hits}/{retr_total} = {retr_hits / retr_total:.0%}")
    if grnd_total:
        print(f"Groundedness (faithfulness) : {grnd_hits}/{grnd_total} = {grnd_hits / grnd_total:.0%}")
    print(f"Chống hallucination (OOS)   : {oos_refused}/{oos_total} câu ngoài phạm vi bị từ chối đúng")
    print(f"In-scope được trả lời       : {in_answered}/{in_total} = {in_answered / in_total:.0%}")
    print(f"Fallback đúng tổng          : {fb_correct}/{len(questions)} = {fb_correct / len(questions):.0%}")
    print(f"Latency                     : TB {statistics.mean(latencies):.2f}s | p95 {p95:.2f}s")
    print("==================================================")

    summary = {
        "model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "top_k": k,
        "confidence_threshold": settings.confidence_threshold,
        "retrieval_accuracy_at_k": round(retr_hits / retr_total, 3) if retr_total else None,
        "groundedness": round(grnd_hits / grnd_total, 3) if grnd_total else None,
        "oos_refused": f"{oos_refused}/{oos_total}",
        "in_scope_answered": f"{in_answered}/{in_total}",
        "latency_mean_s": round(statistics.mean(latencies), 2),
        "latency_p95_s": round(p95, 2),
    }
    out = Path(__file__).parent / "results.json"
    out.write_text(
        json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nĐã lưu chi tiết -> {out}")


if __name__ == "__main__":
    main()
