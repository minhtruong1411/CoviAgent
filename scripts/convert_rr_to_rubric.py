"""
Chuyển file đáp án định dạng ROUND-ROBIN  ->  định dạng COMPARE-RUBRIC.

Round-robin lưu:   {label: [answer_str, ...]}            (chỉ có chuỗi đáp án)
Compare-rubric cần: {label: [{question, ground_truth, bot_answer, contexts}]}

Script ghép lại question + ground_truth theo VỊ TRÍ từ file câu hỏi gốc.
LƯU Ý: round-robin không lưu `contexts` -> đặt contexts = [] (rỗng). Vì vậy 2 tiêu
chí rubric phụ thuộc context (faithfulness, context_quality) sẽ KHÔNG đáng tin;
3 tiêu chí còn lại (correctness, completeness, relevance) vẫn đánh giá bình thường.

Usage:
  python scripts/convert_rr_to_rubric.py \
      --rr experiments/rr_answers.json \
      --questions data/gold_cases.json \
      --out experiments/answers_from_rr.json
"""

import sys
import json
import argparse
from pathlib import Path

# Cho phép import load_questions từ script benchmark để dùng ĐÚNG thứ tự/định dạng câu hỏi.
sys.path.append(str(Path(__file__).parent.parent))
from scripts.run_judge_benchmark import load_questions


def main():
    parser = argparse.ArgumentParser(description="Convert round-robin answers -> compare-rubric format")
    parser.add_argument("--rr", default="experiments/rr_answers.json", help="File đáp án round-robin")
    parser.add_argument("--questions", default="data/gold_cases.json", help="File câu hỏi gốc (đúng bộ đã dùng khi generate)")
    parser.add_argument("--out", default="experiments/answers_from_rr.json", help="File output định dạng compare-rubric")
    args = parser.parse_args()

    with open(args.rr, "r", encoding="utf-8") as f:
        rr = json.load(f)

    dataset = load_questions(args.questions)
    print(f"📝 {len(dataset)} câu hỏi từ {args.questions}")

    converted = {}
    for label, answers in rr.items():
        if len(answers) != len(dataset):
            print(f"⚠️  {label}: số đáp án ({len(answers)}) != số câu hỏi ({len(dataset)}). "
                  f"Ghép theo min(len) — kiểm tra lại file câu hỏi nếu lệch!")
        rows = []
        for item, ans in zip(dataset, answers):
            rows.append({
                "question": item.question,
                "ground_truth": item.ground_truth,
                "bot_answer": ans,
                "contexts": [],   # round-robin không lưu context
            })
        converted[label] = rows
        print(f"  ✅ {label}: {len(rows)} mục")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    print(f"💾 Đã ghi: {args.out}")
    print(f"\nGiờ chạy chấm rubric:")
    print(f"  python scripts/run_judge_benchmark.py --mode compare-rubric --phase judge \\")
    print(f"    --answers-file {args.out} \\")
    print(f"    --config configs/judge_benchmark.yaml --pipeline-type vector \\")
    print(f"    --config-b configs/judge_benchmark_kg.yaml --pipeline-type-b kg \\")
    print(f"    --config-c configs/judge_benchmark_graphrag.yaml --pipeline-type-c graphrag")


if __name__ == "__main__":
    main()
