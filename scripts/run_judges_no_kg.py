"""
Chạy benchmark KHÔNG bao gồm pipeline KG (chỉ Vector + GraphRAG).

Lý do: KG (GraphCypherQAChain) gọi LLM nhiều lần và thường dính lỗi rate-limit 429
của Groq, làm hỏng cả mẻ benchmark. Script này bỏ KG để mẻ chạy ổn định.

Đây là wrapper mỏng của scripts/run_all_judges.py — tái dùng toàn bộ cơ chế
xoay API key (.env) + cache resume, chỉ ép --pipelines = vector graphrag.

Usage (giống run_all_judges.py, chỉ khác là bỏ KG):
  python scripts/run_judges_no_kg.py --phase generate --questions data/golden_queries_multihop.json --answers-file experiments/answers_mh.json
  python scripts/run_judges_no_kg.py --phase judge    --questions data/golden_queries_multihop.json --answers-file experiments/answers_mh.json

Vẫn có thể tự chỉ định --pipelines để ghi đè (vd chỉ chạy lại graphrag):
  python scripts/run_judges_no_kg.py --phase generate --pipelines graphrag ...
"""

import os
import sys
import subprocess


def main():
    args = sys.argv[1:]

    # Nếu người dùng KHÔNG tự chỉ định --pipelines thì mặc định bỏ KG.
    if "--pipelines" not in args:
        args = args + ["--pipelines", "vector", "graphrag"]
    elif "kg" in args:
        print("⚠️  Bạn đã yêu cầu 'kg' trong --pipelines, nhưng script này dùng để BỎ KG. "
              "KG vẫn sẽ chạy theo yêu cầu của bạn.")

    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "run_all_judges.py")

    print("🚫 Chế độ KHÔNG-KG: chỉ chạy Vector + GraphRAG.\n")
    cmd = [sys.executable, target] + args
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
