import os
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["generate", "judge", "all"], default="judge", help="Phase to run")
    parser.add_argument("--pipelines", nargs="+", default=["vector", "kg", "graphrag"], help="Pipelines to run")
    parser.add_argument("--questions", default="data/gold_cases.json", help="Đường dẫn đến file bộ câu hỏi")
    parser.add_argument("--answers-file", default="experiments/answers_all.json", help="Đường dẫn lưu/đọc file kết quả (answers)")
    args = parser.parse_args()

    # Mỗi pipeline dùng ĐÚNG config của nó để đảm bảo benchmark CÂN BẰNG (parity):
    #   vector  → top_k=5            (≈ 5 đoạn prose)
    #   graphrag→ top_k=2 + đồ thị   (2 vector + 3 evidence ≈ 5 đoạn prose + 25 facts)
    #   kg      → top_k=5
    # Xem mục "Benchmark Cân Bằng" trong BENCHMARK_GUIDE.md.
    config_by_pipeline = {
        "vector": "configs/judge_benchmark.yaml",
        "kg": "configs/judge_benchmark_kg.yaml",
        "graphrag": "configs/judge_benchmark_graphrag.yaml",
    }

    # Đọc các khoá API Groq TỪ FILE .env để luân phiên (KHÔNG hardcode nữa).
    # Lấy mọi biến GROQ_API_KEY / GROQ_API_KEY_1..3 đang đặt, khử trùng, giữ thứ tự.
    from dotenv import load_dotenv
    load_dotenv()
    key_names = ["GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]
    keys, seen = [], set()
    for name in key_names:
        v = (os.environ.get(name) or "").strip()
        if v and v not in seen and not v.startswith("gsk_key"):  # bỏ placeholder
            seen.add(v)
            keys.append(v)
    if not keys:
        print("❌ Không tìm thấy khoá Groq nào trong .env "
              "(cần GROQ_API_KEY hoặc GROQ_API_KEY_1/2/3). Hãy thêm vào .env rồi chạy lại.")
        sys.exit(1)
    print(f"🔑 Dùng {len(keys)} khoá Groq từ .env để luân phiên.")

    for i, pipeline in enumerate(args.pipelines):
        print(f"\n{'='*50}")
        print(f"🚀 Running {args.phase.upper()} for {pipeline.upper()} with Key {i+1}")
        print(f"{'='*50}")
        
        # Thiết lập biến môi trường
        env = os.environ.copy()
        env["GROQ_API_KEY"] = keys[i % len(keys)]
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Câu lệnh CLI — chọn config theo từng pipeline để giữ parity
        cfg = config_by_pipeline.get(pipeline, "configs/judge_benchmark.yaml")
        cmd = [
            sys.executable,
            "scripts/run_judge_benchmark.py",
            "--mode", "compare-rubric",
            "--config", cfg,
            "--pipeline-type", pipeline,
            "--phase", args.phase,
            "--questions", args.questions,
            "--answers-file", args.answers_file
        ]
        
        try:
            # Chạy subprocess
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError:
            print(f"\n❌ Quá trình xử lý bị lỗi ở {pipeline.upper()}. Hệ thống sẽ bỏ qua và chạy tiếp pipeline sau nếu có.")
            continue
            
    print("\n🎉 Chạy tự động hoàn tất!")
    
    # In lại bảng xếp hạng cuối cùng từ cache để đảm bảo luôn nhìn thấy kết quả
    try:
        from coviagent.evaluation.judge_runner import JudgeRunner
        import json
        
        cache_file = args.answers_file.replace(".json", "_cache.json")
        if os.path.exists(cache_file):
            print(f"\n📊 BẢNG XẾP HẠNG TỔNG KẾT TỪ CACHE ({cache_file}):")
            with open(cache_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            
            # Tính toán lại summary và in
            runner = JudgeRunner(None)
            combined_rubric_results = []
            summary_by_pipeline = {}
            for label, results in all_results.items():
                combined_rubric_results.extend(results)
                summary_by_pipeline[label] = runner._compute_summary_metrics(results)
            
            ranking = sorted(list(all_results.keys()), key=lambda x: summary_by_pipeline[x].get("avg_overall_score", 0), reverse=True)
            summary = {
                "total_questions": len(combined_rubric_results) // len(all_results) if all_results else 0,
                "pipelines_evaluated": list(all_results.keys()),
                "ranking": ranking,
                "pipeline_summaries": summary_by_pipeline
            }
            runner.print_compare_leaderboard(summary)
    except Exception as e:
        pass

if __name__ == "__main__":
    main()
