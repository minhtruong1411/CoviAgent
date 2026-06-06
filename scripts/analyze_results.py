"""
Gộp & phân tích kết quả benchmark từ MỘT hoặc NHIỀU file cache (rubric).

Cho phép chạy nhiều tập câu hỏi riêng rẽ rồi cộng dồn kết quả thành một báo cáo
(per-tiêu chí + per-category + leaderboard). Hữu ích để gộp tập 1 + tập 2 → N lớn hơn.

Usage:
  # Một tập
  python scripts/analyze_results.py --cache experiments/answers_mh_cache.json --gold data/golden_queries_multihop.json

  # Gộp tập 1 + tập 2 (N=30)
  python scripts/analyze_results.py \
    --cache experiments/answers_mh_cache.json experiments/answers_set2_cache.json \
    --gold data/golden_queries_multihop.json data/golden_queries_set2.json
"""

import json
import argparse
from collections import defaultdict

CRIT = ["correctness", "completeness", "relevance", "faithfulness", "context_quality"]


def _fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0


def _overall(row):
    return sum(_fnum(row.get(c, 0)) for c in CRIT) / len(CRIT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True, help="Một hoặc nhiều file *_cache.json")
    ap.add_argument("--gold", nargs="+", default=[], help="File gold tương ứng (để lấy category)")
    args = ap.parse_args()

    # map câu hỏi -> category từ tất cả file gold
    q2cat = {}
    for g in args.gold:
        for item in json.load(open(g, encoding="utf-8")):
            q2cat[item["question"]] = item.get("category", "?")

    # gộp rows theo pipeline (dedupe theo câu hỏi để tránh đếm trùng)
    merged = defaultdict(list)
    seen = defaultdict(set)
    for cf in args.cache:
        data = json.load(open(cf, encoding="utf-8"))
        for label, rows in data.items():
            for r in rows:
                q = r.get("question", "")
                if q in seen[label]:
                    continue
                seen[label].add(q)
                merged[label].append(r)

    n_q = max((len(v) for v in merged.values()), default=0)
    print(f"\n{'='*78}")
    print(f"📊 BÁO CÁO GỘP — {len(args.cache)} tập, {n_q} câu/pipeline, {len(merged)} pipeline")
    print(f"   Nguồn: {', '.join(args.cache)}")
    print(f"{'='*78}")

    # bảng tiêu chí
    print(f"\n{'Pipeline':<10}|{'Corr':>6}|{'Comp':>6}|{'Rel':>6}|{'Faith':>6}|{'Ctx':>6}|{'OVERALL':>8}|{'Pass%':>6}")
    print("-" * 78)
    overall = {}
    for label, rows in merged.items():
        a = {c: sum(_fnum(r.get(c, 0)) for r in rows) / len(rows) for c in CRIT}
        ov = sum(a.values()) / len(CRIT)
        overall[label] = ov
        npass = sum(1 for r in rows if str(r.get("final_decision", "")).lower().startswith("pass"))
        print(f"{label:<10}|{a['correctness']:>6.2f}|{a['completeness']:>6.2f}|{a['relevance']:>6.2f}"
              f"|{a['faithfulness']:>6.2f}|{a['context_quality']:>6.2f}|{ov:>8.2f}|{100*npass/len(rows):>5.0f}%")

    ranking = sorted(overall, key=overall.get, reverse=True)
    print("-" * 78)
    print("🏆 Xếp hạng: " + " > ".join(f"{l} ({overall[l]:.2f})" for l in ranking))

    # bảng theo category
    cats = sorted({c for c in q2cat.values()})
    if cats:
        print(f"\n{'─'*78}\nTHEO CATEGORY (điểm overall 0-5)\n{'─'*78}")
        print(f"{'Pipeline':<10}|" + "|".join(f"{c[:12]:>13}" for c in cats))
        for label, rows in merged.items():
            by = defaultdict(list)
            for r in rows:
                by[q2cat.get(r.get("question", ""), "?")].append(_overall(r))
            cells = []
            for c in cats:
                v = by.get(c, [])
                cells.append(f"{(sum(v)/len(v) if v else 0):>13.2f}")
            print(f"{label:<10}|" + "|".join(cells))
        # đếm số câu mỗi category
        cnt = defaultdict(int)
        any_rows = next(iter(merged.values()), [])
        for r in any_rows:
            cnt[q2cat.get(r.get("question", ""), "?")] += 1
        print(f"{'(số câu)':<10}|" + "|".join(f"{cnt.get(c,0):>13}" for c in cats))
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
