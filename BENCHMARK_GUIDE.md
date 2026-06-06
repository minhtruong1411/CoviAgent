# Hướng Dẫn Benchmark Hệ Thống CoviAgent

Hệ thống Benchmark của CoviAgent được thiết kế để đánh giá tự động và tự động xếp hạng các phương pháp RAG khác nhau (như `VECTOR`, `KG`, `GRAPHRAG`) bằng phương pháp **LLM-as-a-Judge** (sử dụng một LLM lớn đóng vai trò làm giám khảo chấm điểm).

---

## 1. Cơ chế hoạt động

Hệ thống chấm điểm dựa trên **5 tiêu chí chuẩn** của RAG (mỗi tiêu chí từ 0 đến 5 điểm):
- **Correctness (Độ chính xác)**
- **Completeness (Độ đầy đủ)**
- **Relevance (Độ liên quan)**
- **Faithfulness (Độ trung thực - bám sát context)**
- **Context Quality (Chất lượng tài liệu truy xuất được)**

Hệ thống được chia làm 2 giai đoạn (Phase) độc lập để tối ưu chi phí và tránh lỗi API:
1. **Generate**: Chỉ gọi các pipeline nội bộ để truy xuất dữ liệu và sinh câu trả lời, sau đó lưu toàn bộ vào file `.json`.
2. **Judge**: LLM Giám khảo đọc file `.json` vừa sinh, chấm điểm từng câu, tính trung bình và xuất ra Bảng Xếp Hạng.

---

## 2. Cách chạy Benchmark Cơ Bản

Cách đơn giản và an toàn nhất là sử dụng script tự động hóa `run_all_judges.py`. Script này được thiết kế với cơ chế **Tự động thay đổi API Key** (xoay vòng 3 API Key) và **Fault-tolerant (tự động lưu cache)** để vượt qua mọi giới hạn Rate Limit của các dịch vụ miễn phí (như Groq).

### 2.1 Sinh câu trả lời (Generate Phase)
Lệnh này sẽ tự động chạy tạo câu trả lời cho cả 3 phương pháp và lưu vào `experiments/answers_all.json`.
```powershell
python scripts/run_all_judges.py --phase generate
```

### 2.2 Chấm điểm (Judge Phase)
Lệnh này sẽ lấy kết quả từ file `answers_all.json` để tiến hành chấm điểm và in ra Bảng Xếp Hạng.
```powershell
python scripts/run_all_judges.py --phase judge
```
*(Nếu bạn gõ `python scripts/run_all_judges.py` không kèm tham số, nó sẽ mặc định chạy phase `judge`).*

> **Lưu ý (parity):** `run_all_judges.py` tự động chọn **config riêng cho từng pipeline** (`vector → judge_benchmark.yaml`, `graphrag → judge_benchmark_graphrag.yaml`, `kg → judge_benchmark_kg.yaml`) để các pipeline được benchmark trên lượng ngữ cảnh tương đương. Chi tiết cơ chế cân bằng xem **Mục 7**.

---

## 3. Khôi phục sau lỗi (Fault Tolerance)

Nếu quá trình Judge bị ngắt hoặc báo lỗi `Rate Limit` (quá giới hạn Token của API), **bạn không cần lo lắng hay làm lại từ đầu**. 

Hệ thống đã tự động lưu trữ các phương pháp đã chấm xong vào một file cache (ví dụ: `experiments/answers_all_cache.json`).
Để chạy tiếp, bạn **chỉ cần chạy lại đúng câu lệnh cũ**:
```powershell
python scripts/run_all_judges.py
```
Hệ thống sẽ tự động in ra dòng chữ `✅ Skipping VECTOR (already evaluated in cache)` và chỉ bắt đầu chạy lại những phương pháp chưa hoàn thành. Bảng xếp hạng cuối cùng vẫn sẽ tổng hợp đầy đủ các phương pháp.

---

## 4. Chạy với các bộ câu hỏi (Dataset) khác nhau

Nếu bạn có nhiều bộ câu hỏi (Golden Queries) khác nhau và muốn đánh giá song song mà không làm ghi đè (overwrite) dữ liệu cũ, bạn có thể truyền thêm đường dẫn file đầu vào và file đầu ra:

**Ví dụ, với bộ dataset thứ 2:**
1. Sinh đáp án:
```powershell
python scripts/run_all_judges.py --phase generate --questions experiments/golden_queries_v2.json --answers-file experiments/answers_v2.json
```
2. Chấm điểm:
```powershell
python scripts/run_all_judges.py --phase judge --questions experiments/golden_queries_v2.json --answers-file experiments/answers_v2.json
```
Lúc này, file cache cũng sẽ được tự động đổi tên thành `experiments/answers_v2_cache.json` để hoàn toàn tách biệt.

---

## 5. Ép chạy (Force Run) một phương pháp duy nhất

Nếu bạn chỉ muốn sửa/sinh lại duy nhất phương pháp `graphrag` vì cấu hình của nó vừa thay đổi:
```powershell
# Sinh lại đáp án cho riêng graphrag
python scripts/run_all_judges.py --phase generate --pipelines graphrag
```
Hệ thống sẽ sinh lại riêng `graphrag` và tự động gộp (merge) hoặc đắp thêm vào file `answers_all.json` mà không xóa mất dữ liệu của `vector` và `kg` đã sinh trước đó.

**Để chấm điểm (Judge) lại duy nhất phương pháp đó:**
```powershell
python scripts/run_all_judges.py --phase judge --pipelines graphrag
```
*Lưu ý: Hệ thống chấm điểm sử dụng Cache. Nếu trước đó bạn đã chấm xong `graphrag` và độ dài câu hỏi không đổi, nó có thể báo `Skipping...`. Để ép nó chấm lại từ đầu, bạn chỉ cần vào file `experiments/answers_all_cache.json` xóa đoạn dữ liệu của `graphrag` đi, hoặc xóa hẳn file cache đó nếu muốn chấm lại toàn bộ.*

---

## 6. Hướng dẫn làm việc nhóm (Scale lên tập dữ liệu lớn)

Nếu bạn muốn chấm một bộ dữ liệu lớn (ví dụ: 80-100 câu) cho 3 phương pháp, tổng số Token mà mô hình Judge tiêu thụ sẽ vượt quá giới hạn miễn phí 100,000 Token/ngày của 1 tài khoản Groq.
Giải pháp tối ưu là **Chia để trị (Distributed Benchmark)** cho nhiều thành viên cùng chạy.

### Bước 6.1: Nhóm trưởng chia nhỏ dữ liệu
1. Chia file `gold_cases.json` 80 câu thành 4 file nhỏ, mỗi file 20 câu (VD: `gold_cases_p1.json`, `gold_cases_p2.json`...).
2. Gửi mỗi file cho một thành viên.

### Bước 6.2: Các thành viên tự chạy trên máy cá nhân
Mỗi thành viên cập nhật API Key Groq cá nhân vào file `.env` trên máy mình. Sau đó chạy 2 lệnh sau với tên file tương ứng (VD với thành viên nhận phần 1):

```powershell
# 1. Sinh đáp án cho 20 câu của mình
python scripts/run_all_judges.py --phase generate --questions data/gold_cases_p1.json --answers-file experiments/answers_p1.json

# 2. Chấm điểm cho 20 câu của mình
python scripts/run_all_judges.py --phase judge --questions data/gold_cases_p1.json --answers-file experiments/answers_p1.json
```
*(Nếu quá trình bị gián đoạn do lỗi mạng, thành viên chỉ cần chạy lại đúng lệnh cũ, hệ thống sẽ tự động dùng Cache để chạy tiếp).*

### Bước 6.3: Nộp bài và Gộp điểm
1. Sau khi chạy xong, mỗi thành viên sẽ thu được một file Cache kết quả, ví dụ: `experiments/answers_p1_cache.json`.
2. Mọi người gửi file Cache này cho Nhóm trưởng.
3. Nhóm trưởng sẽ dùng một Script Gộp Điểm (Merge Script) để nối 4 file Cache lại thành 1 Bảng Xếp Hạng cuối cùng hoàn chỉnh cho cả 80 câu.
*(Bằng cách này, nhóm bạn có thể Scale lên hàng nghìn câu hỏi mà không tốn một xu phí API nào!)*

---

## 7. Benchmark Cân Bằng (Fair Comparison): Vector RAG vs GraphRAG

### 7.1 Vì sao cần "cân bằng"?
GraphRAG nạp vào LLM **nhiều hơn** Vector RAG: ngoài các chunk vector, nó còn thêm **chunk chứng cứ từ đồ thị** và một khối **facts có cấu trúc** (các bộ ba `chủ thể -[quan hệ]-> đối tượng`). Nếu để Vector RAG `top_k` quá thấp, GraphRAG thắng hiển nhiên vì *đọc được nhiều thông tin hơn* — kết quả này **không chứng minh được** phương pháp đồ thị tốt hơn.

Vì vậy ta cân **lượng ngữ cảnh prose (đoạn văn) ngang nhau**, rồi để **lớp facts đồ thị** là điểm khác biệt duy nhất — chính là thứ ta muốn đo.

### 7.2 Cấu hình đã ghim sẵn

| Pipeline | Config | vector `top_k` | Tham số đồ thị | ≈ đoạn prose |
|---|---|---|---|---|
| **Vector RAG** (① số liệu chính) | `judge_benchmark.yaml` | **5** | — | ~5 |
| **GraphRAG** | `judge_benchmark_graphrag.yaml` | **2** | `expand_hops=2`, `max_graph_chunks=3`, `max_facts=25` | ~5 (2 vector + 3 evidence) + 25 facts |
| **Vector RAG** (② robustness) | `judge_benchmark_method_b.yaml` | **8** | — | ~8 |

Tham số đồ thị được khai báo ở mục `graphrag:` trong file config và **ghim cố định** (không đổi giữa các lần chạy) để kết quả tái lập được và khớp với demo trên `app.py`.

### 7.3 Bước 1 — Kiểm tra parity (KHÔNG tốn LLM)
Trước khi chấm điểm, hãy xác minh các pipeline thực sự thấy lượng ngữ cảnh tương đương. Mode `parity` chỉ chạy phần truy xuất (vector + đồ thị) và in bảng so sánh, **không gọi LLM nên miễn phí**:

```powershell
python scripts/run_judge_benchmark.py --mode parity `
  --config configs/judge_benchmark.yaml --pipeline-type vector `
  --config-b configs/judge_benchmark_graphrag.yaml --pipeline-type-b graphrag `
  --config-c configs/judge_benchmark_method_b.yaml --pipeline-type-c vector
```

Kết quả mẫu (8 câu hỏi):
```
  Pipeline               |  Đoạn prose TB |   Facts TB |     Ký tự TB
  ------------------------------------------------------------------
  VECTOR                 |            5.0 |        0.0 |        7,751
  GRAPHRAG               |            5.0 |       25.0 |        7,173
  VECTOR_C               |            8.0 |        0.0 |       12,253
```
→ Vector (7,751) và GraphRAG (7,173) gần bằng nhau về số ký tự ⇒ **parity đạt**. Nếu sau này đổi tham số, chạy lại lệnh này để chắc chắn ngữ cảnh vẫn cân.

### 7.4 Bước 2 — ① Số liệu chính (round-robin, 1 giám khảo chung)
Mode `round-robin` dùng **một giám khảo duy nhất** (lấy từ config đầu tiên) để chấm cả 3 pipeline ⇒ chấm hoàn toàn công bằng. Pipeline được dựng từ config riêng nên giữ đúng parity:

```powershell
python scripts/run_judge_benchmark.py --mode round-robin `
  --config configs/judge_benchmark.yaml --pipeline-type vector `
  --config-b configs/judge_benchmark_kg.yaml --pipeline-type-b kg `
  --config-c configs/judge_benchmark_graphrag.yaml --pipeline-type-c graphrag
```

### 7.5 Bước 3 — ② Kiểm chứng robustness (chống phản biện "đọc nhiều chữ hơn")
Cho Vector RAG **dư** ngữ cảnh (`top_k=8`) rồi so tay đôi với GraphRAG. Nếu GraphRAG **vẫn thắng** dù đọc ít chữ hơn → ưu thế đến từ **cấu trúc đồ thị**, không phải khối lượng văn bản. Đây là luận điểm mạnh nhất:

```powershell
python scripts/run_judge_benchmark.py --mode pairwise `
  --config configs/judge_benchmark_method_b.yaml --pipeline-type vector `
  --config-b configs/judge_benchmark_graphrag.yaml --pipeline-type-b graphrag
```

### 7.6 Cách đọc kết quả & lưu ý
- **`completeness` (đầy đủ)** thường là nơi GraphRAG vượt trội nhất, do facts đồ thị gom đủ các mục liên quan (vd: liệt kê đủ triệu chứng/yếu tố nguy cơ).
- **`faithfulness` (trung thực)**: theo dõi để chắc lớp facts không gây "ảo giác" — nếu điểm này tụt khi tăng `max_facts`, hãy giảm `max_facts` xuống.
- **Cùng mô hình sinh**: cả 3 pipeline (Vector, GraphRAG, **KG**) đều dùng cùng LLM sinh **Groq Llama-3.3-70B**, nên chênh lệch điểm là do *retrieval*, không do *generator*. (Lưu ý kỹ thuật: `KGRAGPipeline` luôn dùng `ChatGroq` và **bỏ qua `provider`** trong YAML — chỉ đọc `model_name` — nên `model_name` trong `judge_benchmark_kg.yaml` **bắt buộc** phải là model hợp lệ trên Groq.)
- **Một giám khảo chung & KHÁC model với generator**: block `judge:` để **giống hệt nhau** ở cả 4 config (cùng model, `num_runs=3` majority-vote, cùng `temperature`/`delay`/`threshold`). Giám khảo dùng **model khác generator** (generator = Llama-3.3-70B → giám khảo = `openai/gpt-oss-120b` trên Groq) để tránh *self-preference bias* (LLM chấm cao cho văn phong của chính nó). Ở mode `round-robin`/`pairwise`, giám khảo còn được build **một lần duy nhất** từ config đầu tiên → đảm bảo tuyệt đối.
- Muốn đổi mức cân bằng, chỉ cần sửa `retriever.top_k` trong các config và mục `graphrag:` của `judge_benchmark_graphrag.yaml`, rồi chạy lại **Bước 1** để xác nhận parity.

---

## 8. Quy chuẩn Benchmark Học thuật (để kết quả đáng tin & bảo vệ được)

Một benchmark "đúng chuẩn" phải kiểm soát biến và minh bạch. Checklist áp dụng cho dự án này:

### 8.1 Nguyên tắc kiểm soát biến — chỉ khác **một biến** (retrieval)
| Biến | Phải giữ cố định | Trạng thái |
|---|---|---|
| LLM sinh đáp án (generator) | cùng model cho 3 pipeline | ✅ Groq Llama-3.3-70B |
| Lượng ngữ cảnh (token) | parity giữa các pipeline | ✅ xem Mục 7 |
| `temperature` | = 0 (xác định) | ✅ |
| Giám khảo (model + giao thức) | một giám khảo, **khác model generator** | ✅ `openai/gpt-oss-120b`, `num_runs=3` |
| Dataset + prompt chấm | giống nhau | ✅ |

### 8.2 Tránh self-preference bias
**Giám khảo KHÔNG được trùng model với generator.** Ở đây generator là Llama-3.3-70B nên giám khảo dùng `openai/gpt-oss-120b` (khác họ, mạnh hơn). Đây là lỗi học thuật phổ biến nhất của LLM-as-Judge — đã được xử lý.

### 8.3 Kiểm định độ tin cậy của giám khảo (làm TRƯỚC khi tin điểm)
Cho giám khảo chấm các đáp án đã biết đúng/sai; nếu đồng thuận ≥ 80% mới đáng tin:
```powershell
python scripts/run_judge_benchmark.py --mode gold-validate `
  --config configs/judge_benchmark.yaml --gold-cases data/gold_cases.json
```

### 8.4 Dataset đủ lớn & phân tầng
- **Số lượng:** ≥ 50–100 câu (N=10 chỉ để thử nghiệm, không đủ kết luận).
- **Phân tầng theo `category`** trong `gold_cases.json`:
  - *one-hop / factoid* → kỳ vọng Vector ≈ GraphRAG.
  - *aggregation / multi-hop* (liệt kê, tổng hợp) → kỳ vọng GraphRAG vượt trội.
- **Báo cáo điểm TÁCH theo từng tầng** — đây là bằng chứng thuyết phục nhất cho thấy GraphRAG mạnh ở đâu.

### 8.5 Chấm bằng 2 chế độ bổ trợ
- **Rubric 0–5** (`compare-rubric`): điểm tuyệt đối theo 5 tiêu chí → *chẩn đoán mạnh/yếu ở đâu*.
- **Pairwise mù** (`round-robin`): win-rate tương đối, không lệ thuộc cách neo thang điểm → *ai thắng*.

### 8.6 Quy trình chuẩn (tóm tắt)
```
1. gold-validate  → giám khảo đạt ≥ 80% mới chạy tiếp
2. generate (1 lần, lưu answer + contexts)   →  run_all_judges.py --phase generate
3. judge:
   - rubric:    run_all_judges.py --phase judge
   - pairwise:  run_judge_benchmark.py --mode round-robin --phase judge
4. Báo cáo: trung bình từng tiêu chí + TÁCH theo category + win-rate;
            ghi rõ N, model generator, model giám khảo, cấu hình parity.
```

> **Lưu ý reproducibility:** mỗi lần đổi tham số (top_k, max_facts, model giám khảo...) đều phải chạy lại **`--mode parity`** (Mục 7.3) và **`gold-validate`** (8.3) rồi mới generate/judge, và ghi lại cấu hình kèm kết quả.

---

## 9. Các script tiện ích & quy trình thực dụng

### 9.1 Tách Generate / Judge (an toàn quota)
Mọi script đều hỗ trợ `--phase`. **Luôn tách 2 phase** để tiết kiệm và chịu lỗi tốt:
- **`--phase generate`**: chỉ chạy pipeline sinh đáp án + lưu file `.json`. **KHÔNG gọi giám khảo** → không tốn token judge.
- **`--phase judge`**: đọc đáp án đã lưu rồi chấm. Đây là nơi tốn token judge; **có cache resume** (hết quota giữa chừng → chạy lại đúng lệnh là chấm tiếp).

### 9.2 Benchmark KHÔNG có KG — `run_judges_no_kg.py`
KG-RAG (Text-to-Cypher) tốn nhiều lượt gọi LLM và dễ dính rate-limit/Cypher rỗng, làm hỏng cả mẻ. Script này chạy **chỉ Vector + GraphRAG** (bỏ KG), tái dùng nguyên cơ chế xoay key `.env` + cache của `run_all_judges.py`:
```powershell
python scripts/run_judges_no_kg.py --phase generate --questions data/golden_queries_multihop.json --answers-file experiments/answers_mh.json
python scripts/run_judges_no_kg.py --phase judge    --questions data/golden_queries_multihop.json --answers-file experiments/answers_mh.json
```
Vẫn ghi đè được `--pipelines` (vd chạy lại riêng graphrag):
```powershell
python scripts/run_judges_no_kg.py --phase generate --pipelines graphrag --questions data/golden_queries_multihop.json --answers-file experiments/answers_mh.json
```

### 9.3 Chạy KG RIÊNG (khi cần số KG)
Vì KG hay lỗi, hãy chạy nó **tách hẳn** khỏi Vector/GraphRAG, dùng `--pipelines kg`:
```powershell
python scripts/run_all_judges.py --phase generate --pipelines kg --questions data/golden_queries_multihop.json --answers-file experiments/answers_kg.json
python scripts/run_all_judges.py --phase judge    --pipelines kg --questions data/golden_queries_multihop.json --answers-file experiments/answers_kg.json
```
- KG cần **Neo4j đang chạy** (ở dự án này là Neo4j Aura cloud — xem `NEO4J_URI` trong `.env`).
- KG sinh đáp án bằng generator Groq llama-3.3-70b (model riêng, TPD tách với judge).
- Bình thường ~60% câu trả "Tôi không biết" do Cypher keyword-miss → điểm thấp; đó là kết quả thật, không phải lỗi cấu hình.

### 9.4 Gộp & phân tích kết quả nhiều mẻ — `analyze_results.py`
Chạy nhiều tập / nhiều pipeline riêng rẽ rồi **cộng dồn** thành một báo cáo (per-tiêu chí + per-category + leaderboard). Tự khử trùng theo câu hỏi:
```powershell
# Gộp 2 tập × 3 pipeline → báo cáo N=30 ba chiều
python scripts/analyze_results.py `
  --cache experiments/answers_mh_cache.json experiments/answers_set2_cache.json `
          experiments/answers_kg_cache.json experiments/answers_kg_set2_cache.json `
  --gold  data/golden_queries_multihop.json data/golden_queries_set2.json
```
- `--cache`: một hoặc nhiều file `*_cache.json` (kết quả judge).
- `--gold`: các file câu hỏi tương ứng (để lấy `category`).
- In: bảng tiêu chí, xếp hạng, bảng theo category, Pass%.

### 9.5 Quy trình đầy đủ đã dùng cho báo cáo (tham khảo)
```powershell
# 1) Vector + GraphRAG trên 2 tập (no-KG, an toàn)
python scripts/run_judges_no_kg.py --phase generate --questions data/golden_queries_multihop.json --answers-file experiments/answers_mh.json
python scripts/run_judges_no_kg.py --phase judge    --questions data/golden_queries_multihop.json --answers-file experiments/answers_mh.json
python scripts/run_judges_no_kg.py --phase generate --questions data/golden_queries_set2.json     --answers-file experiments/answers_set2.json
python scripts/run_judges_no_kg.py --phase judge    --questions data/golden_queries_set2.json     --answers-file experiments/answers_set2.json

# 2) KG chạy riêng trên 2 tập
python scripts/run_all_judges.py --phase generate --pipelines kg --questions data/golden_queries_multihop.json --answers-file experiments/answers_kg.json
python scripts/run_all_judges.py --phase judge    --pipelines kg --questions data/golden_queries_multihop.json --answers-file experiments/answers_kg.json
python scripts/run_all_judges.py --phase generate --pipelines kg --questions data/golden_queries_set2.json      --answers-file experiments/answers_kg_set2.json
python scripts/run_all_judges.py --phase judge    --pipelines kg --questions data/golden_queries_set2.json      --answers-file experiments/answers_kg_set2.json

# 3) Gộp tất cả → leaderboard N=30 ba chiều (Mục 9.4)
```
> **Mẹo quota:** judge `num_runs=1` để vừa free-tier; nâng lên `num_runs=3` ở lần chạy cuối khi có đủ hạn ngạch. Giám khảo tự xoay các key Groq trong `.env` và tự retry khi gặp 429 / JSON hỏng.
