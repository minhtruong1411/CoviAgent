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
