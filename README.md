# CoviAgent Benchmark Pipeline

Hệ thống đánh giá Benchmark chuyên nghiệp dành cho Hội thoại Y tế (Medical Conversational AI), được thiết kế để đo lường hiệu năng của các hệ thống Retrieval-Augmented Generation (RAG). 

Phase 1 hỗ trợ **Vector RAG** và được cấu trúc theo dạng Module (Clean Architecture) để dễ dàng tích hợp **GraphRAG** trong Phase 2.

---

## 📂 Kiến Trúc Dự Án

- `coviagent/`: Core source code chứa các abstract classes và logic.
- `configs/`: File cấu hình YAML (`default_vector_rag.yaml`) quyết định Provider và model.
- `scripts/`: Chứa các file thực thi chính để nạp dữ liệu và chấm điểm.
- `data/`: 
  - `raw/`: Chứa file `.txt` hoặc tài liệu thô.
  - `eval/`: Chứa tập dữ liệu câu hỏi - đáp án chuẩn (JSONL).
- `experiments/`: Thư mục tự động sinh ra chứa kết quả chấm điểm Ragas sau mỗi lần chạy.

---

## 🚀 Hướng Dẫn Cài Đặt

### Bước 1: Cài đặt thư viện Python
Tạo môi trường ảo (Virtual Environment) và cài đặt các thư viện cần thiết:
```bash
python -m venv venv
source venv/bin/activate hoặc .\venv\Scripts\Activate.ps1 # Trên Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Bước 2: Thiết lập Vector Database (Qdrant)
Hệ thống sử dụng Qdrant thông qua Docker để lưu trữ Vector và phục vụ truy xuất với hiệu năng cao.
```bash
# Khởi chạy Qdrant Server ngầm
docker-compose up -d
```
*Giao diện quản lý Qdrant sẽ có tại: http://localhost:6333/dashboard*

### Bước 3: Cấu hình API Keys
Tham khảo file `.env.example` để config `.env` ở thư mục gốc của dự án và điền các API Key của bạn vào:
```env
OPENAI_API_KEY="sk-..."
GROQ_API_KEY="gsk_..."
OPENROUTER_API_KEY="sk-or-v1-..."
```
*(Lưu ý: Hệ thống sử dụng OpenAI để đánh giá độ chính xác (Ragas Evaluation), vì vậy hãy luôn đảm bảo có OPENAI_API_KEY nếu bạn chạy lệnh Benchmark).*

---

## ⚙️ Tùy Chỉnh LLM & Embedding

Bạn có thể dễ dàng chuyển đổi qua lại giữa OpenAI, Groq, OpenRouter hoặc HuggingFace bằng cách chỉnh sửa file `configs/default_vector_rag.yaml`.

```yaml
llm:
  provider: "openrouter" # hoặc "openai", "groq"
  model_name: "google/gemini-flash-1.5"
  temperature: 0.0

embedding:
  provider: "huggingface" # hoặc "openai"
  model_name: "BAAI/bge-small-en-v1.5"
  dimension: 384
```

---

## 🏃 Cách Chạy Hệ Thống

### 1. Ingestion & Indexing (Nạp dữ liệu)
Đọc các file trong thư mục `data/raw/`, chunking và đẩy vào Qdrant Server:
```bash
python scripts/build_index.py --config configs/default_vector_rag.yaml
```

### 2. Benchmarking (Chấm điểm & Đánh giá)
Chạy bộ pipeline RAG trên tập dữ liệu đánh giá và sử dụng Ragas để chấm điểm:
```bash
python scripts/run_benchmark.py \
    --config configs/default_vector_rag.yaml \
    --dataset data/eval/sample_medical_qa.jsonl \
    --exp_name coviagent_baseline
```

Sau khi chạy xong, kết quả và log chi tiết sẽ được xuất ra trong thư mục `experiments/`.
