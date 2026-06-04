# 🔥 Bộ Câu Hỏi Thử Thách Sức Mạnh GraphRAG (Multi-hop Reasoning)

Dưới đây là danh sách các câu hỏi được thiết kế đặc biệt dựa trên schema của Neo4j để kiểm tra khả năng "kết nối các điểm mù" (Connecting the dots) của hệ thống GraphRAG. Vector RAG thông thường sẽ rất dễ trả lời sai hoặc thiếu ý đối với những câu hỏi này.

---

### 1. Câu hỏi Suy luận Chéo (Multi-hop Reasoning)
**Mục tiêu test:** Kiểm tra khả năng kết nối `[Xét nghiệm] -> [Bệnh] -> [Triệu chứng]`
> *"Những loại xét nghiệm nào có thể dùng để phát hiện loại virus gây ra tình trạng đau đầu (Headache) và mắt đỏ kích ứng (Red Or Irritated Eyes)?"*

- **Tại sao Vector RAG thất bại:** Các tài liệu nói về "Triệu chứng đau đầu/mắt đỏ" thường nằm ở một đoạn văn bản khác hoàn toàn so với đoạn văn bản nói về "Phương pháp xét nghiệm PCR".
- **Tại sao GraphRAG chiến thắng:** Nó sẽ tìm thấy `[Red Or Irritated Eyes]`, tự động men theo mũi tên `<-[CAUSES]- [Covid-19]`, và từ đó dò ngược tiếp lên `<-[DETECTS]- [Pcr Tests]`. 

---

### 2. Câu hỏi "Phá bẫy" Mối Quan Hệ (Relationship Traversal)
**Mục tiêu test:** Tránh Hallucination bằng các liên kết Entity.
> *"Sars-Cov-2 liên quan như thế nào đến PCR Tests, và nó có thể gây ra những triệu chứng điển hình nào ngoài ho và sốt?"*

- **Sức mạnh của GraphRAG:** Cụm từ "Sars-Cov-2" và "PCR Tests" có thể không đứng cạnh nhau trong câu. Nhưng GraphRAG sẽ truy xuất được cụm liên kết: `[Pcr Tests] -(DETECTS)-> [Sars-Cov-2]` và lôi ra toàn bộ các nhánh `-(CAUSES)-> [Sore Throat]`, `[Tiredness]` một cách chính xác tuyệt đối.

---

### 3. Câu hỏi Phân tích Bao quát (Global Sensemaking)
**Mục tiêu test:** Đòi hỏi LLM hiểu trật tự nhân quả của thuật ngữ chuyên ngành.
> *"Hãy liệt kê một chuỗi logic: Loại bệnh nào gây ra tình trạng Covid-19 Rebound, bệnh đó lây lan ra sao và xét nghiệm nào phát hiện được nó?"*

- **Sức mạnh của GraphRAG:** GraphRAG sẽ kích hoạt truy vấn để bắt trọn cụm `[Covid-19] -(CAUSES)-> [Covid-19 Rebound]`. Nhờ có mũi tên quan hệ rõ ràng, LLM sẽ hiểu được "Rebound" là một *hệ quả* của bệnh, chứ không phải là tên một loại thuốc hay một xét nghiệm, từ đó sắp xếp câu trả lời theo đúng trật tự.

---

### 💡 Hướng dẫn sử dụng:
1. Mở giao diện Streamlit (`python -m streamlit run app.py`)
2. Chọn phương pháp **GraphRAG (Kết hợp đỉnh cao)** ở thanh bên trái.
3. Copy và dán từng câu hỏi trên vào khung chat.
4. Mở rộng phần `Cấu trúc Tri thức Truy xuất` để xem các dòng `Mở rộng:` màu xanh lá thể hiện luồng tư duy đồ thị!
