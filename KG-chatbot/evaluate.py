import os
import json
import csv
from dotenv import load_dotenv
import time
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

# 1. Load môi trường và kết nối Neo4j
load_dotenv()
graph = Neo4jGraph(
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
    database=os.environ.get("NEO4J_DATABASE")
)
graph.refresh_schema()

# Khởi tạo mô hình cho Chatbot (Groq - nhanh và miễn phí)
bot_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0
)

cypher_prompt_template = """Bạn là chuyên gia Neo4j Cypher. Chỉ trả về mã Cypher pure, không giải thích. Không dùng markdown.

Schema:
{schema}

HƯỚNG DẪN TƯ DUY CYPHER:
1. NGÔN NGỮ: Dữ liệu CHỨA CẢ TIẾNG VIỆT và TIẾNG ANH. 
   - Với câu hỏi Tiếng Việt: Dùng TỪ KHÓA TIẾNG VIỆT + từ khóa Tiếng Anh liên quan.
   - Dùng từ khóa Tiếng Việt gốc (không dịch).
2. TỐI ƯU TRUY VẤN: 
   - KHÔNG dùng `MATCH (n)-[r]-(m)` vì sẽ gây trùng lặp dữ liệu khổng lồ (làm tràn bộ nhớ).
   - BẮT BUỘC dùng `MATCH (c:Chunk)`.
3. CHIẾN LƯỢC TỪ KHÓA: 
   - Với câu hỏi so sánh hoặc liệt kê: Lấy từ khóa chính của cả 2 thực thể.
   - Với câu hỏi logic (thời gian, điều kiện): Lấy các từ khóa quan trọng (time, days, ngày, khoảng, symptoms, triệu chứng, risk).
4. CÚ PHÁP CHUẨN: 
   `MATCH (c:Chunk) WHERE (toLower(c.text) CONTAINS 'keyword1' OR toLower(c.text) CONTAINS 'keyword2') RETURN DISTINCT c.text LIMIT 20`
5. Đừng bắt buộc tìm kiếm các từ khóa phân loại như 'side effect' hay 'adverse reaction' nếu chúng không xuất hiện trong câu văn mô tả.
6. Với câu hỏi về tác dụng phụ, hãy tìm tên thuốc kết hợp với các từ khóa mô tả cảm giác (taste, nausea, etc.) hoặc đơn giản là chỉ cần tìm tên thuốc.
7. Luôn ưu tiên dùng OR để quét được nhiều đoạn văn nhất có thể.

VÍ DỤ TƯ DUY:
Q: Thuốc Paxlovid và Lagevrio khác nhau thế nào về đối tượng dùng?
A: MATCH (c:Chunk) WHERE (toLower(c.text) CONTAINS 'paxlovid' OR toLower(c.text) CONTAINS 'lagevrio' OR toLower(c.text) CONTAINS 'molnupiravir') AND (toLower(c.text) CONTAINS 'adult' OR toLower(c.text) CONTAINS 'children' OR toLower(c.text) CONTAINS 'age') RETURN DISTINCT c.text LIMIT 20

Q: Trẻ em 13 tuổi dùng thuốc gì?
A: MATCH (c:Chunk) WHERE (toLower(c.text) CONTAINS 'paxlovid' OR toLower(c.text) CONTAINS 'remdesivir' OR toLower(c.text) CONTAINS 'veklury') AND (toLower(c.text) CONTAINS 'children' OR toLower(c.text) CONTAINS '12') RETURN DISTINCT c.text LIMIT 20

Q: Cân nặng và thời gian tiêm Pemgarda?
A: MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS 'pemivibart' OR toLower(c.text) CONTAINS 'pemgarda' RETURN DISTINCT c.text LIMIT 20

Q: Paxlovid có tác dụng phụ gì?
A: MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS 'paxlovid' AND (toLower(c.text) CONTAINS 'taste' OR toLower(c.text) CONTAINS 'mouth' OR toLower(c.text) CONTAINS 'side' OR toLower(c.text) CONTAINS 'effect') RETURN DISTINCT c.text LIMIT 20

Q: Đối tượng nguy cơ cao?
A: MATCH (c:Chunk) WHERE toLower(c.text) CONTAINS 'risk' AND (toLower(c.text) CONTAINS 'severe' OR toLower(c.text) CONTAINS 'illness' OR toLower(c.text) CONTAINS 'serious') RETURN DISTINCT c.text LIMIT 20

Q: Mất trung bình bao nhiêu ngày kể từ khi một người bị nhiễm virus để các triệu chứng bắt đầu xuất hiện?
A: MATCH (c:Chunk) WHERE (toLower(c.text) CONTAINS 'khoảng' OR toLower(c.text) CONTAINS 'ngày' OR toLower(c.text) CONTAINS 'incubation') AND (toLower(c.text) CONTAINS 'triệu chứng' OR toLower(c.text) CONTAINS 'symptoms' OR toLower(c.text) CONTAINS 'xuất hiện' OR toLower(c.text) CONTAINS 'appear') RETURN DISTINCT c.text LIMIT 20

Câu hỏi: {question}
Cypher query:"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], 
    template=cypher_prompt_template
)

# ================= ĐOẠN PROMPT ÉP TRẢ LỜI BẰNG TIẾNG VIỆT & DÙNG CONTEXT =================
qa_prompt_template = """Bạn là trợ lý y khoa chuyên nghiệp. TUYỆT ĐỐI KHÔNG được tự bịa hoặc sử dụng kiến thức có sẵn.

**HƯỚNG DẪN BẮT BUỘC TUYỆT ĐỐI:**
1. CHỈ DÙNG thông tin trong CƠ SỞ DỮ LIỆU bên dưới.
2. TUYỆT ĐỐI KHÔNG sử dụng kiến thức riêng của bạn để bịa câu trả lời.
3. Nếu CƠ SỞ DỮ LIỆU rỗng hoặc không chứa câu trả lời → BẮT BUỘC trả: "Tôi không biết"
4. Trả lời bằng tiếng Việt, chính xác, đầy đủ dựa HOÀN TOÀN trên context.

CƠ SỞ DỮ LIỆU (DÙNG THÔNG TIN NÀY ĐỂ TRẢ LỜI):
{context}

Câu hỏi: {question}
Trả lời:"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], 
    template=qa_prompt_template
)

# Khởi tạo chuỗi QA của Chatbot
chain = GraphCypherQAChain.from_llm(
    cypher_llm=bot_llm, 
    qa_llm=bot_llm,     
    graph=graph,
    verbose=False, 
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt,          # ĐÃ BỔ SUNG THAM SỐ NÀY
    return_intermediate_steps=False,
    allow_dangerous_requests=True
)

eval_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0
)
eval_prompt = PromptTemplate.from_template(
    """Bạn là một giám khảo y khoa khắt khe. Nhiệm vụ của bạn là so sánh câu trả lời của Chatbot với Đáp án gốc (Ground Truth).
    
    Câu hỏi: {question}
    Đáp án gốc: {ground_truth}
    Chatbot trả lời: {bot_answer}
    
    Hãy đánh giá xem câu trả lời của Chatbot có đúng và đủ ý so với Đáp án gốc hay không. 
    Chỉ trả về 1 trong 2 từ: ĐẠT hoặc KHÔNG ĐẠT. Dòng tiếp theo giải thích ngắn gọn lý do (1 câu)."""
)
eval_chain = eval_prompt | eval_llm

# 3. Đọc file JSON
print("Đang đọc file questions.json...")
with open('questions.json', 'r', encoding='utf-8') as f:
    test_cases = json.load(f)

results = []

# 4. Chạy Test Tự động
print(f"Bắt đầu test {len(test_cases)} câu hỏi...")

for idx, test in enumerate(test_cases):
    q_id = test['question_id']
    question = test['question']
    ground_truth = test['ground_truth']
    
    print(f"\n{'='*60}")
    print(f"[{q_id}] CÂU HỎI: {question}")
    
    try:
        # Lấy câu trả lời từ Bot
        bot_response = chain.invoke({"query": question})
        bot_answer = bot_response['result']
        
        # Nhờ Gemini giám khảo chấm điểm
        evaluation = eval_chain.invoke({
            "question": question,
            "ground_truth": ground_truth,
            "bot_answer": bot_answer
        }).content
        
    except Exception as e:
        bot_answer = f"Lỗi thực thi: {str(e)}"
        evaluation = "KHÔNG ĐẠT\nLỗi hệ thống khi sinh Cypher."
        
    # --- PHẦN IN RA MÀN HÌNH ---
    print(f"🤖 BOT TRẢ LỜI : {bot_answer}")
    print(f"🎯 ĐÁP ÁN GỐC  : {ground_truth}")
    
    # Tách dòng chữ ĐẠT/KHÔNG ĐẠT và lý do để in cho rõ ràng
    eval_lines = evaluation.split('\n')
    kq = eval_lines[0].strip()
    ly_do = eval_lines[1].strip() if len(eval_lines) > 1 else ""
    
    if "ĐẠT" in kq and "KHÔNG" not in kq:
        print(f"✅ ĐÁNH GIÁ    : {kq} - {ly_do}")
    else:
        print(f"❌ ĐÁNH GIÁ    : {kq} - {ly_do}")
    # ---------------------------------------------
    
    # Lưu vào mảng kết quả
    results.append({
        "ID": q_id,
        "Category": test['category'],
        "Question": question,
        "Ground_Truth": ground_truth,
        "Bot_Answer": bot_answer,
        "Evaluation": evaluation
    })

    print("⏳ Nghỉ 10 giây")
    time.sleep(10)

# 5. Xuất kết quả ra file CSV
csv_file = "test_results.csv"
with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=["ID", "Category", "Question", "Ground_Truth", "Bot_Answer", "Evaluation"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Đã test xong! Kết quả chi tiết được lưu tại file: {csv_file}")