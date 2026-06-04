import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

# 1. Load môi trường
load_dotenv()

# 2. Khởi tạo đồ thị Neo4j
graph = Neo4jGraph(
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
    database=os.environ.get("NEO4J_DATABASE")
)

# Refresh schema để LangChain hiểu cấu trúc đồ thị (Nodes, Relationships)
graph.refresh_schema()

# 3. Khởi tạo mô hình Groq (nhanh, miễn phí)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0, # Để 0 để mô hình bám sát dữ liệu từ Graph, tránh ảo giác
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

Q: Hãy liệt kê tất cả các nhóm đối tượng và các bệnh nền có nguy cơ cao chuyển biến thành bệnh nghiêm trọng hoặc tử vong khi mắc COVID-19?
A: MATCH (c:Chunk) WHERE (toLower(c.text) CONTAINS 'người lớn tuổi' OR toLower(c.text) CONTAINS 'lớn tuổi' OR toLower(c.text) CONTAINS 'bệnh lý nền' OR toLower(c.text) CONTAINS 'bệnh tim' OR toLower(c.text) CONTAINS 'tiểu đường' OR toLower(c.text) CONTAINS 'phổi' OR toLower(c.text) CONTAINS 'hệ miễn dịch' OR toLower(c.text) CONTAINS 'vaccine' OR toLower(c.text) CONTAINS 'nguy cơ' OR toLower(c.text) CONTAINS 'risk' OR toLower(c.text) CONTAINS 'severe' OR toLower(c.text) CONTAINS 'bệnh nặng') AND (toLower(c.text) CONTAINS 'covid' OR toLower(c.text) CONTAINS 'coronavirus') RETURN DISTINCT c.text LIMIT 30

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
# =========================================================================================

# 5. Khởi tạo Chuỗi QA tương tác với Graph
chain = GraphCypherQAChain.from_llm(
    cypher_llm=llm, # LLM dùng để viết Cypher
    qa_llm=llm,     # LLM dùng để trả lời người dùng
    graph=graph,
    verbose=True,   # Bật để xem Cypher queries và debug
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt,
    return_intermediate_steps=True, # Trả về Cypher query để debug
    allow_dangerous_requests=True
)

# 6. Vòng lặp Chatbot
print("--- Chatbot Y khoa (Gõ 'exit' hoặc 'quit' để thoát) ---")
while True:
    user_input = input("Bạn: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    try:
        # Chạy chain để lấy câu trả lời
        response = chain.invoke({"query": user_input})
        print(f"\nChatbot: {response['result']}\n")
    except Exception as e:
        print(f"\n[Lỗi]: {e}\n")