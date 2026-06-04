from typing import Dict, Any, Optional
import os
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

from coviagent.core.interfaces import BasePipeline

CYPHER_PROMPT_TEMPLATE = """Bạn là chuyên gia Neo4j Cypher. Chỉ trả về mã Cypher pure, không giải thích. Không dùng markdown.

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

Câu hỏi: {question}
Cypher query:"""

QA_PROMPT_TEMPLATE = """Bạn là trợ lý y khoa chuyên nghiệp. TUYỆT ĐỐI KHÔNG được tự bịa hoặc sử dụng kiến thức có sẵn.

**HƯỚNG DẪN BẮT BUỘC TUYỆT ĐỐI:**
1. CHỈ DÙNG thông tin trong CƠ SỞ DỮ LIỆU bên dưới.
2. TUYỆT ĐỐI KHÔNG sử dụng kiến thức riêng của bạn để bịa câu trả lời.
3. Nếu CƠ SỞ DỮ LIỆU rỗng hoặc không chứa câu trả lời → BẮT BUỘC trả: "Tôi không biết"
4. Trả lời bằng tiếng Việt, chính xác, đầy đủ dựa HOÀN TOÀN trên context.

CƠ SỞ DỮ LIỆU (DÙNG THÔNG TIN NÀY ĐỂ TRẢ LỜI):
{context}

Câu hỏi: {question}
Trả lời:"""

class KGRAGPipeline(BasePipeline):
    """
    Adapter để chạy Knowledge Graph (KG) Chatbot dưới dạng BasePipeline
    cho phép hệ thống LLM Judge chấm điểm.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        self.graph = Neo4jGraph(
            url=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USERNAME", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
            database=os.environ.get("NEO4J_DATABASE", "neo4j")
        )
        self.graph.refresh_schema()

        self.bot_llm = ChatGroq(
            model=model_name,
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            temperature=0
        )

        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"], 
            template=CYPHER_PROMPT_TEMPLATE
        )
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"], 
            template=QA_PROMPT_TEMPLATE
        )

        self.chain = GraphCypherQAChain.from_llm(
            cypher_llm=self.bot_llm, 
            qa_llm=self.bot_llm,     
            graph=self.graph,
            verbose=False, 
            cypher_prompt=cypher_prompt,
            qa_prompt=qa_prompt,
            return_intermediate_steps=True, # Bật để lấy context
            allow_dangerous_requests=True
        )
        
    def query(self, question: str) -> Dict[str, Any]:
        response = self.chain.invoke({"query": question})
        
        answer = response.get("result", "")
        contexts = []
        
        # Trích xuất context từ intermediate steps của Neo4j (nếu có)
        steps = response.get("intermediate_steps", [])
        if steps and len(steps) >= 2:
            try:
                # context context thường nằm ở bước 2
                raw_contexts = steps[1].get("context", [])
                if isinstance(raw_contexts, list):
                    contexts = [str(c) for c in raw_contexts]
                elif isinstance(raw_contexts, str):
                    contexts = [raw_contexts]
            except Exception:
                pass
                
        return {
            "answer": answer,
            "contexts": contexts
        }
