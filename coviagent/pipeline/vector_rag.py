from typing import Dict, Any
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate

from coviagent.core.interfaces import BasePipeline, BaseRetriever

DEFAULT_QA_PROMPT = (
    "Bạn là một trợ lý y tế chuyên nghiệp về COVID-19.\n\n"
    "NGUYÊN TẮC BẮT BUỘC:\n"
    "- CHỈ sử dụng thông tin có trong ngữ cảnh bên dưới để trả lời.\n"
    "- TUYỆT ĐỐI KHÔNG được bịa, suy luận, hoặc thêm bất kỳ thông tin y tế nào không có trong ngữ cảnh.\n"
    "- ĐẶC BIỆT: Nếu câu hỏi hỏi về việc NÊN DÙNG thuốc/phương pháp gì cho một đối tượng/bệnh lý, nhưng trong ngữ cảnh chỉ có thông tin CHỐNG CHỈ ĐỊNH (KHÔNG ĐƯỢC DÙNG) hoặc CẢNH BÁO cho đối tượng đó, bạn PHẢI nêu rõ các cảnh báo/chống chỉ định này để đảm bảo an toàn cho người dùng.\n"
    "- Nếu ngữ cảnh hoàn toàn không chứa bất kỳ thông tin nào liên quan, hãy nói rõ: 'Tài liệu hiện tại không có thông tin về vấn đề này.'\n"
    "- Trả lời bằng tiếng Việt, trình bày rõ ràng có cấu trúc.\n\n"
    "Ngữ cảnh:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Câu hỏi: {query_str}\n"
    "Trả lời (chỉ dựa trên ngữ cảnh ở trên): "
)

class VectorRAGPipeline(BasePipeline):
    """
    Standard Vector RAG Pipeline combining a BaseRetriever with an LLM generation step.
    """
    def __init__(self, retriever: BaseRetriever, llm: LLM, prompt_template: str = DEFAULT_QA_PROMPT):
        self.retriever = retriever
        self.llm = llm
        self.prompt = PromptTemplate(prompt_template)
        
    def query(self, question: str) -> Dict[str, Any]:
        """
        1. Retrieve contexts using the retriever.
        2. Format contexts into the prompt.
        3. Generate answer using the LLM.
        4. Return response dictionary containing answer and retrieved contexts.
        """
        # Step 1: Retrieval
        retrieval_contexts = self.retriever.retrieve(question)
        
        # Extract text for the prompt
        context_texts = [ctx.text for ctx in retrieval_contexts]
        context_str = "\n\n".join(context_texts)
        
        # Step 2: Prompt Formatting
        formatted_prompt = self.prompt.format(
            context_str=context_str,
            query_str=question
        )
        
        # Step 3: Generation
        response = self.llm.complete(formatted_prompt)
        answer = str(response)
        
        # Step 4: Return formatted results for evaluation
        return {
            "answer": answer,
            "contexts": context_texts,  # We return strings as required by many evaluation frameworks like Ragas
            "raw_contexts": retrieval_contexts # Raw context objects for detailed tracking
        }
