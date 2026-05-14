from typing import Dict, Any
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate

from coviagent.core.interfaces import BasePipeline, BaseRetriever

DEFAULT_QA_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in the context of the medical domain and COVID-19.\n"
    "Query: {query_str}\n"
    "Answer: "
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
