from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import RetrievalContext, QAItem, BenchmarkResult

class BaseRetriever(ABC):
    """Abstract base class for all retrievers (Vector, Graph, Hybrid)."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalContext]:
        """
        Retrieve relevant contexts for a given query.
        
        Args:
            query (str): The question or query string.
            top_k (Optional[int]): Number of contexts to retrieve.
            
        Returns:
            List[RetrievalContext]: Retrieved contexts with metadata and scores.
        """
        pass

class BasePipeline(ABC):
    """Abstract base class for RAG pipelines."""
    
    @abstractmethod
    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute the RAG pipeline for a given question.
        
        Args:
            question (str): The user's question.
            
        Returns:
            Dict[str, Any]: Dictionary containing 'answer' (str) and 'contexts' (List[str]).
        """
        pass

class BaseEvaluator(ABC):
    """Abstract base class for benchmark evaluators."""
    
    @abstractmethod
    def evaluate(self, dataset: List[QAItem], pipeline: BasePipeline) -> BenchmarkResult:
        """
        Evaluate a given RAG pipeline on a dataset.
        
        Args:
            dataset (List[QAItem]): The evaluation dataset.
            pipeline (BasePipeline): The pipeline to evaluate.
            
        Returns:
            BenchmarkResult: The evaluation results.
        """
        pass
