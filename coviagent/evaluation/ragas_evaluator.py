import pandas as pd
from typing import List, Dict, Any

# Note: In ragas >= 0.1.7, the import paths and logic have slightly changed.
# The standard usage is to use `evaluate` and pass the specific metrics.
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

class RagasEvaluator:
    """Wrapper around Ragas for standard metric execution."""
    
    def __init__(self, metrics: List[Any] = None):
        # Default metrics for RAG evaluation
        self.metrics = metrics or [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        
    def evaluate(self, data_dict: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Evaluate a dataset dictionary formatted for Ragas.
        
        Args:
            data_dict: Must contain 'question', 'answer', 'contexts', 'ground_truth'
            
        Returns:
            Dict containing scores and the detailed pandas dataframe.
        """
        # Convert dictionary to HuggingFace Dataset, required by ragas
        dataset = Dataset.from_dict(data_dict)
        
        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
        )
        
        # Convert to dict format
        result_dict = {
            "metrics": {k: v for k, v in result.items()},
            "details": result.to_pandas()
        }
        
        return result_dict
