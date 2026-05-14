from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RetrievalContext(BaseModel):
    """Represents a single retrieved chunk/node from the knowledge base."""
    node_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QAItem(BaseModel):
    """Represents a single Q&A pair in the evaluation dataset."""
    question: str
    ground_truth: str
    context_references: Optional[List[str]] = Field(default_factory=list, description="IDs of ground truth contexts if known")

class BenchmarkResult(BaseModel):
    """Represents the results of a benchmark run."""
    experiment_name: str
    metrics: Dict[str, float]
    details: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed per-question metrics and predictions")
