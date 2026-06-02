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


# ──────────────────────────────────────────────
#  LLM-as-a-Judge Models
# ──────────────────────────────────────────────

class JudgeRubricScore(BaseModel):
    """Kết quả chấm điểm theo rubric từ 1 lần judge."""
    accuracy: int = Field(ge=0, le=5, description="Độ chính xác so với ground truth (0-5)")
    completeness: int = Field(ge=0, le=5, description="Độ đầy đủ các ý chính (0-5)")
    evidence_support: int = Field(ge=0, le=5, description="Bằng chứng hỗ trợ từ context (0-5)")
    hallucination: int = Field(ge=0, le=5, description="Không bịa đặt, 5=không bịa (0-5)")
    final_decision: str = Field(description="Pass hoặc Fail")
    explanation: str = Field(description="Giải thích ngắn gọn")

    @property
    def average_score(self) -> float:
        return (self.accuracy + self.completeness + self.evidence_support + self.hallucination) / 4.0


class PairwiseResult(BaseModel):
    """Kết quả pairwise comparison giữa 2 output (blind)."""
    winner: str = Field(description="A, B, hoặc Tie")
    explanation: str = Field(description="Giải thích lý do chọn")


class GoldCase(BaseModel):
    """Một gold case có đáp án chuẩn do người tạo ra để validate judge."""
    id: str
    question: str
    ground_truth: str
    category: str = "general"
    expected_verdict: str = Field(default="Pass", description="Pass hoặc Fail")
    expected_min_accuracy: int = Field(default=4, ge=0, le=5)


class JudgeBenchmarkResult(BaseModel):
    """Kết quả benchmark tổng hợp từ LLM Judge."""
    experiment_name: str
    judge_model: str
    num_judge_runs: int
    rubric_results: List[Dict[str, Any]] = Field(default_factory=list)
    pairwise_results: Optional[List[Dict[str, Any]]] = None
    gold_case_accuracy: Optional[float] = None
    summary_metrics: Dict[str, float] = Field(default_factory=dict)
