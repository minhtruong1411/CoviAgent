"""
LLM-as-a-Judge Engine for CoviAgent Benchmark.

Implements:
- Structured rubric scoring (Accuracy, Completeness, Evidence Support, Hallucination)
- Majority vote across multiple judge runs
- Blind pairwise comparison (model names hidden, position randomized)
- Gold case validation to verify judge reliability

Design principles:
- Judge prompts never reveal model names → no identity bias
- Pairwise runs both AB and BA orderings → no position bias
- JSON-structured output → deterministic parsing
"""

import json
import time
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from llama_index.core.llms import LLM

from coviagent.core.config import JudgeConfig
from coviagent.core.models import JudgeRubricScore, PairwiseResult, GoldCase


# ──────────────────────────────────────────────
#  Prompt Templates (Bias-free)
# ──────────────────────────────────────────────

RUBRIC_PROMPT = """Bạn là giám khảo y khoa chuyên nghiệp và khắt khe. Đánh giá câu trả lời dưới đây theo 4 tiêu chí.

## Câu hỏi
{question}

## Đáp án tham khảo (Ground Truth)
{ground_truth}

## Câu trả lời cần đánh giá
{answer}

## Ngữ cảnh đã truy xuất (Retrieved Contexts)
{contexts}

## Rubric chấm điểm (0-5 cho mỗi tiêu chí):

1. **Accuracy** (Độ chính xác): Thông tin trong câu trả lời có khớp với đáp án tham khảo không?
   - 0: Hoàn toàn sai hoặc không liên quan
   - 1: Có đề cập nhưng phần lớn sai
   - 2: Đúng một phần nhỏ
   - 3: Đúng khoảng nửa ý chính
   - 4: Đúng phần lớn, sai sót nhỏ
   - 5: Chính xác tuyệt đối, khớp hoàn toàn

2. **Completeness** (Độ đầy đủ): Câu trả lời có bao phủ đủ các ý chính trong đáp án tham khảo không?
   - 0: Thiếu hoàn toàn mọi ý chính
   - 1: Chỉ có 1 ý nhỏ
   - 2: Có vài ý nhưng thiếu nhiều
   - 3: Bao phủ khoảng nửa số ý
   - 4: Gần đầy đủ, thiếu 1-2 ý phụ
   - 5: Đầy đủ mọi ý chính và phụ

3. **Evidence Support** (Bằng chứng hỗ trợ): Câu trả lời có được hỗ trợ bởi ngữ cảnh truy xuất không?
   - 0: Không có bất kỳ bằng chứng nào từ ngữ cảnh
   - 1: Rất ít liên quan đến ngữ cảnh
   - 2: Một phần nhỏ có bằng chứng
   - 3: Khoảng nửa tuyên bố có bằng chứng
   - 4: Phần lớn có bằng chứng, 1-2 thiếu
   - 5: Mọi tuyên bố đều có bằng chứng rõ ràng từ ngữ cảnh

4. **Hallucination** (Không bịa đặt): Câu trả lời có chứa thông tin bịa đặt, không có trong ngữ cảnh hoặc đáp án không?
   - 0: Toàn bộ nội dung là bịa đặt
   - 1: Phần lớn bịa đặt
   - 2: Nhiều chi tiết bịa đặt
   - 3: Có vài chi tiết bịa đặt
   - 4: Hầu như không bịa, chỉ 1 chi tiết nhỏ
   - 5: Hoàn toàn không bịa đặt, mọi thông tin đều có nguồn

## YÊU CẦU BẮT BUỘC:
- Trả về ĐÚNG format JSON (không có markdown, không có ```json)
- final_decision là "Pass" nếu trung bình 4 tiêu chí >= {pass_threshold}, ngược lại "Fail"
- explanation bằng tiếng Việt, 1-2 câu ngắn gọn

{{"accuracy": <0-5>, "completeness": <0-5>, "evidence_support": <0-5>, "hallucination": <0-5>, "final_decision": "<Pass/Fail>", "explanation": "<giải thích>"}}"""


PAIRWISE_PROMPT = """Bạn là giám khảo y khoa chuyên nghiệp. So sánh 2 câu trả lời cho cùng 1 câu hỏi y tế.

QUAN TRỌNG: Đánh giá KHÁCH QUAN dựa trên nội dung. Không thiên vị theo thứ tự xuất hiện.

## Câu hỏi
{question}

## Đáp án tham khảo (Ground Truth)
{ground_truth}

## Output A
{output_a}

## Output B
{output_b}

## Tiêu chí đánh giá:
1. Độ chính xác so với đáp án tham khảo
2. Độ đầy đủ các ý chính
3. Bằng chứng hỗ trợ rõ ràng
4. Không bịa đặt thông tin

## YÊU CẦU BẮT BUỘC:
- Trả về ĐÚNG format JSON (không có markdown, không có ```json)
- winner là "A", "B", hoặc "Tie"
- explanation bằng tiếng Việt, 1-2 câu

{{"winner": "<A/B/Tie>", "explanation": "<giải thích>"}}"""


# ──────────────────────────────────────────────
#  JSON Parsing Utilities
# ──────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Extract JSON from LLM response, handling common formatting issues.
    Tries multiple strategies: direct parse, regex extraction, cleanup.
    """
    text = text.strip()
    
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find first complete JSON object
    brace_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Try to find nested JSON
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not parse JSON from LLM response:\n{text[:500]}")


def _clamp_score(value: Any, min_val: int = 0, max_val: int = 5) -> int:
    """Clamp a score value to valid range."""
    try:
        v = int(value)
        return max(min_val, min(max_val, v))
    except (ValueError, TypeError):
        return 0


# ──────────────────────────────────────────────
#  LLM Judge Class
# ──────────────────────────────────────────────

class LLMJudge:
    """
    LLM-as-a-Judge engine for evaluating RAG chatbot outputs.
    
    Features:
    - Structured rubric scoring with 4 criteria (0-5 scale)
    - Majority vote across N judge runs for stability
    - Blind pairwise comparison with position-bias mitigation
    - Gold case validation to verify judge reliability
    """
    
    def __init__(self, judge_llm: LLM, config: JudgeConfig):
        self.llm = judge_llm
        self.config = config
    
    def score_rubric(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: Optional[List[str]] = None,
    ) -> JudgeRubricScore:
        """
        Score a single answer using the structured rubric.
        
        Args:
            question: The original question
            answer: The chatbot's answer to evaluate
            ground_truth: The reference answer
            contexts: Retrieved contexts (for evidence support evaluation)
            
        Returns:
            JudgeRubricScore with scores for each criterion
        """
        contexts_str = "\n\n".join(contexts) if contexts else "(Không có ngữ cảnh)"
        
        prompt = RUBRIC_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            answer=answer,
            contexts=contexts_str,
            pass_threshold=self.config.pass_threshold,
        )
        
        response = self.llm.complete(prompt)
        response_text = str(response).strip()
        
        try:
            data = _extract_json(response_text)
            return JudgeRubricScore(
                accuracy=_clamp_score(data.get("accuracy", 0)),
                completeness=_clamp_score(data.get("completeness", 0)),
                evidence_support=_clamp_score(data.get("evidence_support", 0)),
                hallucination=_clamp_score(data.get("hallucination", 0)),
                final_decision=str(data.get("final_decision", "Fail")),
                explanation=str(data.get("explanation", "Không có giải thích")),
            )
        except (ValueError, KeyError) as e:
            # Fallback: return zero scores with error explanation
            return JudgeRubricScore(
                accuracy=0, completeness=0, evidence_support=0, hallucination=0,
                final_decision="Fail",
                explanation=f"Lỗi parse JSON từ judge: {str(e)[:200]}",
            )
    
    def score_rubric_majority(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        contexts: Optional[List[str]] = None,
        num_runs: Optional[int] = None,
    ) -> Tuple[JudgeRubricScore, List[JudgeRubricScore]]:
        """
        Score using rubric multiple times and aggregate via majority vote.
        
        The final score for each criterion is the median of all runs.
        The final_decision is determined by majority vote of Pass/Fail.
        
        Args:
            question, answer, ground_truth, contexts: Same as score_rubric
            num_runs: Override config.num_runs if specified
            
        Returns:
            Tuple of (aggregated_score, list_of_individual_scores)
        """
        runs = num_runs or self.config.num_runs
        all_scores: List[JudgeRubricScore] = []
        
        for i in range(runs):
            if i > 0:
                time.sleep(self.config.api_delay_seconds)
            
            score = self.score_rubric(question, answer, ground_truth, contexts)
            all_scores.append(score)
        
        # Aggregate: median for scores, majority for decision
        def median_score(scores: List[int]) -> int:
            sorted_s = sorted(scores)
            mid = len(sorted_s) // 2
            return sorted_s[mid]
        
        agg_accuracy = median_score([s.accuracy for s in all_scores])
        agg_completeness = median_score([s.completeness for s in all_scores])
        agg_evidence = median_score([s.evidence_support for s in all_scores])
        agg_hallucination = median_score([s.hallucination for s in all_scores])
        
        # Majority vote for Pass/Fail
        decisions = [s.final_decision for s in all_scores]
        decision_counts = Counter(decisions)
        agg_decision = decision_counts.most_common(1)[0][0]
        
        # Combine explanations
        explanations = [f"[Run {i+1}] {s.explanation}" for i, s in enumerate(all_scores)]
        combined_explanation = " | ".join(explanations)
        
        aggregated = JudgeRubricScore(
            accuracy=agg_accuracy,
            completeness=agg_completeness,
            evidence_support=agg_evidence,
            hallucination=agg_hallucination,
            final_decision=agg_decision,
            explanation=combined_explanation,
        )
        
        return aggregated, all_scores
    
    def pairwise_compare(
        self,
        question: str,
        output_a: str,
        output_b: str,
        ground_truth: str,
    ) -> PairwiseResult:
        """
        Blind pairwise comparison between two outputs.
        
        To mitigate position bias, runs comparison in both orders (AB and BA)
        and takes majority vote. Model names are NEVER shown to the judge.
        
        Args:
            question: The original question
            output_a: First model's answer (will be labeled randomly)
            output_b: Second model's answer (will be labeled randomly)
            ground_truth: Reference answer
            
        Returns:
            PairwiseResult with winner (A/B/Tie) mapped back to original order
        """
        results = []
        
        # Run 1: Original order (A=output_a, B=output_b)
        prompt_ab = PAIRWISE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            output_a=output_a,
            output_b=output_b,
        )
        response_ab = self.llm.complete(prompt_ab)
        try:
            data_ab = _extract_json(str(response_ab))
            winner_ab = str(data_ab.get("winner", "Tie")).strip().upper()
            explanation_ab = str(data_ab.get("explanation", ""))
            # Map back: A→A, B→B
            results.append({"winner": winner_ab, "explanation": explanation_ab})
        except ValueError:
            results.append({"winner": "TIE", "explanation": "Parse error (order AB)"})
        
        time.sleep(self.config.api_delay_seconds)
        
        # Run 2: Reversed order (A=output_b, B=output_a) to detect position bias
        prompt_ba = PAIRWISE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            output_a=output_b,  # Swapped!
            output_b=output_a,  # Swapped!
        )
        response_ba = self.llm.complete(prompt_ba)
        try:
            data_ba = _extract_json(str(response_ba))
            winner_ba_raw = str(data_ba.get("winner", "Tie")).strip().upper()
            explanation_ba = str(data_ba.get("explanation", ""))
            # Map back: A→B, B→A (reverse the labels)
            if winner_ba_raw == "A":
                winner_ba = "B"
            elif winner_ba_raw == "B":
                winner_ba = "A"
            else:
                winner_ba = "TIE"
            results.append({"winner": winner_ba, "explanation": explanation_ba})
        except ValueError:
            results.append({"winner": "TIE", "explanation": "Parse error (order BA)"})
        
        # Aggregate: if both agree → that's the winner. If disagree → Tie
        w1 = results[0]["winner"]
        w2 = results[1]["winner"]
        
        if w1 == w2:
            final_winner = w1
            final_explanation = f"Cả 2 lượt đều chọn {w1}. {results[0]['explanation']}"
        else:
            # Run a tiebreaker (3rd run, random order)
            time.sleep(self.config.api_delay_seconds)
            
            # Randomly pick order for tiebreaker
            if random.random() < 0.5:
                prompt_tie = PAIRWISE_PROMPT.format(
                    question=question, ground_truth=ground_truth,
                    output_a=output_a, output_b=output_b,
                )
                swap = False
            else:
                prompt_tie = PAIRWISE_PROMPT.format(
                    question=question, ground_truth=ground_truth,
                    output_a=output_b, output_b=output_a,
                )
                swap = True
            
            response_tie = self.llm.complete(prompt_tie)
            try:
                data_tie = _extract_json(str(response_tie))
                winner_tie = str(data_tie.get("winner", "Tie")).strip().upper()
                if swap:
                    if winner_tie == "A":
                        winner_tie = "B"
                    elif winner_tie == "B":
                        winner_tie = "A"
                    else:
                        winner_tie = "TIE"
                
                all_winners = [w1, w2, winner_tie]
                winner_counts = Counter(all_winners)
                final_winner = winner_counts.most_common(1)[0][0]
                final_explanation = (
                    f"3 lượt: {all_winners}. Majority vote → {final_winner}. "
                    f"{data_tie.get('explanation', '')}"
                )
            except ValueError:
                final_winner = "Tie"
                final_explanation = f"Không nhất quán giữa các lượt ({w1} vs {w2}), tiebreaker lỗi."
        
        # Normalize winner format
        winner_map = {"A": "A", "B": "B"}
        final_winner_clean = winner_map.get(final_winner, "Tie")
        
        return PairwiseResult(winner=final_winner_clean, explanation=final_explanation)
    
    def validate_gold_cases(
        self,
        gold_cases: List[GoldCase],
        get_answer_fn=None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Validate judge reliability using gold cases with known expected outcomes.
        
        For each gold case, the judge scores the ground_truth itself as the answer.
        A reliable judge should give high scores to correct ground-truth answers.
        
        Args:
            gold_cases: List of GoldCase with expected verdicts
            get_answer_fn: Optional function(question) → answer. 
                          If None, uses ground_truth as the answer (self-validation).
                          
        Returns:
            Tuple of (accuracy_rate, detailed_results)
        """
        correct = 0
        details = []
        
        for i, gold in enumerate(gold_cases):
            if i > 0:
                time.sleep(self.config.api_delay_seconds)
            
            # Use ground_truth as the answer for self-validation
            answer = get_answer_fn(gold.question) if get_answer_fn else gold.ground_truth
            
            score = self.score_rubric(
                question=gold.question,
                answer=answer,
                ground_truth=gold.ground_truth,
                contexts=None,  # Gold validation doesn't need contexts
            )
            
            # Check if judge's verdict matches expected
            verdict_match = (
                score.final_decision.lower().strip() == gold.expected_verdict.lower().strip()
            )
            accuracy_match = score.accuracy >= gold.expected_min_accuracy
            is_correct = verdict_match and accuracy_match
            
            if is_correct:
                correct += 1
            
            details.append({
                "gold_id": gold.id,
                "question": gold.question,
                "expected_verdict": gold.expected_verdict,
                "judge_verdict": score.final_decision,
                "expected_min_accuracy": gold.expected_min_accuracy,
                "judge_accuracy": score.accuracy,
                "verdict_match": verdict_match,
                "accuracy_match": accuracy_match,
                "is_correct": is_correct,
                "all_scores": score.model_dump(),
            })
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} [{gold.id}] verdict={score.final_decision} "
                  f"acc={score.accuracy} (expected: {gold.expected_verdict}, min_acc={gold.expected_min_accuracy})")
        
        accuracy_rate = correct / len(gold_cases) if gold_cases else 0.0
        return accuracy_rate, details
