"""
Judge Benchmark Runner — Orchestrates the full LLM-as-a-Judge evaluation workflow.

Supports 3 modes:
1. rubric     — Score a single pipeline's outputs using structured rubric + majority vote
2. pairwise   — Blind comparison between 2 pipelines 
3. gold       — Validate judge reliability using gold cases with known answers

Outputs JSON + CSV reports to the experiments directory.
"""

import json
import csv
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from coviagent.core.interfaces import BasePipeline
from coviagent.core.models import (
    QAItem, GoldCase, JudgeBenchmarkResult,
    JudgeRubricScore, PairwiseResult,
)
from coviagent.evaluation.llm_judge import LLMJudge


class JudgeBenchmarkRunner:
    """
    Orchestrates LLM-as-a-Judge benchmark evaluation.
    
    Runs inference through pipelines, then uses LLMJudge to evaluate
    outputs with structured rubrics, pairwise comparisons, and gold validation.
    """
    
    def __init__(
        self,
        judge: LLMJudge,
        experiment_name: str = "judge_benchmark",
        output_dir: str = "experiments/judge_benchmark",
    ):
        self.judge = judge
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single_pipeline(
        self,
        dataset: List[QAItem],
        pipeline: BasePipeline,
        use_majority_vote: bool = True,
    ) -> JudgeBenchmarkResult:
        """
        Evaluate a single pipeline using structured rubric scoring.
        
        For each question in the dataset:
        1. Run the pipeline to get answer + contexts
        2. Score using LLM judge rubric (with optional majority vote)
        3. Aggregate results
        
        Args:
            dataset: List of QAItem with questions and ground_truth
            pipeline: The RAG pipeline to evaluate
            use_majority_vote: If True, use majority vote across multiple runs
            
        Returns:
            JudgeBenchmarkResult with detailed scores and summary metrics
        """
        rubric_results = []
        
        print(f"\n{'='*70}")
        print(f"📋 Rubric Evaluation: {self.experiment_name}")
        print(f"📊 Dataset: {len(dataset)} questions")
        print(f"🔁 Majority vote: {'Yes (' + str(self.judge.config.num_runs) + ' runs)' if use_majority_vote else 'No'}")
        print(f"{'='*70}\n")
        
        for item in tqdm(dataset, desc="Evaluating"):
            # Step 1: Pipeline inference
            try:
                result = pipeline.query(item.question)
                answer = result["answer"]
                contexts = result.get("contexts", [])
            except Exception as e:
                answer = f"[Pipeline Error] {str(e)}"
                contexts = []
            
            # Step 2: Judge scoring
            if use_majority_vote:
                agg_score, individual_scores = self.judge.score_rubric_majority(
                    question=item.question,
                    answer=answer,
                    ground_truth=item.ground_truth,
                    contexts=contexts,
                )
                individual_data = [s.model_dump() for s in individual_scores]
            else:
                agg_score = self.judge.score_rubric(
                    question=item.question,
                    answer=answer,
                    ground_truth=item.ground_truth,
                    contexts=contexts,
                )
                individual_data = []
            
            # Print progress
            status = "✅" if agg_score.final_decision == "Pass" else "❌"
            print(f"  {status} Q: {item.question[:80]}...")
            print(f"     Acc={agg_score.accuracy} Comp={agg_score.completeness} "
                  f"Evi={agg_score.evidence_support} Hal={agg_score.hallucination} "
                  f"→ {agg_score.final_decision} (avg={agg_score.average_score:.2f})")
            
            rubric_results.append({
                "question": item.question,
                "ground_truth": item.ground_truth,
                "bot_answer": answer,
                "accuracy": agg_score.accuracy,
                "completeness": agg_score.completeness,
                "evidence_support": agg_score.evidence_support,
                "hallucination": agg_score.hallucination,
                "average_score": agg_score.average_score,
                "final_decision": agg_score.final_decision,
                "explanation": agg_score.explanation,
                "individual_runs": individual_data,
            })
        
        # Compute summary metrics
        summary = self._compute_summary_metrics(rubric_results)
        
        benchmark_result = JudgeBenchmarkResult(
            experiment_name=self.experiment_name,
            judge_model=self.judge.config.model_name,
            num_judge_runs=self.judge.config.num_runs if use_majority_vote else 1,
            rubric_results=rubric_results,
            summary_metrics=summary,
        )
        
        self.save_results(benchmark_result)
        self.print_summary(benchmark_result)
        
        return benchmark_result
    
    def run_pairwise(
        self,
        dataset: List[QAItem],
        pipeline_a: BasePipeline,
        pipeline_b: BasePipeline,
        label_a: str = "Pipeline_A",
        label_b: str = "Pipeline_B",
    ) -> JudgeBenchmarkResult:
        """
        Blind pairwise comparison between two pipelines.
        
        Model names are hidden from the judge. Position is randomized
        to mitigate ordering bias.
        
        Args:
            dataset: List of QAItem
            pipeline_a, pipeline_b: Two pipelines to compare
            label_a, label_b: Human-readable labels (NOT shown to judge)
            
        Returns:
            JudgeBenchmarkResult with pairwise comparison results
        """
        pairwise_results = []
        
        print(f"\n{'='*70}")
        print(f"⚖️  Pairwise Comparison: {label_a} vs {label_b}")
        print(f"📊 Dataset: {len(dataset)} questions")
        print(f"🔒 Blind evaluation (model names hidden from judge)")
        print(f"{'='*70}\n")
        
        for item in tqdm(dataset, desc="Comparing"):
            # Get outputs from both pipelines
            try:
                result_a = pipeline_a.query(item.question)
                answer_a = result_a["answer"]
            except Exception as e:
                answer_a = f"[Error] {str(e)}"
            
            try:
                result_b = pipeline_b.query(item.question)
                answer_b = result_b["answer"]
            except Exception as e:
                answer_b = f"[Error] {str(e)}"
            
            # Blind pairwise comparison (judge doesn't know which is which)
            pw_result = self.judge.pairwise_compare(
                question=item.question,
                output_a=answer_a,
                output_b=answer_b,
                ground_truth=item.ground_truth,
            )
            
            # Map winner back to labels
            if pw_result.winner == "A":
                winner_label = label_a
            elif pw_result.winner == "B":
                winner_label = label_b
            else:
                winner_label = "Tie"
            
            print(f"  🏆 {winner_label}: {item.question[:70]}...")
            
            pairwise_results.append({
                "question": item.question,
                "ground_truth": item.ground_truth,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "label_a": label_a,
                "label_b": label_b,
                "winner_raw": pw_result.winner,
                "winner_label": winner_label,
                "explanation": pw_result.explanation,
            })
        
        # Compute pairwise summary
        a_wins = sum(1 for r in pairwise_results if r["winner_raw"] == "A")
        b_wins = sum(1 for r in pairwise_results if r["winner_raw"] == "B")
        ties = sum(1 for r in pairwise_results if r["winner_raw"] == "Tie")
        total = len(pairwise_results)
        
        summary = {
            f"{label_a}_wins": a_wins,
            f"{label_b}_wins": b_wins,
            "ties": ties,
            f"{label_a}_win_rate": a_wins / total if total > 0 else 0,
            f"{label_b}_win_rate": b_wins / total if total > 0 else 0,
            "tie_rate": ties / total if total > 0 else 0,
        }
        
        benchmark_result = JudgeBenchmarkResult(
            experiment_name=f"{self.experiment_name}_pairwise",
            judge_model=self.judge.config.model_name,
            num_judge_runs=3,  # AB + BA + tiebreaker
            pairwise_results=pairwise_results,
            summary_metrics=summary,
        )
        
        self.save_results(benchmark_result)
        self.print_pairwise_summary(benchmark_result, label_a, label_b)
        
        return benchmark_result
    
    def run_gold_validation(
        self,
        gold_cases: List[GoldCase],
        pipeline: Optional[BasePipeline] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Validate judge reliability using gold cases.
        
        If pipeline is provided, uses pipeline output as answers.
        If not, uses ground_truth as answers (self-validation: a good judge
        should give high scores to correct answers).
        
        Returns:
            Tuple of (accuracy_rate, detailed_results)
        """
        print(f"\n{'='*70}")
        print(f"🥇 Gold Case Validation")
        print(f"📊 Gold cases: {len(gold_cases)}")
        print(f"{'='*70}\n")
        
        get_answer_fn = None
        if pipeline:
            def get_answer_fn(q):
                try:
                    return pipeline.query(q)["answer"]
                except Exception as e:
                    return f"[Error] {str(e)}"
        
        accuracy, details = self.judge.validate_gold_cases(gold_cases, get_answer_fn)
        
        print(f"\n🎯 Gold Case Accuracy: {accuracy:.1%} ({sum(1 for d in details if d['is_correct'])}/{len(details)})")
        
        # Save gold validation results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        gold_dir = self.output_dir / f"gold_validation_{timestamp}"
        gold_dir.mkdir(parents=True, exist_ok=True)
        
        with open(gold_dir / "gold_results.json", "w", encoding="utf-8") as f:
            json.dump({"accuracy": accuracy, "details": details}, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {gold_dir}")
        
        return accuracy, details
    
    # ──────────────────────────────────────────────
    #  Report Generation
    # ──────────────────────────────────────────────
    
    def _compute_summary_metrics(self, rubric_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute aggregate metrics from rubric results."""
        if not rubric_results:
            return {}
        
        n = len(rubric_results)
        
        avg_accuracy = sum(r["accuracy"] for r in rubric_results) / n
        avg_completeness = sum(r["completeness"] for r in rubric_results) / n
        avg_evidence = sum(r["evidence_support"] for r in rubric_results) / n
        avg_hallucination = sum(r["hallucination"] for r in rubric_results) / n
        avg_overall = sum(r["average_score"] for r in rubric_results) / n
        
        pass_count = sum(1 for r in rubric_results if r["final_decision"] == "Pass")
        pass_rate = pass_count / n
        
        return {
            "total_questions": n,
            "avg_accuracy": round(avg_accuracy, 3),
            "avg_completeness": round(avg_completeness, 3),
            "avg_evidence_support": round(avg_evidence, 3),
            "avg_hallucination": round(avg_hallucination, 3),
            "avg_overall_score": round(avg_overall, 3),
            "pass_count": pass_count,
            "fail_count": n - pass_count,
            "pass_rate": round(pass_rate, 3),
        }
    
    def save_results(self, result: JudgeBenchmarkResult):
        """Save benchmark results to JSON and CSV files."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.output_dir / f"{result.experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full JSON report
        report = result.model_dump()
        with open(exp_dir / "full_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save summary metrics
        with open(exp_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(result.summary_metrics, f, indent=2, ensure_ascii=False)
        
        # Save rubric results as CSV (if available)
        if result.rubric_results:
            csv_path = exp_dir / "rubric_results.csv"
            fieldnames = [
                "question", "ground_truth", "bot_answer",
                "accuracy", "completeness", "evidence_support", "hallucination",
                "average_score", "final_decision", "explanation",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(result.rubric_results)
        
        # Save pairwise results as CSV (if available)
        if result.pairwise_results:
            csv_path = exp_dir / "pairwise_results.csv"
            fieldnames = [
                "question", "ground_truth", "answer_a", "answer_b",
                "label_a", "label_b", "winner_raw", "winner_label", "explanation",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(result.pairwise_results)
        
        print(f"\n💾 Results saved to {exp_dir}")
    
    def print_summary(self, result: JudgeBenchmarkResult):
        """Print a formatted summary of rubric evaluation results."""
        m = result.summary_metrics
        
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARK SUMMARY: {result.experiment_name}")
        print(f"{'='*70}")
        print(f"  Judge Model    : {result.judge_model}")
        print(f"  Judge Runs     : {result.num_judge_runs} (majority vote)")
        print(f"  Total Questions: {m.get('total_questions', 0)}")
        print(f"{'─'*70}")
        print(f"  📈 Average Scores (0-5):")
        print(f"     Accuracy          : {m.get('avg_accuracy', 0):.3f}")
        print(f"     Completeness      : {m.get('avg_completeness', 0):.3f}")
        print(f"     Evidence Support  : {m.get('avg_evidence_support', 0):.3f}")
        print(f"     Hallucination     : {m.get('avg_hallucination', 0):.3f}")
        print(f"     Overall           : {m.get('avg_overall_score', 0):.3f}")
        print(f"{'─'*70}")
        print(f"  ✅ Pass: {m.get('pass_count', 0)} | ❌ Fail: {m.get('fail_count', 0)}")
        print(f"  📊 Pass Rate: {m.get('pass_rate', 0):.1%}")
        print(f"{'='*70}\n")
    
    def print_pairwise_summary(
        self, result: JudgeBenchmarkResult, label_a: str, label_b: str
    ):
        """Print a formatted summary of pairwise comparison results."""
        m = result.summary_metrics
        
        print(f"\n{'='*70}")
        print(f"⚖️  PAIRWISE COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"  Judge Model: {result.judge_model}")
        print(f"  {label_a} vs {label_b}")
        print(f"{'─'*70}")
        print(f"  🏆 {label_a} wins : {m.get(f'{label_a}_wins', 0)} "
              f"({m.get(f'{label_a}_win_rate', 0):.1%})")
        print(f"  🏆 {label_b} wins : {m.get(f'{label_b}_wins', 0)} "
              f"({m.get(f'{label_b}_win_rate', 0):.1%})")
        print(f"  🤝 Ties          : {m.get('ties', 0)} "
              f"({m.get('tie_rate', 0):.1%})")
        print(f"{'='*70}\n")
