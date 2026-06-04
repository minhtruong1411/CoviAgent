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
        pipeline: Optional[BasePipeline] = None,
        use_majority_vote: bool = True,
        phase: str = "all",
        answers_file: str = "experiments/intermediate_answers.json",
    ) -> Optional[JudgeBenchmarkResult]:
        """
        Evaluate a single pipeline using structured rubric scoring.
        
        Args:
            dataset: List of QAItem with questions and ground_truth
            pipeline: The RAG pipeline to evaluate (not needed if phase == "judge")
            use_majority_vote: If True, use majority vote across multiple runs
            phase: "all", "generate", or "judge"
            answers_file: Path to save/load generated answers
            
        Returns:
            JudgeBenchmarkResult if phase in ("all", "judge"), else None
        """
        print(f"\n{'='*70}")
        print(f"📋 Rubric Evaluation: {self.experiment_name} (Phase: {phase})")
        print(f"📊 Dataset: {len(dataset)} questions")
        print(f"🔁 Majority vote: {'Yes (' + str(self.judge.config.num_runs) + ' runs)' if use_majority_vote else 'No'}")
        print(f"{'='*70}\n")
        
        generated_answers = []
        if phase in ("all", "generate"):
            print(f"📝 Generating pipeline outputs...")
            for item in tqdm(dataset, desc="Generating"):
                try:
                    result = pipeline.query(item.question)
                    answer = result["answer"]
                    contexts = result.get("contexts", [])
                except Exception as e:
                    answer = f"[Pipeline Error] {str(e)}"
                    contexts = []
                generated_answers.append({
                    "question": item.question,
                    "ground_truth": item.ground_truth,
                    "answer": answer,
                    "contexts": contexts
                })
            
            if phase == "generate":
                Path(answers_file).parent.mkdir(parents=True, exist_ok=True)
                with open(answers_file, "w", encoding="utf-8") as f:
                    json.dump(generated_answers, f, indent=2, ensure_ascii=False)
                print(f"💾 Generated answers saved to {answers_file}")
                return None
        else:
            print(f"📂 Loading generated answers from {answers_file}...")
            with open(answers_file, "r", encoding="utf-8") as f:
                generated_answers = json.load(f)

        rubric_results = []
        for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
            gen_item = generated_answers[idx]
            answer = gen_item["answer"]
            contexts = gen_item.get("contexts", [])
            
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
        pipeline_a: Optional[BasePipeline] = None,
        pipeline_b: Optional[BasePipeline] = None,
        label_a: str = "Pipeline_A",
        label_b: str = "Pipeline_B",
        phase: str = "all",
        answers_file: str = "experiments/intermediate_pairwise.json",
    ) -> Optional[JudgeBenchmarkResult]:
        """
        Blind pairwise comparison between two pipelines.
        
        Args:
            dataset: List of QAItem
            pipeline_a, pipeline_b: Two pipelines to compare (not needed if phase == "judge")
            label_a, label_b: Human-readable labels (NOT shown to judge)
            phase: "all", "generate", or "judge"
            answers_file: Path to save/load generated answers
            
        Returns:
            JudgeBenchmarkResult if phase in ("all", "judge"), else None
        """
        print(f"\n{'='*70}")
        print(f"⚖️  Pairwise Comparison: {label_a} vs {label_b} (Phase: {phase})")
        print(f"📊 Dataset: {len(dataset)} questions")
        print(f"🔒 Blind evaluation (model names hidden from judge)")
        print(f"{'='*70}\n")
        
        generated_answers = []
        if phase in ("all", "generate"):
            print(f"📝 Generating pipeline outputs...")
            for item in tqdm(dataset, desc="Generating"):
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
                    
                generated_answers.append({
                    "question": item.question,
                    "ground_truth": item.ground_truth,
                    "answer_a": answer_a,
                    "answer_b": answer_b
                })
                
            if phase == "generate":
                Path(answers_file).parent.mkdir(parents=True, exist_ok=True)
                with open(answers_file, "w", encoding="utf-8") as f:
                    json.dump(generated_answers, f, indent=2, ensure_ascii=False)
                print(f"💾 Generated pairwise answers saved to {answers_file}")
                return None
        else:
            print(f"📂 Loading generated answers from {answers_file}...")
            with open(answers_file, "r", encoding="utf-8") as f:
                generated_answers = json.load(f)

        pairwise_results = []
        for idx, item in enumerate(tqdm(dataset, desc="Comparing")):
            gen_item = generated_answers[idx]
            answer_a = gen_item["answer_a"]
            answer_b = gen_item["answer_b"]
            
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
        
        avg_correctness = sum(r["correctness"] for r in rubric_results) / n
        avg_completeness = sum(r["completeness"] for r in rubric_results) / n
        avg_relevance = sum(r["relevance"] for r in rubric_results) / n
        avg_faithfulness = sum(r["faithfulness"] for r in rubric_results) / n
        avg_context_quality = sum(r["context_quality"] for r in rubric_results) / n
        avg_overall = sum(r["average_score"] for r in rubric_results) / n
        
        pass_count = sum(1 for r in rubric_results if r["final_decision"] == "Pass")
        pass_rate = pass_count / n
        
        return {
            "total_questions": n,
            "avg_correctness": round(avg_correctness, 3),
            "avg_completeness": round(avg_completeness, 3),
            "avg_relevance": round(avg_relevance, 3),
            "avg_faithfulness": round(avg_faithfulness, 3),
            "avg_context_quality": round(avg_context_quality, 3),
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
                "correctness", "completeness", "relevance", "faithfulness", "context_quality",
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
        print(f"     Correctness       : {m.get('avg_correctness', 0):.3f}")
        print(f"     Completeness      : {m.get('avg_completeness', 0):.3f}")
        print(f"     Relevance         : {m.get('avg_relevance', 0):.3f}")
        print(f"     Faithfulness      : {m.get('avg_faithfulness', 0):.3f}")
        print(f"     Context Quality   : {m.get('avg_context_quality', 0):.3f}")
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
    
    def run_round_robin(
        self,
        dataset: List[QAItem],
        pipelines: Optional[Dict[str, BasePipeline]] = None,
        labels: Optional[List[str]] = None,
        phase: str = "all",
        answers_file: str = "experiments/intermediate_round_robin.json",
    ) -> Optional[JudgeBenchmarkResult]:
        """
        Round-robin pairwise comparison among N pipelines.
        
        Args:
            dataset: List of QAItem with questions and ground_truth
            pipelines: Dict mapping label -> pipeline instance (not needed if phase == "judge")
            labels: List of pipeline labels (required if phase == "judge")
            phase: "all", "generate", or "judge"
            answers_file: Path to save/load generated answers
        
        Returns:
            JudgeBenchmarkResult if phase in ("all", "judge"), else None
        """
        if phase in ("all", "generate") and pipelines is not None:
            pipeline_labels = list(pipelines.keys())
        elif phase == "judge" and labels is not None:
            pipeline_labels = labels
        else:
            raise ValueError("Must provide either pipelines (generate) or labels (judge)")

        all_pairwise_results = []
        
        # Generate all unique pairs
        pairs = []
        for i in range(len(pipeline_labels)):
            for j in range(i + 1, len(pipeline_labels)):
                pairs.append((pipeline_labels[i], pipeline_labels[j]))
        
        print(f"\n{'='*70}")
        print(f"🏆 Round-Robin Benchmark: {' vs '.join(pipeline_labels)} (Phase: {phase})")
        print(f"📊 Dataset: {len(dataset)} questions")
        print(f"⚖️  Matchups: {len(pairs)} pairs")
        print(f"🔒 Blind evaluation (pipeline names hidden from judge)")
        print(f"{'='*70}\n")
        
        # Pre-compute or load pipeline outputs
        if phase in ("all", "generate"):
            print("📝 Step 1/2: Generating pipeline outputs...")
            pipeline_outputs = {}
            for label, pipeline in pipelines.items():
                print(f"  🔧 Running {label}...")
                outputs = []
                for item in tqdm(dataset, desc=f"  {label}"):
                    try:
                        result = pipeline.query(item.question)
                        outputs.append(result["answer"])
                    except Exception as e:
                        outputs.append(f"[Error] {str(e)}"
                        )
                pipeline_outputs[label] = outputs
                
            if phase == "generate":
                Path(answers_file).parent.mkdir(parents=True, exist_ok=True)
                with open(answers_file, "w", encoding="utf-8") as f:
                    json.dump(pipeline_outputs, f, indent=2, ensure_ascii=False)
                print(f"💾 Generated round-robin answers saved to {answers_file}")
                return None
        else:
            print(f"📂 Loading generated answers from {answers_file}...")
            with open(answers_file, "r", encoding="utf-8") as f:
                pipeline_outputs = json.load(f)
        
        # Run all pairwise comparisons
        print(f"\n📝 Step 2/2: Running pairwise comparisons...")
        for label_a, label_b in pairs:
            print(f"\n  ⚖️  {label_a} vs {label_b}")
            
            for idx, item in enumerate(tqdm(dataset, desc=f"  {label_a} vs {label_b}")):
                answer_a = pipeline_outputs[label_a][idx]
                answer_b = pipeline_outputs[label_b][idx]
                
                # Blind pairwise comparison
                pw_result = self.judge.pairwise_compare(
                    question=item.question,
                    output_a=answer_a,
                    output_b=answer_b,
                    ground_truth=item.ground_truth,
                )
                
                # Map winner
                if pw_result.winner == "A":
                    winner_label = label_a
                elif pw_result.winner == "B":
                    winner_label = label_b
                else:
                    winner_label = "Tie"
                
                print(f"    🏆 {winner_label}: {item.question[:60]}...")
                
                all_pairwise_results.append({
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
        
        # Build leaderboard
        leaderboard = {label: {"wins": 0, "losses": 0, "ties": 0} for label in pipeline_labels}
        
        for r in all_pairwise_results:
            la, lb = r["label_a"], r["label_b"]
            if r["winner_raw"] == "A":
                leaderboard[la]["wins"] += 1
                leaderboard[lb]["losses"] += 1
            elif r["winner_raw"] == "B":
                leaderboard[lb]["wins"] += 1
                leaderboard[la]["losses"] += 1
            else:
                leaderboard[la]["ties"] += 1
                leaderboard[lb]["ties"] += 1
        
        # Add win rates
        for label in pipeline_labels:
            total = leaderboard[label]["wins"] + leaderboard[label]["losses"] + leaderboard[label]["ties"]
            leaderboard[label]["total_matches"] = total
            leaderboard[label]["win_rate"] = leaderboard[label]["wins"] / total if total > 0 else 0
        
        # Sort by wins descending
        sorted_labels = sorted(pipeline_labels, key=lambda l: leaderboard[l]["wins"], reverse=True)
        leaderboard["ranking"] = sorted_labels
        
        summary = {
            "leaderboard": leaderboard,
            "total_questions": len(dataset),
            "total_matchups": len(pairs),
            "total_comparisons": len(all_pairwise_results),
        }
        
        benchmark_result = JudgeBenchmarkResult(
            experiment_name=f"{self.experiment_name}_round_robin",
            judge_model=self.judge.config.model_name,
            num_judge_runs=3,
            pairwise_results=all_pairwise_results,
            summary_metrics=summary,
        )
        
        self.save_results(benchmark_result)
        self.print_round_robin_summary(benchmark_result, pipeline_labels)
        
        return benchmark_result
    
    def print_round_robin_summary(
        self, result: JudgeBenchmarkResult, labels: List[str]
    ):
        """Print a formatted leaderboard for round-robin results."""
        m = result.summary_metrics
        lb = m.get("leaderboard", {})
        ranking = lb.get("ranking", labels)
        
        print(f"\n{'='*70}")
        print(f"🏆 ROUND-ROBIN LEADERBOARD")
        print(f"{'='*70}")
        print(f"  Judge Model : {result.judge_model}")
        print(f"  Questions   : {m.get('total_questions', 0)}")
        print(f"  Matchups    : {m.get('total_matchups', 0)} pairs")
        print(f"{'─'*70}")
        print(f"  {'Rank':<6} {'Pipeline':<20} {'Wins':>6} {'Losses':>8} {'Ties':>6} {'Win Rate':>10}")
        print(f"  {'─'*56}")
        
        for rank, label in enumerate(ranking, 1):
            stats = lb.get(label, {})
            medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
            print(f"  {medal} {rank:<4} {label:<20} {stats.get('wins', 0):>6} "
                  f"{stats.get('losses', 0):>8} {stats.get('ties', 0):>6} "
                  f"{stats.get('win_rate', 0):>9.1%}")
        
        print(f"{'='*70}\n")
    
    def run_compare_rubric(
        self,
        dataset: List[QAItem],
        pipelines: Optional[Dict[str, BasePipeline]] = None,
        labels: Optional[List[str]] = None,
        phase: str = "all",
        answers_file: str = "experiments/intermediate_compare_rubric.json",
    ) -> Optional[JudgeBenchmarkResult]:
        """
        Evaluate multiple pipelines independently using the Rubric (0-5 scores) 
        and compare them side-by-side in a leaderboard.
        """
        if phase in ("all", "generate") and pipelines is not None:
            pipeline_labels = list(pipelines.keys())
        elif phase == "judge" and labels is not None:
            pipeline_labels = labels
        else:
            raise ValueError("Must provide either pipelines (generate) or labels (judge)")

        # Phase: Generate
        all_generated = {}
        if phase in ("all", "generate"):
            print(f"\n🔄 GENERATE PHASE (Compare Rubric): {len(pipeline_labels)} pipelines")
            
            for label in pipeline_labels:
                pipeline = pipelines[label]
                all_generated[label] = []
                print(f"Generating for {label}...")
                
                for item in tqdm(dataset, desc=f"Generating {label}"):
                    try:
                        resp = pipeline.query(item.question)
                        all_generated[label].append({
                            "question": item.question,
                            "ground_truth": item.ground_truth,
                            "bot_answer": str(resp),
                            "contexts": [c.text for c in resp.source_nodes] if hasattr(resp, 'source_nodes') else []
                        })
                    except Exception as e:
                        all_generated[label].append({
                            "question": item.question,
                            "ground_truth": item.ground_truth,
                            "bot_answer": f"Error: {e}",
                            "contexts": []
                        })
            
            with open(answers_file, "w", encoding="utf-8") as f:
                json.dump(all_generated, f, indent=2, ensure_ascii=False)
            print(f"💾 Generated answers saved to {answers_file}")
            
            if phase == "generate":
                return None
                
        # Phase: Judge
        print(f"\n⚖️ JUDGE PHASE (Compare Rubric)")
        if phase == "judge":
            print(f"📁 Loading answers from {answers_file}")
            try:
                with open(answers_file, "r", encoding="utf-8") as f:
                    all_generated = json.load(f)
            except FileNotFoundError:
                print(f"❌ Error: {answers_file} not found. Run generate phase first!")
                return None

        # Evaluate each pipeline
        all_results = {}
        for label in pipeline_labels:
            print(f"🧑‍⚖️ Evaluating {label}...")
            rubric_results = []
            gen_data = all_generated.get(label, [])
            
            for gen in tqdm(gen_data, desc=f"Judging {label}"):
                agg_score, _ = self.judge.score_rubric_majority(
                    question=gen["question"],
                    answer=gen["bot_answer"],
                    ground_truth=gen["ground_truth"],
                    contexts=gen["contexts"],
                )
                
                row = {
                    "pipeline": label,
                    "question": gen["question"],
                    "ground_truth": gen["ground_truth"],
                    "bot_answer": gen["bot_answer"],
                }
                row.update(agg_score.model_dump())
                rubric_results.append(row)
                
            all_results[label] = rubric_results
            
        # Combine results
        combined_rubric_results = []
        summary_by_pipeline = {}
        
        for label, results in all_results.items():
            combined_rubric_results.extend(results)
            summary_by_pipeline[label] = self._compute_summary_metrics(results)
            
        # Determine ranking based on average overall score
        ranking = sorted(pipeline_labels, key=lambda x: summary_by_pipeline[x].get("avg_overall_score", 0), reverse=True)
        
        summary = {
            "total_questions": len(dataset),
            "pipelines_evaluated": pipeline_labels,
            "ranking": ranking,
            "pipeline_summaries": summary_by_pipeline
        }
        
        benchmark_result = JudgeBenchmarkResult(
            experiment_name=f"{self.experiment_name}_compare_rubric",
            judge_model=self.judge.config.model_name,
            num_judge_runs=self.judge.config.num_runs,
            rubric_results=combined_rubric_results,
            summary_metrics=summary,
        )
        
        self.save_results(benchmark_result)
        self.print_compare_rubric_summary(benchmark_result)
        
        return benchmark_result
        
    def print_compare_rubric_summary(self, result: JudgeBenchmarkResult):
        """Print a formatted leaderboard for compare-rubric results."""
        m = result.summary_metrics
        ranking = m.get("ranking", [])
        summaries = m.get("pipeline_summaries", {})
        
        print(f"\n{'='*95}")
        print(f"📊 COMPARE RUBRIC LEADERBOARD")
        print(f"{'='*95}")
        print(f"  Judge Model : {result.judge_model}")
        print(f"  Questions   : {m.get('total_questions', 0)}")
        print(f"{'─'*95}")
        print(f"  {'Rank':<4} | {'Pipeline':<18} | {'Correct':<7} | {'Complet':<7} | {'Relev':<7} | {'Faith':<7} | {'Context':<7} | {'OVERALL':<7}")
        print(f"  {'-'*95}")
        
        for rank, label in enumerate(ranking, 1):
            s = summaries.get(label, {})
            medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
            print(f"  {medal} {rank:<2} | {label:<18} | "
                  f"{s.get('avg_correctness', 0):>7.2f} | "
                  f"{s.get('avg_completeness', 0):>7.2f} | "
                  f"{s.get('avg_relevance', 0):>7.2f} | "
                  f"{s.get('avg_faithfulness', 0):>7.2f} | "
                  f"{s.get('avg_context_quality', 0):>7.2f} | "
                  f"{s.get('avg_overall_score', 0):>7.2f}")
        
        print(f"{'='*95}\n")
