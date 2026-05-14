import json
import os
import datetime
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from coviagent.core.interfaces import BasePipeline, BaseEvaluator
from coviagent.core.models import QAItem, BenchmarkResult
from coviagent.evaluation.ragas_evaluator import RagasEvaluator

class BenchmarkRunner(BaseEvaluator):
    """
    Runner that takes an evaluation dataset and a pipeline, executes the pipeline 
    over the dataset, and runs the configured Ragas evaluation.
    """
    
    def __init__(self, experiment_name: str = "default_experiment", output_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = RagasEvaluator()

    def evaluate(self, dataset: List[QAItem], pipeline: BasePipeline) -> BenchmarkResult:
        """
        Run the full evaluation workflow.
        """
        data_dict = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        print(f"Running pipeline inference for {len(dataset)} items...")
        detailed_logs = []
        
        for item in tqdm(dataset, desc="Inference"):
            # Execute Pipeline
            result = pipeline.query(item.question)
            
            # Format for Ragas
            data_dict["question"].append(item.question)
            data_dict["answer"].append(result["answer"])
            data_dict["contexts"].append(result["contexts"])
            data_dict["ground_truth"].append(item.ground_truth)
            
            # Save detail tracking
            detailed_logs.append({
                "question": item.question,
                "ground_truth": item.ground_truth,
                "answer": result["answer"],
                "contexts_retrieved": result["contexts"]
            })

        print("Running Ragas evaluation...")
        eval_result = self.evaluator.evaluate(data_dict)
        
        benchmark_res = BenchmarkResult(
            experiment_name=self.experiment_name,
            metrics=eval_result["metrics"],
            details=eval_result["details"].to_dict(orient="records")
        )
        
        self.save_results(benchmark_res, detailed_logs)
        
        return benchmark_res

    def save_results(self, result: BenchmarkResult, inference_logs: List[Dict[str, Any]]):
        """Save benchmark metrics and detailed raw logs to output_dir."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.output_dir / f"{self.experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = exp_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(result.metrics, f, indent=4)
            
        # Save Ragas details
        details_path = exp_dir / "ragas_details.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(result.details, f, indent=4)
            
        # Save raw inference logs
        logs_path = exp_dir / "inference_logs.json"
        with open(logs_path, "w", encoding="utf-8") as f:
            json.dump(inference_logs, f, indent=4)
            
        print(f"Results saved to {exp_dir}")
