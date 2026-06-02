"""
LLM-as-a-Judge Benchmark Runner Script.

Usage:
  # Rubric evaluation (single pipeline, majority vote)
  python scripts/run_judge_benchmark.py --config configs/judge_benchmark.yaml

  # Pairwise comparison (two pipelines)
  python scripts/run_judge_benchmark.py --config configs/judge_benchmark.yaml --mode pairwise --config-b configs/openrouter_hf.yaml

  # Gold case validation (test judge reliability)
  python scripts/run_judge_benchmark.py --config configs/judge_benchmark.yaml --mode gold-validate

  # Custom options
  python scripts/run_judge_benchmark.py --config configs/judge_benchmark.yaml --questions questions.json --experiment my_exp --max-questions 5
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import Settings

from coviagent.core.config import AppConfig, LLMConfig, JudgeConfig
from coviagent.core.factories import create_llm, create_embedding
from coviagent.core.models import QAItem, GoldCase
from coviagent.retrieval.vector_retriever import VectorRetriever
from coviagent.pipeline.vector_rag import VectorRAGPipeline
from coviagent.core.interfaces import BasePipeline
from coviagent.evaluation.llm_judge import LLMJudge
from coviagent.evaluation.judge_runner import JudgeBenchmarkRunner


def load_config(config_path: str) -> AppConfig:
    """Load AppConfig from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return AppConfig(**config_dict)


def build_pipeline(config: AppConfig, p_type: str = "vector") -> BasePipeline:
    """Build a Pipeline from AppConfig (vector or kg)."""
    if p_type == "kg":
        from coviagent.pipeline.kg_rag import KGRAGPipeline
        # We use Groq model configured in LLMConfig
        return KGRAGPipeline(model_name=config.llm.model_name)
    
    # Default is vector RAG
    embed_model = create_embedding(config.embedding)
    llm = create_llm(config.llm)
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    client = QdrantClient(url=config.qdrant_url)
    retriever = VectorRetriever.from_qdrant_client(
        client=client,
        config=config.retriever,
        embed_model=embed_model,
        reranker_config=config.reranker,
    )
    return VectorRAGPipeline(retriever=retriever, llm=llm)


def build_judge(config: AppConfig) -> LLMJudge:
    """Build LLMJudge from AppConfig's judge section."""
    judge_config = config.judge or JudgeConfig()
    
    # Create judge LLM using the factory
    judge_llm_config = LLMConfig(
        provider=judge_config.provider,
        model_name=judge_config.model_name,
        temperature=judge_config.temperature,
    )
    judge_llm = create_llm(judge_llm_config)
    
    return LLMJudge(judge_llm=judge_llm, config=judge_config)


def load_questions(questions_path: str, max_questions: int = None) -> list[QAItem]:
    """Load questions from JSON file and convert to QAItem list."""
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    items = []
    
    # Handle different JSON formats
    if isinstance(data, list):
        # Format: [{question, ground_truth, ...}, ...]
        for entry in data:
            if "question" in entry and "ground_truth" in entry:
                items.append(QAItem(
                    question=entry["question"],
                    ground_truth=entry["ground_truth"],
                ))
    elif isinstance(data, dict):
        # Format: {one_hop_questions: [...], multi_hop_questions: [...]}
        for category_key in ["one_hop_questions", "multi_hop_questions"]:
            if category_key in data:
                for entry in data[category_key]:
                    # These may not have ground_truth
                    gt = entry.get("ground_truth", "")
                    if gt:
                        items.append(QAItem(
                            question=entry["question"],
                            ground_truth=gt,
                        ))
                    else:
                        print(f"  ⚠️ Skipping {entry.get('id', '?')}: no ground_truth")
    
    if max_questions and max_questions < len(items):
        items = items[:max_questions]
    
    return items


def load_gold_cases(gold_path: str) -> list[GoldCase]:
    """Load gold cases from JSON file."""
    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [GoldCase(**entry) for entry in data]


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge Benchmark for CoviAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (e.g., configs/judge_benchmark.yaml)",
    )
    parser.add_argument(
        "--pipeline-type", choices=["vector", "kg"], default="vector",
        help="Type of pipeline A: vector (default) or kg",
    )
    parser.add_argument(
        "--mode", choices=["rubric", "pairwise", "gold-validate"], default="rubric",
        help="Evaluation mode: rubric (default), pairwise, or gold-validate",
    )
    parser.add_argument(
        "--config-b",
        help="Path to second pipeline config (required for pairwise mode)",
    )
    parser.add_argument(
        "--pipeline-type-b", choices=["vector", "kg"], default="vector",
        help="Type of pipeline B for pairwise mode: vector (default) or kg",
    )
    parser.add_argument(
        "--questions", default="data/gold_cases.json",
        help="Path to questions JSON file (default: data/gold_cases.json)",
    )
    parser.add_argument(
        "--gold-cases", default="data/gold_cases.json",
        help="Path to gold cases JSON file (default: data/gold_cases.json)",
    )
    parser.add_argument(
        "--experiment", default="judge_benchmark",
        help="Experiment name for output directory",
    )
    parser.add_argument(
        "--output-dir", default="experiments/judge_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Limit number of questions to evaluate (useful for testing)",
    )
    parser.add_argument(
        "--no-majority-vote", action="store_true",
        help="Disable majority vote (single judge run per question)",
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Load config
    print(f"📁 Loading config: {args.config}")
    config = load_config(args.config)
    
    # Build judge
    print(f"🧑‍⚖️ Building judge: {(config.judge or JudgeConfig()).model_name}")
    judge = build_judge(config)
    
    # Create runner
    runner = JudgeBenchmarkRunner(
        judge=judge,
        experiment_name=args.experiment,
        output_dir=args.output_dir,
    )
    
    # Execute based on mode
    if args.mode == "rubric":
        print(f"📋 Mode: Rubric Evaluation")
        
        # Build pipeline
        print(f"🔧 Building pipeline: {config.llm.model_name} (Type: {args.pipeline_type.upper()})")
        pipeline = build_pipeline(config, p_type=args.pipeline_type)
        
        # Load questions
        print(f"📝 Loading questions: {args.questions}")
        dataset = load_questions(args.questions, args.max_questions)
        print(f"   Loaded {len(dataset)} questions with ground_truth")
        
        if not dataset:
            print("❌ No questions with ground_truth found! Check your questions file.")
            sys.exit(1)
        
        # Run evaluation
        result = runner.run_single_pipeline(
            dataset=dataset,
            pipeline=pipeline,
            use_majority_vote=not args.no_majority_vote,
        )
    
    elif args.mode == "pairwise":
        if not args.config_b:
            print("❌ Pairwise mode requires --config-b for the second pipeline!")
            sys.exit(1)
        
        print(f"⚖️  Mode: Pairwise Comparison")
        
        # Build pipeline A
        print(f"🔧 Building Pipeline A: {config.llm.model_name} (Type: {args.pipeline_type.upper()})")
        pipeline_a = build_pipeline(config, p_type=args.pipeline_type)
        
        # Build pipeline B
        print(f"📁 Loading config B: {args.config_b}")
        config_b = load_config(args.config_b)
        print(f"🔧 Building Pipeline B: {config_b.llm.model_name} (Type: {args.pipeline_type_b.upper()})")
        pipeline_b = build_pipeline(config_b, p_type=args.pipeline_type_b)
        
        # Load questions
        dataset = load_questions(args.questions, args.max_questions)
        print(f"📝 Loaded {len(dataset)} questions")
        
        if not dataset:
            print("❌ No questions with ground_truth found!")
            sys.exit(1)
        
        # Generate clear labels for comparison
        label_a = f"{args.pipeline_type.upper()}_{config.llm.provider}"
        label_b = f"{args.pipeline_type_b.upper()}_{config_b.llm.provider}"
        if label_a == label_b:
            # If same provider and type, differentiate by config name
            label_a += "_ConfigA"
            label_b += "_ConfigB"
        
        # Run pairwise comparison
        result = runner.run_pairwise(
            dataset=dataset,
            pipeline_a=pipeline_a,
            pipeline_b=pipeline_b,
            label_a=label_a,
            label_b=label_b,
        )
    
    elif args.mode == "gold-validate":
        print(f"🥇 Mode: Gold Case Validation")
        
        # Load gold cases
        print(f"📝 Loading gold cases: {args.gold_cases}")
        gold_cases = load_gold_cases(args.gold_cases)
        print(f"   Loaded {len(gold_cases)} gold cases")
        
        # Run validation (without pipeline — tests judge on ground_truth answers)
        accuracy, details = runner.run_gold_validation(gold_cases)
        
        if accuracy >= 0.8:
            print(f"\n✅ Judge reliability: GOOD ({accuracy:.1%})")
        elif accuracy >= 0.6:
            print(f"\n⚠️ Judge reliability: MODERATE ({accuracy:.1%})")
        else:
            print(f"\n❌ Judge reliability: LOW ({accuracy:.1%}) — consider using a different judge model")
    
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
