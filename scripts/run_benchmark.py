import argparse
import yaml
import json
import sys
from pathlib import Path

# Add project root to path for local imports
sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from llama_index.core import Settings
from dotenv import load_dotenv

from coviagent.core.config import AppConfig
from coviagent.core.models import QAItem
from coviagent.core.factories import create_llm, create_embedding
from coviagent.retrieval.vector_retriever import VectorRetriever
from coviagent.pipeline.vector_rag import VectorRAGPipeline
from coviagent.evaluation.runner import BenchmarkRunner

def load_eval_dataset(path: str):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                dataset.append(QAItem(**data))
    return dataset

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run RAG Benchmark")
    parser.add_argument("--config", type=str, default="configs/default_vector_rag.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="data/eval/sample.jsonl", help="Path to evaluation dataset")
    parser.add_argument("--exp_name", type=str, default="vector_rag_baseline", help="Experiment name")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)

    # Set up Embeddings & LLM via Factory (switch provider by editing YAML only)
    embed_model = create_embedding(config.embedding)
    llm = create_llm(config.llm)
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    print(f"📦 LLM: {config.llm.provider} / {config.llm.model_name}")
    print(f"📦 Embedding: {config.embedding.provider} / {config.embedding.model_name}")

    # Connect to local Docker Qdrant instance
    print(f"Connecting to Qdrant at {config.qdrant_url}...")
    client = QdrantClient(url=config.qdrant_url) 
    
    # Initialize components
    retriever = VectorRetriever.from_qdrant_client(
        client=client, 
        config=config.retriever,
        embed_model=embed_model
    )
    
    pipeline = VectorRAGPipeline(retriever=retriever, llm=llm)

    # Load Evaluation Dataset
    print(f"Loading dataset from {args.dataset}...")
    if not Path(args.dataset).exists():
        print(f"Dataset {args.dataset} not found! Please create it.")
        return
        
    dataset = load_eval_dataset(args.dataset)

    # Run Benchmark
    print("Starting Benchmark Runner...")
    runner = BenchmarkRunner(experiment_name=args.exp_name)
    result = runner.evaluate(dataset=dataset, pipeline=pipeline)
    
    print("\nBenchmark Complete. Summary Metrics:")
    for k, v in result.metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
