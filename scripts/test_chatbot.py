import argparse
import yaml
import sys
import io
from pathlib import Path

# Add project root to path for local imports
sys.path.append(str(Path(__file__).parent.parent))

# Force UTF-8 output to avoid charmap codec errors on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from qdrant_client import QdrantClient
from llama_index.core import Settings
from dotenv import load_dotenv

from coviagent.core.config import AppConfig
from coviagent.core.factories import create_llm, create_embedding
from coviagent.retrieval.vector_retriever import VectorRetriever
from coviagent.pipeline.vector_rag import VectorRAGPipeline

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Test Chatbot interactively")
    parser.add_argument("--config", type=str, default="configs/default_vector_rag.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)

    print("Initializing...")
    
    # Setup Factory Models
    embed_model = create_embedding(config.embedding)
    llm = create_llm(config.llm)
    
    Settings.embed_model = embed_model
    Settings.llm = llm

    # Setup Retrieval
    try:
        client = QdrantClient(url=config.qdrant_url)
        retriever = VectorRetriever.from_qdrant_client(
            client=client, 
            config=config.retriever,
            embed_model=embed_model,
            reranker_config=config.reranker
        )
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Did you forget to run Qdrant? Try: docker compose up -d")
        return

    pipeline = VectorRAGPipeline(retriever=retriever, llm=llm)

    print("\n" + "="*50)
    print(f"CoviAgent Chatbot is Ready!")
    print(f"LLM: {config.llm.provider} ({config.llm.model_name})")
    print(f"Embeddings: {config.embedding.provider} ({config.embedding.model_name})")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            query = input("You: ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
                
            response = pipeline.query(query)
            
            print(f"\n[Bot]: {response['answer']}\n")
            
            # Print sources for debugging
            if response['contexts']:
                print(f"-- Retrieved {len(response['contexts'])} context chunks --\n")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
