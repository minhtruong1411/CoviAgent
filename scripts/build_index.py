import argparse
import yaml
import sys
from pathlib import Path

# Add project root to path for local imports
sys.path.append(str(Path(__file__).parent.parent))

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from coviagent.core.config import AppConfig
from coviagent.core.factories import create_embedding
from coviagent.ingestion.loaders import load_documents_from_dir
from coviagent.ingestion.chunkers import DocumentChunker

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build Qdrant Vector Index from Raw Data")
    parser.add_argument("--config", type=str, default="configs/default_vector_rag.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw text documents")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)

    # Setup Embedding Model via Factory
    print(f"Setting up embedding model: {config.embedding.model_name} (Provider: {config.embedding.provider})")
    embed_model = create_embedding(config.embedding)
    Settings.embed_model = embed_model

    # Ingest Data
    print(f"Loading documents from {args.data_dir}...")
    docs = load_documents_from_dir(args.data_dir)
    print(f"Loaded {len(docs)} documents.")

    if not docs:
        print("No documents found. Please add txt files to data/raw/.")
        return

    # Chunking
    print("Chunking documents...")
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    nodes = chunker.chunk_documents(docs)
    print(f"Created {len(nodes)} nodes/chunks.")

    # Setup Qdrant
    print(f"Connecting to Qdrant at {config.qdrant_url}...")
    client = QdrantClient(url=config.qdrant_url) 
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=config.retriever.collection_name
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build Index
    print("Building Vector Index (this might take a while)...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("Index built successfully!")

if __name__ == "__main__":
    main()
