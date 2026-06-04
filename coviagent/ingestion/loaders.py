from typing import List, Union
from pathlib import Path
from llama_index.core import Document, SimpleDirectoryReader

def load_documents_from_dir(dir_path: Union[str, Path]) -> List[Document]:
    """
    Load all documents from a directory using LlamaIndex SimpleDirectoryReader.
    Adds additional domain-specific metadata if needed.
    """
    reader = SimpleDirectoryReader(
        input_dir=str(dir_path),
        recursive=True,
        required_exts=[".txt", ".md", ".json", ".csv", ".pdf"]
    )
    docs = reader.load_data()
    
    # Inject baseline medical metadata if necessary
    for doc in docs:
        doc.metadata["domain"] = "medical_dialogue"
        doc.metadata["source_type"] = "raw_corpus"
        
    return docs
