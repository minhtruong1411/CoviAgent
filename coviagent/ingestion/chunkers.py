from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

class DocumentChunker:
    """Wrapper around LlamaIndex NodeParsers to ensure consistent chunking strategy."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
    def chunk_documents(self, documents: List[Document]) -> List[BaseNode]:
        """
        Chunk a list of LlamaIndex Documents into Nodes.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of BaseNode objects (chunks)
        """
        nodes = self.parser.get_nodes_from_documents(documents)
        return nodes
