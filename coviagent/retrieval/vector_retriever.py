from typing import List, Optional
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from qdrant_client import QdrantClient

from coviagent.core.interfaces import BaseRetriever
from coviagent.core.models import RetrievalContext
from coviagent.core.config import RetrieverConfig

class VectorRetriever(BaseRetriever):
    """
    Concrete implementation of BaseRetriever using Qdrant and LlamaIndex VectorStoreIndex.
    """
    
    def __init__(self, index: VectorStoreIndex, config: RetrieverConfig):
        self.index = index
        self.config = config
        self.retriever = self.index.as_retriever(
            similarity_top_k=self.config.top_k,
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalContext]:
        """
        Execute vector search and return standard RetrievalContext objects.
        """
        # Override top_k if specified at call time
        if top_k is not None and top_k != self.config.top_k:
            temp_retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                vector_store_query_mode=VectorStoreQueryMode.DEFAULT
            )
            nodes = temp_retriever.retrieve(query)
        else:
            nodes = self.retriever.retrieve(query)

        contexts = []
        for node_with_score in nodes:
            contexts.append(
                RetrievalContext(
                    node_id=node_with_score.node.node_id,
                    text=node_with_score.node.get_content(),
                    score=node_with_score.score or 0.0,
                    metadata=node_with_score.node.metadata
                )
            )
            
        return contexts

    @classmethod
    def from_qdrant_client(
        cls, 
        client: QdrantClient, 
        config: RetrieverConfig,
        embed_model=None
    ) -> "VectorRetriever":
        """
        Factory method to initialize retriever from an existing Qdrant client.
        """
        vector_store = QdrantVectorStore(
            client=client, 
            collection_name=config.collection_name
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        return cls(index, config)
