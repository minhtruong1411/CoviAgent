from typing import List, Optional
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from qdrant_client import QdrantClient

from coviagent.core.interfaces import BaseRetriever
from coviagent.core.models import RetrievalContext
from coviagent.core.config import RetrieverConfig, RerankerConfig

class VectorRetriever(BaseRetriever):
    """
    Concrete implementation of BaseRetriever using Qdrant and LlamaIndex VectorStoreIndex.
    """
    
    def __init__(
        self, 
        index: VectorStoreIndex, 
        config: RetrieverConfig,
        reranker_config: Optional[RerankerConfig] = None
    ):
        self.index = index
        self.config = config
        self.reranker_config = reranker_config or RerankerConfig(enabled=False)
        
        # Determine query mode (dense vs hybrid)
        query_mode_str = self.config.query_mode.lower().strip() if hasattr(self.config, "query_mode") else "dense"
        self.query_mode = VectorStoreQueryMode.HYBRID if query_mode_str == "hybrid" else VectorStoreQueryMode.DEFAULT
        
        # Determine top_k to fetch initially
        self.top_k = self.reranker_config.retrieve_top_k if self.reranker_config.enabled else self.config.top_k
        
        self.retriever = self.index.as_retriever(
            similarity_top_k=self.top_k,
            vector_store_query_mode=self.query_mode
        )
        
        # Setup reranker if enabled
        if self.reranker_config.enabled:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            self.reranker = SentenceTransformerRerank(
                model=self.reranker_config.model_name,
                top_n=self.reranker_config.top_n
            )
        else:
            self.reranker = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalContext]:
        """
        Execute vector search and return standard RetrievalContext objects.
        """
        # Determine top_k for search
        if top_k is not None:
            # If custom top_k is requested and reranking is enabled, we still scale up the search
            search_top_k = self.reranker_config.retrieve_top_k if self.reranker_config.enabled else top_k
        else:
            search_top_k = self.top_k
            
        if search_top_k != self.top_k:
            temp_retriever = self.index.as_retriever(
                similarity_top_k=search_top_k,
                vector_store_query_mode=self.query_mode
            )
            nodes = temp_retriever.retrieve(query)
        else:
            nodes = self.retriever.retrieve(query)

        # Apply reranking if enabled
        if self.reranker and len(nodes) > 0:
            from llama_index.core import QueryBundle
            query_bundle = QueryBundle(query)
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)

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
        embed_model=None,
        reranker_config: Optional[RerankerConfig] = None
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
        return cls(index, config, reranker_config)
