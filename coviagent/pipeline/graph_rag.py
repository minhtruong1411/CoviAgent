"""
GraphRAG Pipeline — Hybrid Vector + Graph retrieval.

Architecture:
1. Vector Search (Qdrant): Semantic similarity search for initial top-k chunks.
2. Graph Expansion (Neo4j): For each retrieved chunk, find neighboring chunks
   connected via graph relationships to gather additional context.
3. Merge + Deduplicate: Combine both sources, remove duplicates, rank by score.
4. LLM Generation: Feed merged context to LLM for answer generation.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase

from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate

from coviagent.core.interfaces import BasePipeline, BaseRetriever

logger = logging.getLogger(__name__)

DEFAULT_GRAPHRAG_PROMPT = (
    "Bạn là một trợ lý y tế chuyên nghiệp về COVID-19.\n\n"
    "NGUYÊN TẮC BẮT BUỘC:\n"
    "- CHỈ sử dụng thông tin có trong ngữ cảnh bên dưới để trả lời.\n"
    "- TUYỆT ĐỐI KHÔNG được bịa, suy luận, hoặc thêm bất kỳ thông tin y tế nào không có trong ngữ cảnh.\n"
    "- Nếu ngữ cảnh hoàn toàn không chứa bất kỳ thông tin nào liên quan, hãy nói rõ: "
    "'Tài liệu hiện tại không có thông tin về vấn đề này.'\n"
    "- Trả lời bằng tiếng Việt, trình bày rõ ràng có cấu trúc.\n\n"
    "Ngữ cảnh (từ Vector Search + Graph Expansion):\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Câu hỏi: {query_str}\n"
    "Trả lời (chỉ dựa trên ngữ cảnh ở trên): "
)


class GraphRAGPipeline(BasePipeline):
    """
    Hybrid GraphRAG Pipeline combining Vector semantic search (Qdrant)
    with Graph structural expansion (Neo4j).
    
    This approach addresses the weakness of pure KG-Chatbot (keyword miss → empty results)
    by using vector search as the primary retrieval, then enriching with graph context.
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        neo4j_uri: str = None,
        neo4j_username: str = None,
        neo4j_password: str = None,
        neo4j_database: str = None,
        expand_hops: int = 1,
        max_graph_chunks: int = 5,
        prompt_template: str = DEFAULT_GRAPHRAG_PROMPT,
    ):
        """
        Args:
            retriever: Vector retriever (Qdrant-based) for semantic search.
            llm: LLM for answer generation.
            neo4j_uri: Neo4j connection URI.
            neo4j_username: Neo4j username.
            neo4j_password: Neo4j password.
            neo4j_database: Neo4j database name.
            expand_hops: Number of hops to traverse in graph expansion.
            max_graph_chunks: Maximum additional chunks from graph expansion.
            prompt_template: Prompt template for LLM generation.
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt = PromptTemplate(prompt_template)
        self.expand_hops = expand_hops
        self.max_graph_chunks = max_graph_chunks
        
        # Neo4j connection (fallback to env vars)
        uri = neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        username = neo4j_username or os.environ.get("NEO4J_USERNAME", "neo4j")
        password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "password")
        self.neo4j_database = neo4j_database or os.environ.get("NEO4J_DATABASE", "neo4j")
        
        self.neo4j_driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Verify connectivity
        try:
            self.neo4j_driver.verify_connectivity()
            logger.info("GraphRAG: Neo4j connected successfully.")
        except Exception as e:
            logger.warning(f"GraphRAG: Neo4j connection failed ({e}). "
                         f"Graph expansion will be skipped, falling back to vector-only.")
            self.neo4j_driver = None
    
    def _graph_expand(self, seed_texts: List[str]) -> List[str]:
        """
        Given seed chunk texts from vector search, find related chunks
        in Neo4j by traversing graph relationships.
        
        Strategy:
        1. Find Chunk nodes whose text matches any of the seed texts.
        2. Traverse relationships from those chunks to find neighboring nodes.
        3. Return texts from neighboring nodes (deduped against seeds).
        
        Args:
            seed_texts: Chunk texts from vector retrieval.
            
        Returns:
            Additional chunk texts from graph neighbors.
        """
        if not self.neo4j_driver or not seed_texts:
            return []
        
        additional_texts = []
        
        try:
            # Use a snippet from each seed to match Chunk nodes in Neo4j,
            # then traverse relationships to find connected chunks.
            # We use CONTAINS with a short substring to increase match probability.
            seed_snippets = []
            for text in seed_texts[:3]:  # Limit to top-3 seeds for efficiency
                # Take first 80 chars as a matching snippet
                snippet = text[:80].strip().replace("'", "\\'").replace('"', '\\"')
                if len(snippet) > 20:
                    seed_snippets.append(snippet)
            
            if not seed_snippets:
                return []
            
            # Build Cypher: match seed chunks, then traverse 1-hop relationships
            # to find connected nodes with text content.
            where_clauses = " OR ".join(
                f"c.text CONTAINS '{s}'" for s in seed_snippets
            )
            
            cypher = f"""
                MATCH (c:Chunk)-[r]-(neighbor)
                WHERE ({where_clauses})
                  AND neighbor.text IS NOT NULL
                RETURN DISTINCT neighbor.text AS text
                LIMIT {self.max_graph_chunks}
            """
            
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                result = session.run(cypher)
                for record in result:
                    text = record["text"]
                    # Only add if not already in seed_texts (dedup)
                    if text and text not in seed_texts:
                        additional_texts.append(text)
            
            # If no relationship-based expansion found, try a fallback:
            # Find chunks that share keywords with the seed chunks
            if not additional_texts and seed_snippets:
                # Extract key terms from seed snippets for keyword search
                fallback_cypher = f"""
                    MATCH (c:Chunk)
                    WHERE ({where_clauses})
                    WITH collect(c.text) AS matched
                    MATCH (other:Chunk)
                    WHERE NOT other.text IN matched
                    WITH other, matched,
                         reduce(score = 0, m IN matched |
                            score + CASE WHEN other.text CONTAINS substring(m, 0, 30) THEN 1 ELSE 0 END
                         ) AS relevance
                    WHERE relevance > 0
                    RETURN DISTINCT other.text AS text
                    LIMIT {self.max_graph_chunks}
                """
                try:
                    with self.neo4j_driver.session(database=self.neo4j_database) as session:
                        result = session.run(fallback_cypher)
                        for record in result:
                            text = record["text"]
                            if text and text not in seed_texts:
                                additional_texts.append(text)
                except Exception:
                    pass  # Fallback query failed, that's OK
                    
        except Exception as e:
            logger.warning(f"GraphRAG graph expansion failed: {e}")
        
        return additional_texts
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute the hybrid GraphRAG pipeline:
        1. Vector retrieval (semantic search via Qdrant)
        2. Graph expansion (structural traversal via Neo4j)
        3. Merge + deduplicate contexts
        4. LLM generation
        """
        # Step 1: Vector Retrieval (primary)
        retrieval_contexts = self.retriever.retrieve(question)
        vector_texts = [ctx.text for ctx in retrieval_contexts]
        
        # Step 2: Graph Expansion (enrichment)
        graph_texts = self._graph_expand(vector_texts)
        
        # Step 3: Merge + Deduplicate
        # Vector results first (higher relevance), then graph expansion
        all_texts = list(vector_texts)  # copy
        seen = set(vector_texts)
        for gt in graph_texts:
            if gt not in seen:
                all_texts.append(gt)
                seen.add(gt)
        
        context_str = "\n\n".join(all_texts)
        
        # Step 4: LLM Generation
        formatted_prompt = self.prompt.format(
            context_str=context_str,
            query_str=question,
        )
        
        response = self.llm.complete(formatted_prompt)
        answer = str(response)
        
        return {
            "answer": answer,
            "contexts": all_texts,
            "vector_contexts": vector_texts,
            "graph_contexts": graph_texts,
        }
    
    def __del__(self):
        """Clean up Neo4j driver on garbage collection."""
        if hasattr(self, "neo4j_driver") and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
            except Exception:
                pass
