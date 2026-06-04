from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class LLMConfig(BaseSettings):
    provider: str = "openai" # Options: openai, groq, openrouter
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None  # Explicit API key (not recommended for git)
    api_key_env_var: Optional[str] = None  # Name of env var (e.g., 'GROQ_API_KEY_2')

class EmbeddingConfig(BaseSettings):
    provider: str = "openai" # Options: openai, huggingface, gemini, cohere, voyage
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536

class RetrieverConfig(BaseSettings):
    top_k: int = 5
    collection_name: str = "coviagent_docs"
    query_mode: str = "dense" # Options: dense, hybrid

class RerankerConfig(BaseSettings):
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-base"
    top_n: int = 5
    retrieve_top_k: int = 15

class JudgeConfig(BaseSettings):
    """Configuration for LLM-as-a-Judge evaluation."""
    provider: str = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    api_key: Optional[str] = None
    api_key_env_var: Optional[str] = None
    num_runs: int = 3              # Number of judge runs for majority vote
    api_delay_seconds: float = 5.0 # Delay between API calls (rate limiting)
    pass_threshold: float = 3.0    # Average rubric score >= this → Pass

class Neo4jConfig(BaseSettings):
    """Configuration for Neo4j connection (used by KG and GraphRAG pipelines)."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

class AppConfig(BaseSettings):
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    reranker: RerankerConfig = RerankerConfig()
    judge: Optional[JudgeConfig] = None
    neo4j: Optional[Neo4jConfig] = None
    
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    
    model_config = SettingsConfigDict(env_nested_delimiter="__")

