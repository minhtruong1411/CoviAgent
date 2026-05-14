from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class LLMConfig(BaseSettings):
    provider: str = "openai" # Options: openai, groq, openrouter
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: Optional[int] = None

class EmbeddingConfig(BaseSettings):
    provider: str = "openai" # Options: openai, huggingface, gemini, cohere, voyage
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536

class RetrieverConfig(BaseSettings):
    top_k: int = 5
    collection_name: str = "coviagent_docs"

class AppConfig(BaseSettings):
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    
    model_config = SettingsConfigDict(env_nested_delimiter="__")
