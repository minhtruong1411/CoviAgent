"""
Factory module for creating LLM and Embedding instances from config.

Centralizes provider switching logic so that changing a provider
only requires editing the YAML config file — no code changes needed.

Usage:
    from coviagent.core.factories import create_llm, create_embedding
    from coviagent.core.config import LLMConfig, EmbeddingConfig

    llm = create_llm(LLMConfig(provider="groq", model_name="llama3-8b-8192"))
    embed = create_embedding(EmbeddingConfig(provider="huggingface", model_name="BAAI/bge-small-en-v1.5"))
"""

from typing import Dict, Any
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from coviagent.core.config import LLMConfig, EmbeddingConfig


# ──────────────────────────────────────────────
#  LLM Factory
# ──────────────────────────────────────────────

def create_llm(config: LLMConfig) -> LLM:
    """
    Create an LLM instance based on provider config.
    
    Supported providers:
        - openai     : GPT-4o, GPT-4o-mini, etc.
        - groq       : Llama-3, Mixtral on Groq Cloud
        - openrouter : Any model via OpenRouter proxy
    
    Args:
        config: LLMConfig with provider, model_name, temperature, max_tokens.
        
    Returns:
        LLM: A llama-index LLM instance ready to use.
        
    Raises:
        ValueError: If the provider is not supported.
    """
    provider = config.provider.lower().strip()
    
    import os
    actual_api_key = config.api_key
    if not actual_api_key and getattr(config, "api_key_env_var", None):
        actual_api_key = os.environ.get(config.api_key_env_var)
    
    common_kwargs: Dict[str, Any] = {
        "model": config.model_name,
        "temperature": config.temperature,
    }
    if config.max_tokens is not None:
        common_kwargs["max_tokens"] = config.max_tokens
    if actual_api_key:
        common_kwargs["api_key"] = actual_api_key

    if provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(**common_kwargs)
    
    elif provider == "groq":
        from llama_index.llms.groq import Groq
        return Groq(**common_kwargs)
    
    elif provider == "openrouter":
        from llama_index.llms.openrouter import OpenRouter
        return OpenRouter(**common_kwargs)
        
    elif provider == "gemini":
        from llama_index.llms.gemini import Gemini
        return Gemini(**common_kwargs)
    
    else:
        supported = ["openai", "groq", "openrouter", "gemini"]
        raise ValueError(
            f"Unknown LLM provider: '{config.provider}'. "
            f"Supported providers: {supported}"
        )


# ──────────────────────────────────────────────
#  Embedding Factory
# ──────────────────────────────────────────────

def create_embedding(config: EmbeddingConfig) -> BaseEmbedding:
    """
    Create an Embedding model instance based on provider config.
    
    Supported providers:
        - openai      : text-embedding-3-small, text-embedding-3-large, etc.
        - huggingface  : Any SentenceTransformer model (e.g. BAAI/bge-small-en-v1.5)
    
    Args:
        config: EmbeddingConfig with provider, model_name, dimension.
        
    Returns:
        BaseEmbedding: A llama-index embedding model instance.
        
    Raises:
        ValueError: If the provider is not supported.
    """
    provider = config.provider.lower().strip()
    
    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(
            model=config.model_name,
            dimensions=config.dimension,
        )
    
    elif provider == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name=config.model_name)
        
    elif provider == "gemini":
        from llama_index.embeddings.gemini import GeminiEmbedding
        return GeminiEmbedding(model_name=config.model_name)
        
    elif provider == "cohere":
        from llama_index.embeddings.cohere import CohereEmbedding
        return CohereEmbedding(
            model_name=config.model_name,
            input_type="search_document"
        )
        
    elif provider == "voyage":
        from llama_index.embeddings.voyageai import VoyageEmbedding
        return VoyageEmbedding(
            model_name=config.model_name,
            voyage_api_key=None  # will be picked up from VOYAGE_API_KEY env var automatically
        )
    
    else:
        supported = ["openai", "huggingface", "gemini", "cohere", "voyage"]
        raise ValueError(
            f"Unknown embedding provider: '{config.provider}'. "
            f"Supported providers: {supported}"
        )
