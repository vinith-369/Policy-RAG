"""
Application configuration — reads from .env file.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama Model
    ollama_model: str = "qwen2.5:7b"

    # Qdrant
    collection_name: str = "policy_docs"

    # Embedding Models
    dense_model: str = "all-MiniLM-L6-v2"
    dense_vector_size: int = 384

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
