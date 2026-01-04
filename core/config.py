"""
Trooper Worker configuration using Pydantic Settings.
Loads environment variables from .env file.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Zentrale Konstante für Vector Collection - MUSS überall gleich sein!
VECTOR_COLLECTION_NAME = "adizon_knowledge_base"


class Settings(BaseSettings):
    """Trooper Worker settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # PostgreSQL + pgvector
    # -------------------------------------------------------------------------
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5433, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="knowledge_core", alias="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")

    # -------------------------------------------------------------------------
    # Neo4j
    # -------------------------------------------------------------------------
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")

    # -------------------------------------------------------------------------
    # MinIO / S3
    # -------------------------------------------------------------------------
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket_name: str = Field(default="knowledge-documents", alias="MINIO_BUCKET_NAME")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # -------------------------------------------------------------------------
    # AI API (OpenAI-compatible endpoint, e.g., Trooper)
    # -------------------------------------------------------------------------
    embedding_api_url: str = Field(..., alias="EMBEDDING_API_URL")
    embedding_api_key: str = Field(..., alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(
        default="jina/jina-embeddings-v2-base-de",
        alias="EMBEDDING_MODEL",
    )

    # -------------------------------------------------------------------------
    # LLM Model (for graph extraction)
    # -------------------------------------------------------------------------
    llm_model_name: str = Field(
        default="adizon-ministral",
        alias="LLM_MODEL_NAME",
    )

    # -------------------------------------------------------------------------
    # Ontology Configuration (Multi-Tenant Support)
    # -------------------------------------------------------------------------
    ontology_path: str = Field(
        default="config/ontology_voltage.yaml",
        alias="ONTOLOGY_PATH",
    )

    # -------------------------------------------------------------------------
    # Backend Callback URL (to update document status)
    # -------------------------------------------------------------------------
    backend_url: str = Field(
        default="http://localhost:8000",
        alias="BACKEND_URL",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache."""
    get_settings.cache_clear()
