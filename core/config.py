"""
Trooper Worker configuration using Pydantic Settings.

Supports two modes:
1. Legacy mode: Load from environment variables (.env file)
2. Multi-tenant mode: Connection configs passed per request
"""

from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Zentrale Konstante für Vector Collection - MUSS überall gleich sein!
VECTOR_COLLECTION_NAME = "adizon_knowledge_base"


# =============================================================================
# Request-Scoped Connection Configs (for multi-tenant support)
# =============================================================================

class MinioConfig(BaseModel):
    """MinIO/S3 connection configuration."""
    endpoint: str = Field(..., description="MinIO endpoint (host:port)")
    access_key: str = Field(..., description="Access key")
    secret_key: str = Field(..., description="Secret key")
    bucket_name: str = Field(..., description="Bucket name")
    secure: bool = Field(default=False, description="Use HTTPS")


class PostgresConfig(BaseModel):
    """PostgreSQL connection configuration."""
    host: str = Field(..., description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")


class Neo4jConfig(BaseModel):
    """Neo4j connection configuration."""
    uri: str = Field(..., description="Neo4j URI (bolt://host:port)")
    user: str = Field(..., description="Neo4j user")
    password: str = Field(..., description="Neo4j password")


class EmbeddingConfig(BaseModel):
    """Embedding API configuration."""
    api_url: str = Field(..., description="OpenAI-compatible API URL")
    api_key: str = Field(..., description="API key")
    model: str = Field(default="jina/jina-embeddings-v2-base-de", description="Embedding model name")
    llm_model: str = Field(default="adizon-ministral", description="LLM model for graph extraction")


class ConnectionConfig(BaseModel):
    """Complete connection configuration for a request."""
    minio: MinioConfig
    postgres: PostgresConfig
    neo4j: Neo4jConfig
    embedding: EmbeddingConfig
    ontology_content: Optional[str] = Field(default=None, description="Ontology YAML content (base64 encoded)")


# =============================================================================
# Legacy Settings (for backwards compatibility / local development)
# =============================================================================

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

    # Local override for embedding URL (used when worker is in same network as Ollama)
    # If set, this overrides the embedding URL from connection_config
    local_embedding_api_url: str | None = Field(default=None, alias="LOCAL_EMBEDDING_API_URL")

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

    def to_connection_config(self) -> ConnectionConfig:
        """Convert legacy settings to ConnectionConfig for backwards compatibility."""
        return ConnectionConfig(
            minio=MinioConfig(
                endpoint=self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                bucket_name=self.minio_bucket_name,
                secure=self.minio_secure,
            ),
            postgres=PostgresConfig(
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
                user=self.postgres_user,
                password=self.postgres_password,
            ),
            neo4j=Neo4jConfig(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
            ),
            embedding=EmbeddingConfig(
                api_url=self.embedding_api_url,
                api_key=self.embedding_api_key,
                model=self.embedding_model,
                llm_model=self.llm_model_name,
            ),
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache."""
    get_settings.cache_clear()
