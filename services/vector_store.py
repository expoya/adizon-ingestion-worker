"""
Vector Store Service using PGVector for document embeddings.

Stores document chunks with their embeddings in PostgreSQL using pgvector.

Supports both:
- Config-based initialization (multi-tenant)
- Legacy singleton mode (backwards compatibility)
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from core.config import (
    VECTOR_COLLECTION_NAME,
    EmbeddingConfig,
    PostgresConfig,
    get_settings,
)

logger = logging.getLogger(__name__)

# Thread pool for running blocking operations
_executor = ThreadPoolExecutor(max_workers=2)


class VectorStoreService:
    """
    Service for storing document embeddings using PGVector.
    """

    def __init__(self, postgres_config: PostgresConfig, embedding_config: EmbeddingConfig):
        """
        Initialize the vector store with explicit configuration.

        Args:
            postgres_config: PostgreSQL connection configuration
            embedding_config: Embedding API configuration
        """
        self.postgres_config = postgres_config
        self.embedding_config = embedding_config

        if not embedding_config.api_key:
            raise ValueError("Embedding API key is required for vector store")

        self.embeddings = OpenAIEmbeddings(
            openai_api_base=embedding_config.api_url,
            openai_api_key=embedding_config.api_key,
            model=embedding_config.model,
            check_embedding_ctx_length=False,
        )

        connection_string = (
            f"postgresql+psycopg://{postgres_config.user}:{postgres_config.password}"
            f"@{postgres_config.host}:{postgres_config.port}/{postgres_config.database}"
            f"?connect_timeout=60&options=-c%20statement_timeout%3D300000"
        )

        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=VECTOR_COLLECTION_NAME,
            connection=connection_string,
            use_jsonb=True,
        )

        logger.info(f"VectorStoreService initialized with collection: {VECTOR_COLLECTION_NAME}")

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous function in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, partial(func, *args, **kwargs)
        )

    async def add_documents(
        self,
        chunks: List[Document],
        document_id: str,
        batch_size: int = 50,
    ) -> List[str]:
        """
        Add document chunks to the vector store with robust error handling.

        Processes chunks in batches. If a batch fails (e.g., NaN error from Jina),
        falls back to processing chunks individually, skipping problematic ones.
        """
        for chunk in chunks:
            chunk.metadata["document_id"] = document_id

        all_ids = []
        failed_chunks = 0

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            try:
                ids = await self._run_sync(
                    self.vector_store.add_documents,
                    batch,
                )
                all_ids.extend(ids)
            except Exception as batch_error:
                # Batch failed - try individual chunks
                logger.warning(f"Batch {i//batch_size + 1} failed: {batch_error}. Processing individually...")

                for j, chunk in enumerate(batch):
                    try:
                        ids = await self._run_sync(
                            self.vector_store.add_documents,
                            [chunk],
                        )
                        all_ids.extend(ids)
                    except Exception as chunk_error:
                        failed_chunks += 1
                        # Log but don't fail - skip problematic chunk
                        content_preview = chunk.page_content[:50].replace('\n', ' ') if chunk.page_content else "<empty>"
                        logger.warning(f"Skipping chunk {i+j+1}: {content_preview}... Error: {chunk_error}")

        if failed_chunks > 0:
            logger.warning(f"Completed with {failed_chunks} failed chunks (skipped)")
            print(f"   ⚠️ {failed_chunks} chunks failed to embed (skipped)")

        return all_ids


def create_vector_store_service(
    postgres_config: PostgresConfig,
    embedding_config: EmbeddingConfig,
) -> VectorStoreService:
    """Create a new vector store service instance with the given configs."""
    return VectorStoreService(postgres_config, embedding_config)


# =============================================================================
# Legacy singleton support (for backwards compatibility)
# =============================================================================
_vector_store_service: VectorStoreService | None = None


def get_vector_store_service() -> VectorStoreService:
    """
    Get or create vector store service singleton using legacy .env settings.

    DEPRECATED: Use create_vector_store_service(postgres_config, embedding_config) for multi-tenant support.
    """
    global _vector_store_service
    if _vector_store_service is None:
        settings = get_settings()
        postgres_config = PostgresConfig(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )
        embedding_config = EmbeddingConfig(
            api_url=settings.embedding_api_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            llm_model=settings.llm_model_name,
        )
        _vector_store_service = VectorStoreService(postgres_config, embedding_config)
    return _vector_store_service
