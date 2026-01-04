"""
Vector Store Service using PGVector for document embeddings.

Stores document chunks with their embeddings in PostgreSQL using pgvector.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from core.config import VECTOR_COLLECTION_NAME, get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Thread pool for running blocking operations
_executor = ThreadPoolExecutor(max_workers=2)


class VectorStoreService:
    """
    Service for storing document embeddings using PGVector.
    """

    def __init__(self):
        """Initialize the vector store with configured embeddings."""
        if not settings.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY is required for vector store")

        self.embeddings = OpenAIEmbeddings(
            openai_api_base=settings.embedding_api_url,
            openai_api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            check_embedding_ctx_length=False,
        )

        connection_string = (
            f"postgresql+psycopg://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
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
    ) -> List[str]:
        """Add document chunks to the vector store."""
        for chunk in chunks:
            chunk.metadata["document_id"] = document_id

        ids = await self._run_sync(
            self.vector_store.add_documents,
            chunks,
        )

        return ids


# Singleton instance
_vector_store_service: VectorStoreService | None = None


def get_vector_store_service() -> VectorStoreService:
    """Get or create vector store service singleton."""
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service
