"""
Adizon Trooper Worker - Compute-Intensive Microservice

This worker handles:
- Document processing (PDF, DOCX, TXT, PPTX, XLSX, etc. with OCR support)
- Graph extraction via LLM
- Vector embedding generation
- Neo4j graph storage

Designed to run on GPU-enabled infrastructure (Trooper server).
Supports multi-tenant mode with per-request connection configurations.
"""

import logging
import traceback
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from core.config import (
    ConnectionConfig,
    EmbeddingConfig,
    MinioConfig,
    Neo4jConfig,
    PostgresConfig,
    get_settings,
)
from workflow import run_ingestion_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trooper-worker")

app = FastAPI(
    title="Adizon Trooper Worker",
    description="Compute-intensive microservice for document processing and graph extraction",
    version="2.0.0",
)


# =============================================================================
# Request/Response Models with Validation
# =============================================================================

class IngestRequest(BaseModel):
    """Request model for document ingestion tasks (multi-tenant mode)."""

    document_id: str = Field(..., description="UUID of the document to process")
    filename: str = Field(..., description="Original filename of the document")
    storage_path: str = Field(..., description="Path to the document in MinIO/S3")
    callback_url: str = Field(..., description="Webhook URL for status updates")

    # Connection configurations (required for multi-tenant mode)
    minio: MinioConfig = Field(..., description="MinIO connection configuration")
    postgres: PostgresConfig = Field(..., description="PostgreSQL connection configuration")
    neo4j: Neo4jConfig = Field(..., description="Neo4j connection configuration")
    embedding: EmbeddingConfig = Field(..., description="Embedding API configuration")

    # Optional: Ontology content (base64 encoded YAML)
    ontology_content: Optional[str] = Field(
        default=None,
        description="Base64-encoded ontology YAML content for graph extraction"
    )

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("document_id cannot be empty")
        return v.strip()

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("filename cannot be empty")
        return v.strip()

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("storage_path cannot be empty")
        return v.strip()

    @field_validator("callback_url")
    @classmethod
    def validate_callback_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("callback_url cannot be empty")
        if not v.startswith(("http://", "https://")):
            raise ValueError("callback_url must start with http:// or https://")
        return v.strip()


class IngestRequestLegacy(BaseModel):
    """Legacy request model for backwards compatibility (single-tenant mode)."""

    document_id: str = Field(..., description="UUID of the document to process")
    filename: str = Field(..., description="Original filename of the document")
    storage_path: str = Field(..., description="Path to the document in MinIO/S3")
    callback_url: str = Field(..., description="Webhook URL for status updates")


class IngestResponse(BaseModel):
    """Response model for ingestion task acceptance."""

    status: str = Field(..., description="Task status")
    document_id: str = Field(..., description="Document ID being processed")
    message: str = Field(..., description="Status message")


class ConnectionTestResult(BaseModel):
    """Result of a connection test."""
    service: str
    status: str
    message: str
    endpoint: Optional[str] = None


class ConnectionTestResponse(BaseModel):
    """Response for connection test endpoint."""
    overall_status: str
    results: list[ConnectionTestResult]


# =============================================================================
# Connection Testing Functions
# =============================================================================

def test_minio_connection(config: MinioConfig) -> ConnectionTestResult:
    """Test MinIO connection."""
    try:
        from services.storage import create_minio_service
        service = create_minio_service(config)
        # Try to list buckets (simple connectivity test)
        service.client.head_bucket(Bucket=config.bucket_name)
        return ConnectionTestResult(
            service="MinIO",
            status="ok",
            message=f"Connected to bucket '{config.bucket_name}'",
            endpoint=config.endpoint,
        )
    except Exception as e:
        return ConnectionTestResult(
            service="MinIO",
            status="error",
            message=f"Connection failed: {str(e)}",
            endpoint=config.endpoint,
        )


def test_postgres_connection(config: PostgresConfig) -> ConnectionTestResult:
    """Test PostgreSQL connection."""
    try:
        import psycopg
        conn_str = (
            f"host={config.host} port={config.port} dbname={config.database} "
            f"user={config.user} password={config.password} connect_timeout=10"
        )
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return ConnectionTestResult(
            service="PostgreSQL",
            status="ok",
            message=f"Connected to database '{config.database}'",
            endpoint=f"{config.host}:{config.port}",
        )
    except Exception as e:
        return ConnectionTestResult(
            service="PostgreSQL",
            status="error",
            message=f"Connection failed: {str(e)}",
            endpoint=f"{config.host}:{config.port}",
        )


def test_neo4j_connection(config: Neo4jConfig) -> ConnectionTestResult:
    """Test Neo4j connection."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
        )
        driver.verify_connectivity()
        driver.close()
        return ConnectionTestResult(
            service="Neo4j",
            status="ok",
            message="Connected successfully",
            endpoint=config.uri,
        )
    except Exception as e:
        return ConnectionTestResult(
            service="Neo4j",
            status="error",
            message=f"Connection failed: {str(e)}",
            endpoint=config.uri,
        )


def test_embedding_api(config: EmbeddingConfig) -> ConnectionTestResult:
    """Test Embedding API connection."""
    try:
        import httpx
        # Test with a simple request to the API base
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{config.api_url}/models")
            if response.status_code in [200, 401, 403]:
                # 401/403 means API is reachable but auth might be different
                return ConnectionTestResult(
                    service="Embedding API",
                    status="ok",
                    message=f"API reachable (status: {response.status_code})",
                    endpoint=config.api_url,
                )
            else:
                return ConnectionTestResult(
                    service="Embedding API",
                    status="warning",
                    message=f"API returned status {response.status_code}",
                    endpoint=config.api_url,
                )
    except Exception as e:
        return ConnectionTestResult(
            service="Embedding API",
            status="error",
            message=f"Connection failed: {str(e)}",
            endpoint=config.api_url,
        )


# =============================================================================
# Logging Helpers
# =============================================================================

def log_config_summary(request: IngestRequest) -> None:
    """Log a summary of the configuration (without secrets)."""
    logger.info("=" * 60)
    logger.info(f"Received ingestion task: {request.filename}")
    logger.info(f"  Document ID: {request.document_id}")
    logger.info(f"  Storage path: {request.storage_path}")
    logger.info(f"  Callback URL: {request.callback_url}")
    logger.info("Connection Configuration:")
    logger.info(f"  MinIO:")
    logger.info(f"    - Endpoint: {request.minio.endpoint}")
    logger.info(f"    - Bucket: {request.minio.bucket_name}")
    logger.info(f"    - Secure: {request.minio.secure}")
    logger.info(f"    - Access Key: {mask_secret(request.minio.access_key)}")
    logger.info(f"  PostgreSQL:")
    logger.info(f"    - Host: {request.postgres.host}:{request.postgres.port}")
    logger.info(f"    - Database: {request.postgres.database}")
    logger.info(f"    - User: {request.postgres.user}")
    logger.info(f"    - Password: {mask_secret(request.postgres.password)}")
    logger.info(f"  Neo4j:")
    logger.info(f"    - URI: {request.neo4j.uri}")
    logger.info(f"    - User: {request.neo4j.user}")
    logger.info(f"    - Password: {mask_secret(request.neo4j.password)}")
    logger.info(f"  Embedding API:")
    logger.info(f"    - URL: {request.embedding.api_url}")
    logger.info(f"    - Model: {request.embedding.model}")
    logger.info(f"    - LLM Model: {request.embedding.llm_model}")
    logger.info(f"    - API Key: {mask_secret(request.embedding.api_key)}")
    logger.info(f"  Ontology: {'provided' if request.ontology_content else 'not provided'}")
    logger.info("=" * 60)


def mask_secret(secret: str, visible_chars: int = 4) -> str:
    """Mask a secret, showing only first few characters."""
    if not secret:
        return "<empty>"
    if len(secret) <= visible_chars:
        return "*" * len(secret)
    return secret[:visible_chars] + "*" * (len(secret) - visible_chars)


def validate_config_completeness(request: IngestRequest) -> list[str]:
    """Validate that all required config fields are present and non-empty."""
    errors = []

    # MinIO validation
    if not request.minio.endpoint:
        errors.append("minio.endpoint is empty")
    if not request.minio.access_key:
        errors.append("minio.access_key is empty")
    if not request.minio.secret_key:
        errors.append("minio.secret_key is empty")
    if not request.minio.bucket_name:
        errors.append("minio.bucket_name is empty")

    # PostgreSQL validation
    if not request.postgres.host:
        errors.append("postgres.host is empty")
    if not request.postgres.database:
        errors.append("postgres.database is empty")
    if not request.postgres.user:
        errors.append("postgres.user is empty")
    if not request.postgres.password:
        errors.append("postgres.password is empty")

    # Neo4j validation
    if not request.neo4j.uri:
        errors.append("neo4j.uri is empty")
    if not request.neo4j.user:
        errors.append("neo4j.user is empty")
    if not request.neo4j.password:
        errors.append("neo4j.password is empty")

    # Embedding API validation
    if not request.embedding.api_url:
        errors.append("embedding.api_url is empty")
    if not request.embedding.api_key:
        errors.append("embedding.api_key is empty")

    return errors


# =============================================================================
# Background Task Processors
# =============================================================================

async def process_document(
    document_id: str,
    storage_path: str,
    filename: str,
    callback_url: str,
    connection_config: ConnectionConfig,
):
    """
    Background task to process a document.

    This runs the full ingestion workflow asynchronously.
    """
    try:
        logger.info(f"Starting background processing for: {filename}")
        await run_ingestion_workflow(
            document_id=document_id,
            storage_path=storage_path,
            filename=filename,
            callback_url=callback_url,
            connection_config=connection_config,
        )
        logger.info(f"Completed processing for: {filename}")
    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}")
        logger.error(traceback.format_exc())


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "trooper-worker", "version": "2.0.0"}


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Adizon Trooper Worker",
        "version": "2.0.0",
        "description": "Compute-intensive microservice for document processing",
        "features": [
            "Multi-tenant support with per-request connection configs",
            "PDF, DOCX, PPTX, XLSX, TXT, and more via UnstructuredLoader",
            "Vector embeddings with PGVector",
            "Knowledge graph extraction with Neo4j",
        ],
        "endpoints": {
            "/ingest": "POST - Submit document for processing (multi-tenant)",
            "/ingest/legacy": "POST - Submit document for processing (legacy .env mode)",
            "/test-connections": "POST - Test all connection configs before processing",
            "/health": "GET - Health check",
        },
    }


@app.post("/test-connections", response_model=ConnectionTestResponse)
async def test_connections(request: IngestRequest):
    """
    Test all connection configurations before submitting a document.

    This endpoint validates that all services (MinIO, PostgreSQL, Neo4j, Embedding API)
    are reachable with the provided configurations. Use this to debug connection issues.

    Returns:
        ConnectionTestResponse with individual test results for each service
    """
    logger.info(f"Testing connections for request from: {request.callback_url}")

    results = [
        test_minio_connection(request.minio),
        test_postgres_connection(request.postgres),
        test_neo4j_connection(request.neo4j),
        test_embedding_api(request.embedding),
    ]

    # Determine overall status
    has_errors = any(r.status == "error" for r in results)
    has_warnings = any(r.status == "warning" for r in results)

    if has_errors:
        overall_status = "error"
    elif has_warnings:
        overall_status = "warning"
    else:
        overall_status = "ok"

    # Log results
    for result in results:
        if result.status == "ok":
            logger.info(f"  {result.service}: {result.message}")
        elif result.status == "warning":
            logger.warning(f"  {result.service}: {result.message}")
        else:
            logger.error(f"  {result.service}: {result.message}")

    return ConnectionTestResponse(
        overall_status=overall_status,
        results=results,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Accept a document ingestion task (multi-tenant mode).

    This endpoint receives document processing requests with full connection
    configurations and immediately returns "accepted". The actual processing
    happens asynchronously in the background.

    Args:
        request: IngestRequest containing document info and connection configs
        background_tasks: FastAPI BackgroundTasks for async processing

    Returns:
        IngestResponse with task acceptance status

    Raises:
        HTTPException 400: If configuration validation fails
    """
    # Log configuration summary
    log_config_summary(request)

    # Validate configuration completeness
    config_errors = validate_config_completeness(request)
    if config_errors:
        error_msg = "Configuration validation failed: " + "; ".join(config_errors)
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # Build connection config from request
    connection_config = ConnectionConfig(
        minio=request.minio,
        postgres=request.postgres,
        neo4j=request.neo4j,
        embedding=request.embedding,
        ontology_content=request.ontology_content,
    )

    # Add the processing task to background tasks
    background_tasks.add_task(
        process_document,
        document_id=request.document_id,
        storage_path=request.storage_path,
        filename=request.filename,
        callback_url=request.callback_url,
        connection_config=connection_config,
    )

    logger.info(f"Task accepted for background processing: {request.filename}")

    return IngestResponse(
        status="accepted",
        document_id=request.document_id,
        message=f"Task accepted for background processing: {request.filename}",
    )


@app.post("/ingest/legacy", response_model=IngestResponse)
async def ingest_document_legacy(request: IngestRequestLegacy, background_tasks: BackgroundTasks):
    """
    Accept a document ingestion task (legacy single-tenant mode).

    DEPRECATED: Use /ingest with full connection configs for multi-tenant support.

    This endpoint uses connection settings from environment variables (.env file).

    Args:
        request: IngestRequestLegacy containing document info only
        background_tasks: FastAPI BackgroundTasks for async processing

    Returns:
        IngestResponse with task acceptance status
    """
    logger.info("=" * 60)
    logger.info(f"[LEGACY] Received ingestion task: {request.filename}")
    logger.info(f"  Document ID: {request.document_id}")
    logger.info(f"  Storage path: {request.storage_path}")
    logger.info(f"  Callback URL: {request.callback_url}")
    logger.info("  Using .env configuration")
    logger.info("=" * 60)

    # Build connection config from legacy settings
    try:
        settings = get_settings()
        connection_config = settings.to_connection_config()
    except Exception as e:
        error_msg = f"Failed to load .env configuration: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # Add the processing task to background tasks
    background_tasks.add_task(
        process_document,
        document_id=request.document_id,
        storage_path=request.storage_path,
        filename=request.filename,
        callback_url=request.callback_url,
        connection_config=connection_config,
    )

    logger.info(f"[LEGACY] Task accepted for background processing: {request.filename}")

    return IngestResponse(
        status="accepted",
        document_id=request.document_id,
        message=f"[LEGACY] Task accepted for background processing: {request.filename}",
    )
