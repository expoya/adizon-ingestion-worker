"""
Adizon Trooper Worker - Compute-Intensive Microservice

This worker handles:
- Document processing (PDF, DOCX, TXT with OCR support)
- Graph extraction via LLM
- Vector embedding generation
- Neo4j graph storage

Designed to run on GPU-enabled infrastructure (Trooper server).
"""

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field

from workflow import run_ingestion_workflow

app = FastAPI(
    title="Adizon Trooper Worker",
    description="Compute-intensive microservice for document processing and graph extraction",
    version="1.0.0",
)


class IngestRequest(BaseModel):
    """Request model for document ingestion tasks."""

    document_id: str = Field(..., description="UUID of the document to process")
    filename: str = Field(..., description="Original filename of the document")
    storage_path: str = Field(..., description="Path to the document in MinIO/S3")


class IngestResponse(BaseModel):
    """Response model for ingestion task acceptance."""

    status: str = Field(..., description="Task status")
    document_id: str = Field(..., description="Document ID being processed")
    message: str = Field(..., description="Status message")


async def process_document(document_id: str, storage_path: str, filename: str):
    """
    Background task to process a document.

    This runs the full ingestion workflow asynchronously.
    """
    try:
        await run_ingestion_workflow(
            document_id=document_id,
            storage_path=storage_path,
            filename=filename,
        )
    except Exception as e:
        print(f"Error processing document {filename}: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "trooper-worker"}


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Adizon Trooper Worker",
        "version": "1.0.0",
        "description": "Compute-intensive microservice for document processing",
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Accept a document ingestion task.

    This endpoint receives document processing requests from the main backend
    and immediately returns "accepted". The actual processing happens
    asynchronously in the background.

    Args:
        request: IngestRequest containing document_id, filename, and storage_path
        background_tasks: FastAPI BackgroundTasks for async processing

    Returns:
        IngestResponse with task acceptance status
    """
    print(f"\n{'='*60}")
    print(f"📥 Received ingestion task: {request.filename}")
    print(f"   - Document ID: {request.document_id}")
    print(f"   - Storage path: {request.storage_path}")
    print(f"{'='*60}\n")

    # Add the processing task to background tasks
    background_tasks.add_task(
        process_document,
        document_id=request.document_id,
        storage_path=request.storage_path,
        filename=request.filename,
    )

    return IngestResponse(
        status="accepted",
        document_id=request.document_id,
        message=f"Task accepted for background processing: {request.filename}",
    )
