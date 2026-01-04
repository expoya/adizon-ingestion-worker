# Changelog

All notable changes to the Adizon Ingestion Worker project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2025-01-04

### Added

#### Enhanced OCR Support
- **Unstructured Library Integration**: Upgraded document processing with `unstructured[all-docs]>=0.16.0`
  - Better PDF parsing with elements mode
  - Improved text extraction quality
  - Enhanced table and layout recognition
- **Python Magic**: Added `python-magic>=0.4.27` for better file type detection

#### Dynamic Callback System
- **Flexible Webhook URLs**: Callback URLs are now provided per request instead of hardcoded
  - New `callback_url` field in `IngestRequest` model
  - Callback URL flows through entire ingestion pipeline
  - Enables multi-tenant deployment with different backends
  
#### Neo4j Stability Improvements
- **Property Sanitization**: Automatic conversion of nested metadata structures
  - New `_sanitize_props()` method in `GraphStoreService`
  - Converts nested dicts/lists to JSON strings
  - Prevents Neo4j crashes with complex metadata

### Changed

- **PDF Processing**: Switched from `PyPDFLoader` to `UnstructuredFileLoader`
  - More robust text extraction
  - Better handling of complex layouts
  - Strategy: "fast" mode for optimal performance
  
- **Callback Mechanism**: Removed hardcoded `BACKEND_URL` dependency
  - `finalize_node()` now uses dynamic callback URL from state
  - More flexible for different deployment scenarios
  
- **API Contract**: `IngestRequest` now requires `callback_url` parameter
  - **Breaking Change**: Existing clients must provide callback URL
  - Example: `{"document_id": "...", "filename": "...", "storage_path": "...", "callback_url": "http://backend/api/v1/documents/{id}/status"}`

### Fixed

- Neo4j crashes when entities/relationships contain nested metadata structures
- Improved error logging for callback failures

### Technical Details

#### Modified Files
- `requirements.txt`: Added new dependencies
- `main.py`: Extended API to accept callback URLs
- `workflow.py`: Integrated Unstructured loader and dynamic callbacks
- `services/graph_store.py`: Added property sanitization

#### Migration Notes

**For Backend Services:**
When calling the ingestion worker, now include the callback URL:

```python
# Old (v1.0.0)
payload = {
    "document_id": doc_id,
    "filename": filename,
    "storage_path": path
}

# New (v1.1.0)
payload = {
    "document_id": doc_id,
    "filename": filename,
    "storage_path": path,
    "callback_url": f"http://backend:8000/api/v1/documents/{doc_id}/status"  # NEW
}
```

**Docker Compose:**
The `BACKEND_URL` environment variable is no longer used by the worker. You can remove it from your deployment configuration.

## [1.0.0] - 2025-01-04

### Added

- Initial release of Adizon Ingestion Worker
- Document processing (PDF, DOCX, TXT)
- OCR support with Tesseract
- Vector embeddings with Jina German models
- Knowledge graph extraction via LLM
- Neo4j storage with PENDING/APPROVED workflow
- MinIO/S3 integration for document storage
- PostgreSQL with pgvector for embeddings
- FastAPI REST API with background task processing
- Health check endpoint
- Docker containerization with GPU support

### Features

- **Document Processing Pipeline**:
  - PDF extraction with PyPDFLoader
  - DOCX extraction with Docx2txt
  - TXT/MD file support
  - Text sanitization for PostgreSQL compatibility
  
- **Vector Store**:
  - OpenAI-compatible embeddings API
  - PGVector storage
  - Chunk-based document indexing
  
- **Knowledge Graph**:
  - Dynamic ontology support via YAML
  - Entity extraction with structured output
  - Relationship extraction
  - PENDING status for review workflow
  
- **Microservice Architecture**:
  - Async processing with background tasks
  - Status callbacks to main backend
  - Isolated compute-intensive operations

[Unreleased]: https://github.com/expoya/adizon-ingestion-worker/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/expoya/adizon-ingestion-worker/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/expoya/adizon-ingestion-worker/releases/tag/v1.0.0

