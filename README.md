# Adizon Ingestion Worker

Compute-intensive microservice for document processing and knowledge graph extraction.

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/)

## Overview

The Ingestion Worker is a standalone service that handles heavy processing tasks, designed to run on GPU-enabled infrastructure:

- **Enhanced Document Processing**: PDF, DOCX, TXT with advanced OCR support
  - Powered by Unstructured library for superior PDF parsing
  - Tesseract OCR with German language support
  - Automatic text sanitization for database compatibility
- **Vector Embeddings**: Generates embeddings using Jina German models
- **Graph Extraction**: Extracts entities and relationships via LLM with dynamic ontology
- **Neo4j Storage**: Stores knowledge graphs with review workflow (PENDING/APPROVED states)
- **Dynamic Callbacks**: Flexible webhook system for status updates

## What's New in v1.1.0

🎉 **Enhanced OCR**: Upgraded to Unstructured library for better PDF parsing  
🔗 **Dynamic Callbacks**: Flexible webhook URLs per request  
🛡️ **Neo4j Stability**: Automatic property sanitization prevents crashes  

See [CHANGELOG.md](CHANGELOG.md) for full details.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Backend Service                             │
│              (e.g., Adizon Knowledge Core)                  │
│                                                              │
│  1. Upload document                                          │
│  2. Store metadata in PostgreSQL                            │
│  3. Store file in MinIO                                     │
│  4. Call Ingestion Worker /ingest with callback_url         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Ingestion Worker                           │
│                                                              │
│  5. Download file from MinIO                                │
│  6. Extract text with Unstructured (enhanced OCR)           │
│  7. Split into chunks                                        │
│  8. Generate embeddings → PGVector                          │
│  9. Extract entities → Neo4j (PENDING status)               │
│  10. POST to callback_url with status (INDEXED/ERROR)       │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker & Docker Compose
- External network `my-ai-stack_default` (or configure your own)
- Running services:
  - PostgreSQL with pgvector extension
  - Neo4j
  - MinIO (S3-compatible storage)
  - LLM API (OpenAI-compatible, e.g., local Trooper server)

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 2. Start the Worker

```bash
docker-compose up -d --build
```

### 3. Verify Health

```bash
curl http://localhost:8000/health
# {"status":"healthy","service":"trooper-worker"}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/health` | GET | Health check |
| `/ingest` | POST | Submit document for processing |

### POST /ingest

Submit a document for processing with a callback URL for status updates.

**Request body:**
```json
{
  "document_id": "uuid-string",
  "filename": "document.pdf",
  "storage_path": "documents/uuid/document.pdf",
  "callback_url": "http://backend:8000/api/v1/documents/{doc_id}/status"
}
```

**Response:**
```json
{
  "status": "accepted",
  "document_id": "uuid-string",
  "message": "Task accepted for background processing: document.pdf"
}
```

**Callback Payload** (sent to `callback_url` upon completion):
```json
{
  "status": "INDEXED",
  "error_message": null
}
```

Or in case of error:
```json
{
  "status": "ERROR",
  "error_message": "Failed to load document: ..."
}
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POSTGRES_HOST` | PostgreSQL host | `postgres` | ✅ |
| `POSTGRES_PORT` | PostgreSQL port | `5432` | ✅ |
| `POSTGRES_DB` | Database name | `knowledge_core` | ✅ |
| `POSTGRES_USER` | Database user | `postgres` | ✅ |
| `POSTGRES_PASSWORD` | Database password | - | ✅ |
| `NEO4J_URI` | Neo4j connection URI | `bolt://neo4j:7687` | ✅ |
| `NEO4J_USER` | Neo4j user | `neo4j` | ✅ |
| `NEO4J_PASSWORD` | Neo4j password | - | ✅ |
| `MINIO_ENDPOINT` | MinIO endpoint | `minio:9000` | ✅ |
| `MINIO_ACCESS_KEY` | MinIO access key | - | ✅ |
| `MINIO_SECRET_KEY` | MinIO secret key | - | ✅ |
| `MINIO_BUCKET_NAME` | S3 bucket name | `knowledge-documents` | ✅ |
| `EMBEDDING_API_URL` | OpenAI-compatible API URL | - | ✅ |
| `EMBEDDING_API_KEY` | API key for embeddings | - | ✅ |
| `EMBEDDING_MODEL` | Embedding model name | `jina/jina-embeddings-v2-base-de` | ✅ |
| `LLM_MODEL_NAME` | LLM for graph extraction | `adizon-ministral` | ✅ |
| `ONTOLOGY_PATH` | Path to ontology YAML | `config/ontology_voltage.yaml` | ✅ |

**Note:** `BACKEND_URL` is no longer used since v1.1.0. Use `callback_url` in the request payload instead.

## Ontology Configuration

The knowledge graph schema is defined in `config/ontology_voltage.yaml`. This allows multi-tenant support with different domain-specific ontologies.

## Caddy Reverse Proxy

For Caddy integration, see `docs/Caddyfile_snippet.txt`.

## Project Structure

```
adizon-ingestion-worker/
├── main.py              # FastAPI application entry
├── workflow.py          # LangGraph ingestion workflow
├── Dockerfile           # Container build configuration
├── docker-compose.yml   # Container orchestration
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── CHANGELOG.md         # Version history
├── README.md            # This file
├── .gitignore           # Git ignore rules
├── config/
│   └── ontology_voltage.yaml  # Knowledge graph schema
├── core/
│   ├── __init__.py
│   └── config.py        # Pydantic settings
├── services/
│   ├── __init__.py
│   ├── graph_store.py   # Neo4j service (with sanitization)
│   ├── schema_factory.py # Dynamic Pydantic models
│   ├── storage.py       # MinIO service
│   └── vector_store.py  # PGVector service
└── docs/
    └── Caddyfile_snippet.txt  # Reverse proxy config
```

## Development

### Local Development (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment
cp .env.example .env
# Edit .env with local development values

# Run
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Submit a document for processing
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "test-123",
    "filename": "test.pdf",
    "storage_path": "documents/test-123/test.pdf",
    "callback_url": "http://backend:8000/api/v1/documents/test-123/status"
  }'
```

## Upgrading

See [CHANGELOG.md](CHANGELOG.md) for version history and migration notes.

### From v1.0.0 to v1.1.0

**Breaking Change:** The `/ingest` endpoint now requires a `callback_url` field.

Update your API calls:
```python
# Add callback_url to your request
payload = {
    "document_id": doc_id,
    "filename": filename,
    "storage_path": path,
    "callback_url": f"http://backend:8000/api/v1/documents/{doc_id}/status"  # NEW
}
```

## Contributing

Please read [CHANGELOG.md](CHANGELOG.md) for details on our release process.

## License

Proprietary - Adizon GmbH

