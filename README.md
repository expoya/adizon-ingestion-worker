# Adizon Trooper Worker

Compute-intensive microservice for document processing and knowledge graph extraction.

## Overview

The Trooper Worker is a standalone service that handles heavy processing tasks, designed to run on GPU-enabled infrastructure:

- **Document Processing**: PDF, DOCX, TXT with OCR support (Tesseract)
- **Vector Embeddings**: Generates embeddings using Jina German models
- **Graph Extraction**: Extracts entities and relationships via LLM
- **Neo4j Storage**: Stores knowledge graphs with review workflow (PENDING/APPROVED states)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Adizon Knowledge Core                       │
│                     (Main Backend)                           │
│                                                              │
│  1. Upload document                                          │
│  2. Store metadata in PostgreSQL                            │
│  3. Store file in MinIO                                     │
│  4. Call Trooper Worker /ingest endpoint                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Trooper Worker                            │
│                                                              │
│  5. Download file from MinIO                                │
│  6. Extract text (PDF/DOCX/OCR)                             │
│  7. Split into chunks                                        │
│  8. Generate embeddings → PGVector                          │
│  9. Extract entities → Neo4j (PENDING status)               │
│  10. Callback to backend with status (INDEXED/ERROR)        │
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

Request body:
```json
{
  "document_id": "uuid-string",
  "filename": "document.pdf",
  "storage_path": "documents/uuid/document.pdf"
}
```

Response:
```json
{
  "status": "accepted",
  "document_id": "uuid-string",
  "message": "Task accepted for background processing: document.pdf"
}
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_HOST` | PostgreSQL host | `postgres` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://neo4j:7687` |
| `MINIO_ENDPOINT` | MinIO endpoint | `minio:9000` |
| `EMBEDDING_API_URL` | OpenAI-compatible API URL | - |
| `EMBEDDING_MODEL` | Embedding model name | `jina/jina-embeddings-v2-base-de` |
| `LLM_MODEL_NAME` | LLM for graph extraction | `adizon-ministral` |
| `ONTOLOGY_PATH` | Path to ontology YAML | `config/ontology_voltage.yaml` |
| `BACKEND_URL` | Callback URL for status updates | `http://adizon-backend:8000` |

## Ontology Configuration

The knowledge graph schema is defined in `config/ontology_voltage.yaml`. This allows multi-tenant support with different domain-specific ontologies.

## Caddy Reverse Proxy

For Caddy integration, see `docs/Caddyfile_snippet.txt`.

## Project Structure

```
Adizon-trooper/
├── main.py              # FastAPI application entry
├── workflow.py          # LangGraph ingestion workflow
├── Dockerfile           # Container build configuration
├── docker-compose.yml   # Container orchestration
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── config/
│   └── ontology_voltage.yaml  # Knowledge graph schema
├── core/
│   └── config.py        # Pydantic settings
├── services/
│   ├── graph_store.py   # Neo4j service
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

## License

Proprietary - Adizon GmbH

