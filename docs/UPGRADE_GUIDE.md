# Upgrade Guide

This guide helps you upgrade the Adizon Ingestion Worker and adapt your backend services accordingly.

## Table of Contents

- [Upgrading from v1.0.0 to v1.1.0](#upgrading-from-v100-to-v110)
- [Breaking Changes](#breaking-changes)
- [Backend Integration Changes](#backend-integration-changes)
- [Deployment Updates](#deployment-updates)
- [Rollback Instructions](#rollback-instructions)

---

## Upgrading from v1.0.0 to v1.1.0

### Overview

Version 1.1.0 introduces three major improvements:
1. **Enhanced OCR** with Unstructured library
2. **Dynamic callback URLs** instead of hardcoded backend URL
3. **Neo4j stability** through property sanitization

### Breaking Changes

#### 1. `callback_url` Field is Now Required

**What Changed:**
- The `/ingest` endpoint now requires a `callback_url` parameter in the request body
- The `BACKEND_URL` environment variable is no longer used

**Migration Required:** ✅ Yes

**Old Code (v1.0.0):**
```python
import httpx

async def submit_document(doc_id: str, filename: str, storage_path: str):
    payload = {
        "document_id": doc_id,
        "filename": filename,
        "storage_path": storage_path
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://ingestion-worker:8000/ingest",
            json=payload
        )
    return response.json()
```

**New Code (v1.1.0):**
```python
import httpx

async def submit_document(doc_id: str, filename: str, storage_path: str):
    # Construct the callback URL for status updates
    callback_url = f"http://backend:8000/api/v1/documents/{doc_id}/status"
    
    payload = {
        "document_id": doc_id,
        "filename": filename,
        "storage_path": storage_path,
        "callback_url": callback_url  # NEW: Required field
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://ingestion-worker:8000/ingest",
            json=payload
        )
    return response.json()
```

### Backend Integration Changes

#### Update Your Status Update Endpoint

Ensure your backend has an endpoint that matches the callback URL pattern:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class StatusUpdate(BaseModel):
    status: str  # "INDEXED" or "ERROR"
    error_message: str | None = None

@router.post("/api/v1/documents/{document_id}/status")
async def update_document_status(document_id: str, update: StatusUpdate):
    """
    Receives status updates from the ingestion worker.
    """
    # Update document status in your database
    document = await get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document.status = update.status
    if update.error_message:
        document.error_message = update.error_message
    
    await save_document(document)
    
    return {"status": "ok"}
```

#### Multi-Tenant Support

With dynamic callbacks, you can now support multiple backends or tenants:

```python
# Tenant-specific callbacks
tenant_configs = {
    "tenant_a": "https://tenant-a.example.com/api/ingest/callback",
    "tenant_b": "https://tenant-b.example.com/api/ingest/callback"
}

async def submit_for_tenant(tenant_id: str, doc_id: str, filename: str, path: str):
    callback_url = tenant_configs[tenant_id]
    
    payload = {
        "document_id": doc_id,
        "filename": filename,
        "storage_path": path,
        "callback_url": callback_url
    }
    
    # Submit to shared ingestion worker
    await submit_to_worker(payload)
```

### Deployment Updates

#### 1. Update Docker Compose / Environment Variables

**Remove (no longer needed):**
```yaml
# docker-compose.yml or .env
BACKEND_URL=http://backend:8000  # ❌ Remove this
```

**No changes needed for other variables** - all existing configuration remains valid.

#### 2. Rolling Update Strategy

For zero-downtime deployment:

```bash
# Step 1: Deploy v1.1.0 worker (backward compatible for reads)
docker-compose pull
docker-compose up -d ingestion-worker

# Step 2: Update backend to send callback_url
# Deploy your backend changes

# Step 3: Monitor logs
docker-compose logs -f ingestion-worker

# Step 4: Verify callbacks are working
curl http://backend:8000/api/v1/documents/<test-id>
```

#### 3. Kubernetes / Production Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adizon-ingestion-worker
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: worker
        image: your-registry/adizon-ingestion-worker:1.1.0
        env:
        # Remove BACKEND_URL
        # - name: BACKEND_URL
        #   value: "http://backend:8000"
        
        # Keep all other vars
        - name: POSTGRES_HOST
          valueFrom:
            configMapKeyRef:
              name: worker-config
              key: postgres-host
        # ... other env vars ...
```

### Testing the Upgrade

#### 1. Integration Test

```python
import httpx
import pytest

@pytest.mark.asyncio
async def test_ingestion_with_callback():
    """Test v1.1.0 API with callback_url"""
    
    # Mock callback server
    callback_received = []
    
    @app.post("/test-callback")
    async def mock_callback(update: dict):
        callback_received.append(update)
        return {"status": "ok"}
    
    # Submit document
    payload = {
        "document_id": "test-123",
        "filename": "test.pdf",
        "storage_path": "test/test.pdf",
        "callback_url": "http://localhost:8001/test-callback"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://ingestion-worker:8000/ingest",
            json=payload
        )
    
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    
    # Wait for processing
    await asyncio.sleep(10)
    
    # Verify callback was called
    assert len(callback_received) == 1
    assert callback_received[0]["status"] in ["INDEXED", "ERROR"]
```

#### 2. Manual Verification

```bash
# Test with a real document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "manual-test",
    "filename": "sample.pdf",
    "storage_path": "documents/sample.pdf",
    "callback_url": "http://backend:8000/api/v1/documents/manual-test/status"
  }'

# Check worker logs
docker-compose logs -f ingestion-worker | grep "manual-test"

# Verify callback was sent
# Check your backend logs for the status update POST request
```

### Performance & Features

#### Enhanced PDF Processing

The Unstructured library provides better text extraction:

**Benefits:**
- Improved table detection
- Better layout preservation
- Enhanced OCR quality
- Automatic language detection

**Example output comparison:**

```
Old (PyPDFLoader):
"Header\nTable Cell 1 Table Cell 2\nFooter"

New (Unstructured):
"Header\n\n[Table]\n| Cell 1 | Cell 2 |\n\nFooter"
```

#### Neo4j Stability

Properties are now automatically sanitized:

```python
# Before: Could crash Neo4j
entity = {
    "label": "PERSON",
    "name": "John Doe",
    "properties": {
        "metadata": {  # Nested dict - causes crash
            "source": "pdf",
            "page": 1
        }
    }
}

# After: Automatically converted to JSON string
entity_properties = {
    "metadata": '{"source": "pdf", "page": 1}'  # Serialized
}
```

### Rollback Instructions

If you need to rollback to v1.0.0:

```bash
# Step 1: Rollback worker
docker-compose down
docker pull your-registry/adizon-ingestion-worker:1.0.0
docker-compose up -d

# Step 2: Restore BACKEND_URL in environment
echo "BACKEND_URL=http://backend:8000" >> .env

# Step 3: Revert backend code changes
git revert <commit-hash>

# Step 4: Remove callback_url from API calls
# Deploy old backend code
```

### Common Issues

#### Issue: "callback_url field required"

**Cause:** Backend is sending old API format without `callback_url`

**Fix:** Update your backend code to include `callback_url` in the payload

---

#### Issue: Callbacks not being received

**Cause:** Firewall or network issue blocking callback requests

**Fix:**
1. Check network connectivity from worker to backend
2. Verify callback URL is reachable from worker container
3. Check backend logs for incoming POST requests

---

#### Issue: Neo4j errors with complex metadata

**Cause:** Extremely nested structures (more than 3 levels)

**Fix:** This should be automatically handled. If it persists:
1. Check Neo4j logs
2. Verify `_sanitize_props` is being called
3. Report issue with sample document

### Support

For issues or questions:
- Check [CHANGELOG.md](../CHANGELOG.md)
- Review logs: `docker-compose logs ingestion-worker`
- Contact: support@adizon.de

---

**Last Updated:** 2025-01-04  
**Version:** 1.1.0

