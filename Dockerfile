# =============================================================================
# Adizon Trooper Worker - Dockerfile
# Compute-intensive microservice for document processing & graph extraction
# =============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies for document processing
# - libmagic-dev: file type detection (unstructured)
# - poppler-utils: PDF rendering (pdf2image)
# - tesseract-ocr: OCR for scanned PDFs
# - tesseract-ocr-deu: German language pack for OCR
# - libpq-dev: PostgreSQL client library
# - build-essential: compilation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-deu \
    libpq-dev \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
