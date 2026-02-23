FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY doc_server/ ./doc_server/

# Create data directories
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DOC_SERVER_MODE=remote
ENV DOC_SERVER_STORAGE_PATH=/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run backend server
CMD ["python", "-m", "uvicorn", "doc_server.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
