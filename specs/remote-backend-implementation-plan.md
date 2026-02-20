# Remote Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a Remote Backend architecture that splits Doc Server into client and server components, enabling team-wide shared documentation indexing with REST API communication.

**Architecture:** The system will have two modes - local (existing behavior) and remote (client communicates via REST API to backend server). Backend will use FastAPI with API Key authentication. Configuration will control which mode is active.

**Tech Stack:** FastAPI, httpx, uvicorn, Pydantic, existing Doc Server core services (GitCloner, DocumentProcessor, VectorStore, HybridSearch)

---

## Task 1: Update Configuration (config.py)

**Files:**
- Modify: `config.py:1-50`

**Step 1: Read current config.py to understand structure**

```bash
# Run: Read config.py first 80 lines
```

**Step 2: Add remote backend settings**

Add these fields to the Settings class:
```python
# Mode configuration
mode: Literal["local", "remote"] = "local"

# Remote backend configuration
backend_url: str = "http://localhost:8000"
backend_api_key: str = ""
backend_timeout: int = 30
backend_verify_ssl: bool = True
```

**Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add remote backend configuration settings"
```

---

## Task 2: Create API Client Module (api_client.py)

**Files:**
- Create: `doc_server/api_client.py`
- Test: `tests/test_api_client.py`

**Step 1: Write failing test**

```python
# tests/test_api_client.py
import pytest
from doc_server.api_client import APIClient
from httpx import HTTPStatusError

def test_client_initialization():
    client = APIClient(base_url="http://localhost:8000", api_key="test-key")
    assert client.base_url == "http://localhost:8000"
    assert client.api_key == "test-key"

@pytest.mark.asyncio
async def test_search_forwarded_to_backend():
    client = APIClient(base_url="http://localhost:8000", api_key="test-key")
    # Mock response would go here
    result = await client.search("pandas", "/pandas", limit=10)
    assert isinstance(result, list)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_api_client.py -v
# Expected: FAIL - module not found
```

**Step 3: Write minimal implementation**

```python
# doc_server/api_client.py
"""API Client for communicating with remote Doc Server backend."""

from typing import Any
import httpx
from pydantic import BaseModel


class SearchResult(BaseModel):
    content: str
    file_path: str
    score: float
    metadata: dict[str, Any]


class LibraryInfo(BaseModel):
    library_id: str
    version: str | None = None
    document_count: int
    created_at: float | None = None


class IngestResult(BaseModel bool
    library_id: str
):
    success:    version: str | None = None
    documents_ingested: int
    status: str


class HealthResult(BaseModel):
    status: str
    components: dict[str, str]
    timestamp: float | None = None


class APIClient:
    """Client for remote Doc Server backend communication."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "",
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search(
        self,
        query: str,
        library_id: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search through ingested documentation."""
        client = await self._get_client()
        response = await client.post(
            "/api/v1/search",
            json={"query": query, "library_id": library_id, "limit": limit},
            headers=self._get_headers(),
        )
        response.raise_for_status()
        data = response.json()
        return [SearchResult(**item) for item in data]

    async def ingest(
        self,
        source: str,
        library_id: str,
        version: str | None = None,
        batch_size: int = 32,
    ) -> IngestResult:
        """Trigger ingestion on the backend."""
        client = await self._get_client()
        payload: dict[str, Any] = {
            "source": source,
            "library_id": library_id,
            "batch_size": batch_size,
        }
        if version is not None:
            payload["version"] = version
        response = await client.post(
            "/api/v1/ingest",
            json=payload,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return IngestResult(**response.json())

    async def list_libraries(self) -> list[LibraryInfo]:
        """List all available libraries."""
        client = await self._get_client()
        response = await client.get(
            "/api/v1/libraries",
            headers=self._get_headers(),
        )
        response.raise_for_status()
        data = response.json()
        return [LibraryInfo(**item) for item in data]

    async def remove_library(self, library_id: str) -> bool:
        """Remove a library and its documents."""
        client = await self._get_client()
        response = await client.delete(
            f"/api/v1/libraries/{library_id}",
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> HealthResult:
        """Check backend health status."""
        client = await self._get_client()
        response = await client.get("/api/v1/health")
        response.raise_for_status()
        return HealthResult(**response.json())

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_api_client.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add doc_server/api_client.py tests/test_api_client.py
git commit -m "feat: add API client for remote backend communication"
```

---

## Task 3: Create API Server Module (api_server.py)

**Files:**
- Create: `doc_server/api_server.py`
- Test: `tests/test_api_server.py`

**Step 1: Write failing test**

```python
# tests/test_api_server.py
import pytest
from fastapi.testclient import TestClient
from doc_server.api_server import app

def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_api_server.py -v
# Expected: FAIL - module not found
```

**Step 3: Write minimal implementation**

```python
# doc_server/api_server.py
"""FastAPI server for remote Doc Server backend."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from doc_server import config, logging_config
from doc_server.ingestion.git_cloner import GitCloner
from doc_server.ingestion.document_processor import DocumentProcessor
from doc_server.search.vector_store import VectorStore
from doc_server.search.hybrid_search import HybridSearch

logger = logging_config.get_logger(__name__)


# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    library_id: str
    limit: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    content: str
    file_path: str
    score: float
    metadata: dict[str, Any]


class IngestRequest(BaseModel):
    source: str
    library_id: str
    version: str | None = None
    batch_size: int = Field(default=32, ge=1, le=128)


class IngestResponse(BaseModel):
    success: bool
    library_id: str
    version: str | None = None
    documents_ingested: int
    status: str


class LibraryInfo(BaseModel):
    library_id: str
    version: str | None = None
    document_count: int
    created_at: float | None = None


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]


# Global service instances
git_cloner: GitCloner | None = None
document_processor: DocumentProcessor | None = None
vector_store: VectorStore | None = None
hybrid_search: HybridSearch | None = None


def get_settings():
    return config.get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global git_cloner, document_processor, vector_store, hybrid_search
    
    settings = get_settings()
    
    logger.info("Initializing backend services", storage_path=settings.vector_db_path)
    
    git_cloner = GitCloner()
    document_processor = DocumentProcessor(
        batch_size=32,
        max_file_size=settings.max_file_size,
        encoding=settings.encoding,
    )
    vector_store = VectorStore(
        persist_directory=settings.vector_db_path,
        embedding_function=None,  # Will be set from settings
    )
    hybrid_search = HybridSearch(vector_store=vector_store)
    
    yield
    
    logger.info("Shutting down backend services")


app = FastAPI(
    title="Doc Server Backend",
    description="Remote backend for Doc Server API",
    version="1.0.0",
    lifespan=lifespan,
)


# Authentication dependency
async def verify_api_key(request: Request):
    """Verify API key from request headers."""
    settings = get_settings()
    
    # Skip auth if no API key configured
    if not settings.backend_api_key:
        return True
    
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != settings.backend_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )
    return True


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Check server health status."""
    settings = get_settings()
    
    # Check vector store
    vector_store_status = "connected"
    try:
        if vector_store:
            vector_store.client.heartbeat()
    except Exception:
        vector_store_status = "disconnected"
    
    return HealthResponse(
        status="healthy",
        components={
            "vector_store": vector_store_status,
            "api": "ready",
        },
    )


@app.post("/api/v1/search", response_model=list[SearchResult])
async def search(request: SearchRequest, _: bool = None):
    """Search through ingested documentation."""
    if not hybrid_search:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service not initialized",
        )
    
    results = await hybrid_search.search(
        query=request.query,
        library_id=request.library_id,
        limit=request.limit,
    )
    
    return [
        SearchResult(
            content=r.content,
            file_path=r.file_path,
            score=r.score,
            metadata=r.metadata,
        )
        for r in results
    ]


@app.post("/api/v1/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(request: IngestRequest):
    """Trigger full ingestion pipeline."""
    global vector_store, hybrid_search
    
    if not all([git_cloner, document_processor, vector_store, hybrid_search]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion services not initialized",
        )
    
    try:
        # Clone repository
        repo_path = await git_cloner.clone(request.source)
        
        # Process documents
        documents = await document_processor.process_directory(repo_path)
        
        # Add to vector store
        await vector_store.add_documents(
            documents=documents,
            library_id=request.library_id,
        )
        
        doc_count = len(documents)
        
        return IngestResponse(
            success=True,
            library_id=request.library_id,
            version=request.version,
            documents_ingested=doc_count,
            status="completed",
        )
        
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@app.get("/api/v1/libraries", response_model=list[LibraryInfo])
async def list_libraries(_: bool = None):
    """List all available libraries."""
    if not vector_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized",
        )
    
    libraries = vector_store.list_libraries()
    
    return [
        LibraryInfo(
            library_id=lib.library_id,
            version=lib.metadata.get("version"),
            document_count=lib.document_count,
            created_at=lib.metadata.get("created_at"),
        )
        for lib in libraries
    ]


@app.delete("/api/v1/libraries/{library_id}", response_model=bool)
async def remove_library(library_id: str, _: bool = None):
    """Remove a library and its documents."""
    if not vector_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized",
        )
    
    # Sanitize input
    library_id = config._sanitize_input(library_id)
    
    success = vector_store.delete_library(library_id)
    return success


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_api_server.py -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add doc_server/api_server.py tests/test_api_server.py
git commit -m "feat: add FastAPI server with core endpoints"
```

---

## Task 4: Update CLI for Backend Command

**Files:**
- Modify: `cli.py:1-50`

**Step 1: Read current cli.py**

```bash
# Run: Read cli.py to understand current structure
```

**Step 2: Add backend command**

```python
# In cli.py, add import and command
import uvicorn
from doc_server.api_server import app as api_app

@click.command("backend")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--workers", default=1, help="Number of worker processes")
def backend(host: str, port: int, workers: int):
    """Start the Doc Server backend server."""
    settings = config.get_settings()
    
    if not settings.backend_api_key:
        click.echo(
            "Warning: No API key configured. Set DOC_SERVER_API_KEY environment variable.",
            err=True,
        )
    
    click.echo(f"Starting backend server on {host}:{port}")
    uvicorn.run(
        "doc_server.api_server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
    )
```

**Step 3: Update ingest/search to support remote mode**

Modify existing commands to check `settings.mode`:

```python
# In ingest command
settings = config.get_settings()

if settings.mode == "remote":
    # Use API client
    from doc_server.api_client import APIClient
    
    async def run_remote_ingest():
        async with APIClient(
            base_url=settings.backend_url,
            api_key=settings.backend_api_key,
            timeout=settings.backend_timeout,
            verify_ssl=settings.backend_verify_ssl,
        ) as client:
            result = await client.ingest(
                source=source,
                library_id=library_id,
                version=version,
                batch_size=batch_size,
            )
            click.echo(f"Ingested {result.documents_ingested} documents")
            return result
    
    import asyncio
    asyncio.run(run_remote_ingest())
    return
```

**Step 4: Commit**

```bash
git add cli.py
git commit -m "feat: add backend server command and remote mode support"
```

---

## Task 5: Create Dockerfile

**Files:**
- Create: `Dockerfile`

**Step 1: Write Dockerfile**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY doc_server/ ./doc_server/
COPY config.py .

# Create data directories
RUN mkdir -p /data/vector_db /data/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DOC_SERVER_MODE=remote

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run backend server
CMD ["python", "-m", "uvicorn", "doc_server.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile for backend containerization"
```

---

## Task 6: Integration Tests

**Files:**
- Create: `tests/test_integration_remote.py`

**Step 1: Write integration test**

```python
"""Integration tests for remote backend communication."""

import pytest
from httpx import ASGITransport, AsyncClient
from doc_server.api_server import app


@pytest.mark.asyncio
async def test_full_ingestion_flow():
    """Test: Ingest -> Search -> List -> Remove"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # Health check
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        
        # List libraries (should be empty initially)
        response = await client.get(
            "/api/v1/libraries",
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code in [200, 401]  # Depends on auth setup
```

**Step 2: Run test**

```bash
pytest tests/test_integration_remote.py -v
```

**Step 3: Commit**

```bash
git add tests/test_integration_remote.py
git commit -m "test: add integration tests for remote backend"
```

---

## Task 7: Linting and Final Verification

**Step 1: Run ruff**

```bash
ruff check doc_server/
```

**Step 2: Run black**

```bash
black --check doc_server/
```

**Step 3: Run mypy**

```bash
mypy doc_server/
```

**Step 4: Run full test suite**

```bash
pytest tests/ -v
```

**Step 5: Final commit**

```bash
git add .
git commit -m "feat: complete remote backend implementation"
```

---

## Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | Configuration | config.py |
| 2 | API Client | api_client.py, test_api_client.py |
| 3 | API Server | api_server.py, test_api_server.py |
| 4 | CLI Updates | cli.py |
| 5 | Dockerfile | Dockerfile |
| 6 | Integration Tests | test_integration_remote.py |
| 7 | Verification | All |

**Total: 7 major tasks**
