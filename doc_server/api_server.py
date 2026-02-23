"""FastAPI server for remote Doc Server backend."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from doc_server import config, logging_config

logger = logging_config.get_logger(__name__)


# Request/Response Models
class SearchRequest(BaseModel):
    """Request model for search endpoint."""

    query: str
    library_id: str
    limit: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    """Response model for search results."""

    content: str
    file_path: str
    score: float
    metadata: dict[str, Any]


class IngestRequest(BaseModel):
    """Request model for ingest endpoint."""

    source: str
    library_id: str
    version: str | None = None
    batch_size: int = Field(default=32, ge=1, le=128)


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""

    success: bool
    library_id: str
    version: str | None = None
    documents_ingested: int
    status: str


class LibraryInfo(BaseModel):
    """Response model for library information."""

    library_id: str
    version: str | None = None
    document_count: int
    created_at: float | None = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    components: dict[str, str]


def get_settings() -> config.Settings:
    """Get application settings."""
    return config.settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup and clean up on shutdown."""
    logger.info("Starting Doc Server Backend API")

    yield

    logger.info("Shutting down Doc Server Backend API")


app = FastAPI(
    title="Doc Server Backend",
    description="Remote backend for Doc Server API",
    version="1.0.0",
    lifespan=lifespan,
)


def _sanitize_input(input_str: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent injection attacks."""
    return config._sanitize_input(input_str, max_length)


# Authentication dependency
async def verify_api_key(request: Request) -> bool:
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
    # Check if we can import and initialize components
    vector_store_status = "unknown"
    try:
        from doc_server.mcp_server import get_vector_store

        vs = get_vector_store()
        vs.client.heartbeat()
        vector_store_status = "connected"
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
async def search(request: SearchRequest):
    """Search through ingested documentation."""
    try:
        from doc_server.mcp_server import search_docs

        # Sanitize inputs
        sanitized_query = _sanitize_input(request.query, max_length=2000)
        sanitized_library_id = _sanitize_input(request.library_id)

        # Call search via .fn attribute
        results = await search_docs.fn(
            sanitized_query, sanitized_library_id, request.limit
        )

        return [
            SearchResult(
                content=r["content"],
                file_path=r["file_path"],
                score=r.get("relevance_score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]
    except Exception as e:
        logger.error("Search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        ) from e


@app.post(
    "/api/v1/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest(request: IngestRequest):
    """Trigger full ingestion pipeline."""
    try:
        from doc_server.mcp_server import ingest_library

        # Sanitize inputs
        sanitized_source = _sanitize_input(request.source, max_length=2000)
        sanitized_library_id = _sanitize_input(request.library_id)

        # Call ingest via .fn attribute - note: version not in the mcp function
        result = await ingest_library.fn(sanitized_source, sanitized_library_id)

        if result.get("success"):
            return IngestResponse(
                success=True,
                library_id=result["library_id"],
                version=request.version,
                documents_ingested=result.get("documents_ingested", 0),
                status="completed",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ingestion failed: {result.get('error', 'Unknown error')}",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        ) from e


@app.get("/api/v1/libraries", response_model=list[LibraryInfo])
async def list_libraries():
    """List all available libraries."""
    try:
        from doc_server.mcp_server import list_libraries as mcp_list_libraries

        libraries = await mcp_list_libraries.fn()

        return [
            LibraryInfo(
                library_id=lib["library_id"],
                version=lib.get("metadata", {}).get("version"),
                document_count=lib["document_count"],
                created_at=lib.get("created_at"),
            )
            for lib in libraries
        ]
    except Exception as e:
        logger.error("List libraries failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"List libraries failed: {str(e)}",
        ) from e


@app.delete("/api/v1/libraries/{library_id}", response_model=bool)
async def remove_library(library_id: str):
    """Remove a library and its documents."""
    try:
        from doc_server.mcp_server import remove_library as mcp_remove_library

        # Sanitize input
        sanitized_library_id = _sanitize_input(library_id)

        return await mcp_remove_library.fn(sanitized_library_id)
    except Exception as e:
        logger.error("Remove library failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Remove library failed: {str(e)}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
