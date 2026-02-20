"""API Client for communicating with remote Doc Server backend."""

from typing import Any

import httpx
from pydantic import BaseModel


class SearchResult(BaseModel):
    """Search result from remote backend."""

    content: str
    file_path: str
    score: float
    metadata: dict[str, Any]


class LibraryInfo(BaseModel):
    """Library information from remote backend."""

    library_id: str
    version: str | None = None
    document_count: int
    created_at: float | None = None


class IngestResult(BaseModel):
    """Ingestion result from remote backend."""

    success: bool
    library_id: str
    version: str | None = None
    documents_ingested: int
    status: str


class HealthResult(BaseModel):
    """Health check result from remote backend."""

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
        """Get HTTP headers including API key."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
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
