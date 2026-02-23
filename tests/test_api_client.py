"""Tests for API Client module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from doc_server.api_client import (
    APIClient,
    HealthResult,
    LibraryInfo,
    SearchResult,
)


class TestAPIClient:
    """Test cases for APIClient."""

    def test_client_initialization(self):
        """Test client initialization with default values."""
        client = APIClient()
        assert client.base_url == "http://localhost:8000"
        assert client.api_key == ""
        assert client.timeout == 30
        assert client.verify_ssl is True

    def test_client_initialization_with_params(self):
        """Test client initialization with custom parameters."""
        client = APIClient(
            base_url="https://api.example.com",
            api_key="secret-key",
            timeout=60,
            verify_ssl=False,
        )
        assert client.base_url == "https://api.example.com"
        assert client.api_key == "secret-key"
        assert client.timeout == 60
        assert client.verify_ssl is False

    def test_client_url_stripping(self):
        """Test that trailing slash is stripped from URL."""
        client = APIClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_get_headers_without_key(self):
        """Test headers without API key."""
        client = APIClient()
        headers = client._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert "X-API-Key" not in headers

    def test_get_headers_with_key(self):
        """Test headers with API key."""
        client = APIClient(api_key="test-key")
        headers = client._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["X-API-Key"] == "test-key"

    @pytest.mark.asyncio
    async def test_search_forwarded_to_backend(self):
        """Test search method calls correct endpoint."""
        client = APIClient(base_url="http://localhost:8000", api_key="test-key")
        client._client = AsyncMock()

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "content": "test content",
                "file_path": "test.py",
                "score": 0.85,
                "metadata": {"library": "/test"},
            }
        ]
        mock_response.raise_for_status = MagicMock()
        client._client.post = AsyncMock(return_value=mock_response)

        results = await client.search("test query", "/test", limit=10)

        client._client.post.assert_called_once_with(
            "/api/v1/search",
            json={"query": "test query", "library_id": "/test", "limit": 10},
            headers={"Content-Type": "application/json", "X-API-Key": "test-key"},
        )
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].content == "test content"

    @pytest.mark.asyncio
    async def test_list_libraries(self):
        """Test list_libraries method."""
        client = APIClient(base_url="http://localhost:8000", api_key="test-key")
        client._client = AsyncMock()

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "library_id": "/pandas",
                "version": "v2.2.0",
                "document_count": 1500,
                "created_at": 1234567890.0,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        client._client.get = AsyncMock(return_value=mock_response)

        libraries = await client.list_libraries()

        assert len(libraries) == 1
        assert isinstance(libraries[0], LibraryInfo)
        assert libraries[0].library_id == "/pandas"
        assert libraries[0].version == "v2.2.0"

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health_check method."""
        client = APIClient(base_url="http://localhost:8000")
        client._client = AsyncMock()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "healthy",
            "components": {"vector_store": "connected"},
        }
        mock_response.raise_for_status = MagicMock()
        client._client.get = AsyncMock(return_value=mock_response)

        health = await client.health_check()

        assert isinstance(health, HealthResult)
        assert health.status == "healthy"

    @pytest.mark.asyncio
    async def test_close(self):
        """Test close method cleans up client."""
        client = APIClient()
        # Initialize client via _get_client
        await client._get_client()
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        async with APIClient(base_url="http://localhost:8000") as client:
            assert client._client is None
        # After exiting context, client should be closed
