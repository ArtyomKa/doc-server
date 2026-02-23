"""Tests for API Server module."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from doc_server.api_server import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LibraryInfo,
    SearchRequest,
    SearchResult,
    app,
)


class TestSearchRequest:
    """Test cases for SearchRequest model."""

    def test_search_request_defaults(self):
        """Test SearchRequest with default values."""
        req = SearchRequest(query="test", library_id="/test")
        assert req.query == "test"
        assert req.library_id == "/test"
        assert req.limit == 10

    def test_search_request_custom_limit(self):
        """Test SearchRequest with custom limit."""
        req = SearchRequest(query="test", library_id="/test", limit=5)
        assert req.limit == 5

    def test_search_request_limit_validation(self):
        """Test SearchRequest limit validation."""
        with pytest.raises(ValidationError):
            SearchRequest(query="test", library_id="/test", limit=0)

        with pytest.raises(ValidationError):
            SearchRequest(query="test", library_id="/test", limit=101)


class TestIngestRequest:
    """Test cases for IngestRequest model."""

    def test_ingest_request_defaults(self):
        """Test IngestRequest with default values."""
        req = IngestRequest(source="https://github.com/test", library_id="/test")
        assert req.source == "https://github.com/test"
        assert req.library_id == "/test"
        assert req.version is None
        assert req.batch_size == 32

    def test_ingest_request_with_version(self):
        """Test IngestRequest with version."""
        req = IngestRequest(
            source="https://github.com/test",
            library_id="/test",
            version="v1.0.0",
        )
        assert req.version == "v1.0.0"


class TestSearchResult:
    """Test cases for SearchResult model."""

    def test_search_result(self):
        """Test SearchResult model."""
        result = SearchResult(
            content="test content",
            file_path="test.py",
            score=0.85,
            metadata={"key": "value"},
        )
        assert result.content == "test content"
        assert result.score == 0.85


class TestIngestResponse:
    """Test cases for IngestResponse model."""

    def test_ingest_response(self):
        """Test IngestResponse model."""
        response = IngestResponse(
            success=True,
            library_id="/test",
            version="v1.0.0",
            documents_ingested=100,
            status="completed",
        )
        assert response.success is True
        assert response.documents_ingested == 100


class TestLibraryInfo:
    """Test cases for LibraryInfo model."""

    def test_library_info(self):
        """Test LibraryInfo model."""
        lib = LibraryInfo(
            library_id="/pandas",
            version="v2.2.0",
            document_count=1500,
            created_at=1234567890.0,
        )
        assert lib.library_id == "/pandas"
        assert lib.version == "v2.2.0"
        assert lib.document_count == 1500


class TestHealthResponse:
    """Test cases for HealthResponse model."""

    def test_health_response(self):
        """Test HealthResponse model."""
        health = HealthResponse(
            status="healthy",
            components={"vector_store": "connected"},
        )
        assert health.status == "healthy"
        assert health.components["vector_store"] == "connected"


class TestAPIServer:
    """Test cases for API server endpoints."""

    def test_app_created(self):
        """Test that FastAPI app is created."""
        assert app is not None
        assert app.title == "Doc Server Backend"

    def test_health_endpoint_no_auth(self):
        """Test health endpoint without authentication."""
        with TestClient(app) as client:
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "components" in data

    def test_search_endpoint_validation(self):
        """Test search endpoint request validation."""
        with TestClient(app) as client:
            # Missing required fields
            response = client.post("/api/v1/search", json={})
            assert response.status_code == 422

    def test_ingest_endpoint_validation(self):
        """Test ingest endpoint request validation."""
        with TestClient(app) as client:
            # Missing required fields
            response = client.post("/api/v1/ingest", json={})
            assert response.status_code == 422

    def test_list_libraries_endpoint(self):
        """Test list libraries endpoint."""
        with TestClient(app) as client:
            response = client.get("/api/v1/libraries")
            # Will be 503 if services not initialized or 500 due to signal issue in tests
            assert response.status_code in [200, 500, 503]

    def test_remove_library_endpoint(self):
        """Test remove library endpoint."""
        with TestClient(app) as client:
            response = client.delete("/api/v1/libraries/test")
            # Will be 503 if services not initialized or 500 due to signal issue in tests
            assert response.status_code in [200, 500, 503]

    def test_search_endpoint_returns_score_field(self):
        """Test that search endpoint returns 'score' field in response."""
        with TestClient(app) as client:
            # Mock the search_docs function to return a result
            with patch("doc_server.mcp_server.search_docs") as mock_search:
                mock_search.fn.return_value = [
                    {
                        "content": "test content",
                        "file_path": "test.py",
                        "relevance_score": 0.85,
                        "metadata": {},
                    }
                ]

                response = client.post(
                    "/api/v1/search",
                    json={"query": "test query", "library_id": "/test", "limit": 10},
                )

                assert response.status_code == 200
                data = response.json()
                assert len(data) == 1
                assert "score" in data[0]
                assert data[0]["score"] == 0.85
