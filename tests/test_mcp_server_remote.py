"""
Tests for MCP server remote mode functionality.

These tests verify that when mode is set to "remote", the MCP server
uses the API client to communicate with the backend instead of
directly using local services.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_server.mcp_server import (
    health_check,
    ingest_library,
    list_libraries,
    remove_library,
    search_docs,
    validate_server,
)


class TestSearchDocsRemoteMode:
    """Tests for search_docs in remote mode."""

    @pytest.mark.asyncio
    async def test_search_docs_uses_api_client_when_remote(self):
        """When mode is remote, search_docs should use APIClient instead of local search."""
        with patch("doc_server.mcp_server.settings") as mock_settings:
            mock_settings.mode = "remote"
            mock_settings.backend_url = "http://localhost:8000"
            mock_settings.backend_api_key = "test-key"
            mock_settings.backend_timeout = 30
            mock_settings.backend_verify_ssl = True
            mock_settings.normalize_library_id = MagicMock(return_value="/test-library")

            with patch("doc_server.mcp_server.get_api_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_client.search = AsyncMock(
                    return_value=[
                        MagicMock(
                            content="test content",
                            file_path="test.py",
                            score=0.9,
                            metadata={},
                        )
                    ]
                )
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_get_client.return_value = mock_client

                # Call search_docs.fn - the underlying function
                result = await search_docs.fn("test query", "/test-library", limit=10)

                # Verify APIClient was used
                mock_client.search.assert_called_once_with(
                    query="test query",
                    library_id="/test-library",
                    limit=10,
                )
                assert len(result) == 1
                assert result[0]["content"] == "test content"


class TestListLibrariesRemoteMode:
    """Tests for list_libraries in remote mode."""

    @pytest.mark.asyncio
    async def test_list_libraries_uses_api_client_when_remote(self):
        """When mode is remote, list_libraries should use APIClient."""
        with patch("doc_server.mcp_server.settings") as mock_settings:
            mock_settings.mode = "remote"
            mock_settings.backend_url = "http://localhost:8000"
            mock_settings.backend_api_key = "test-key"
            mock_settings.backend_timeout = 30
            mock_settings.backend_verify_ssl = True

            with patch("doc_server.mcp_server.get_api_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_client.list_libraries = AsyncMock(
                    return_value=[
                        MagicMock(
                            library_id="/test-lib",
                            document_count=100,
                            created_at=1234567890.0,
                        )
                    ]
                )
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_get_client.return_value = mock_client

                result = await list_libraries.fn()

                mock_client.list_libraries.assert_called_once()
                assert len(result) == 1
                assert result[0]["library_id"] == "/test-lib"


class TestRemoveLibraryRemoteMode:
    """Tests for remove_library in remote mode."""

    @pytest.mark.asyncio
    async def test_remove_library_uses_api_client_when_remote(self):
        """When mode is remote, remove_library should use APIClient."""
        with patch("doc_server.mcp_server.settings") as mock_settings:
            mock_settings.mode = "remote"
            mock_settings.backend_url = "http://localhost:8000"
            mock_settings.backend_api_key = "test-key"
            mock_settings.backend_timeout = 30
            mock_settings.backend_verify_ssl = True
            mock_settings.normalize_library_id = MagicMock(return_value="/test-library")

            with patch("doc_server.mcp_server.get_api_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_client.remove_library = AsyncMock(return_value=True)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_get_client.return_value = mock_client

                result = await remove_library.fn("/test-library")

                mock_client.remove_library.assert_called_once_with("/test-library")
                assert result is True


class TestIngestLibraryRemoteMode:
    """Tests for ingest_library in remote mode."""

    @pytest.mark.asyncio
    async def test_ingest_library_uses_api_client_when_remote(self):
        """When mode is remote, ingest_library should use APIClient."""
        with patch("doc_server.mcp_server.settings") as mock_settings:
            mock_settings.mode = "remote"
            mock_settings.backend_url = "http://localhost:8000"
            mock_settings.backend_api_key = "test-key"
            mock_settings.backend_timeout = 30
            mock_settings.backend_verify_ssl = True
            mock_settings.normalize_library_id = MagicMock(return_value="/test-library")

            with patch("doc_server.mcp_server.get_api_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_client.ingest = AsyncMock(
                    return_value=MagicMock(
                        success=True,
                        documents_ingested=50,
                        library_id="/test-library",
                    )
                )
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_get_client.return_value = mock_client

                result = await ingest_library.fn(
                    "https://github.com/test/repo", "/test-library"
                )

                mock_client.ingest.assert_called_once()
                assert result["success"] is True
                assert result["documents_ingested"] == 50


class TestHealthCheckRemoteMode:
    """Tests for health_check in remote mode."""

    @pytest.mark.asyncio
    async def test_health_check_uses_api_client_when_remote(self):
        """When mode is remote, health_check should use APIClient."""
        with patch("doc_server.mcp_server.settings") as mock_settings:
            mock_settings.mode = "remote"
            mock_settings.backend_url = "http://localhost:8000"
            mock_settings.backend_api_key = "test-key"
            mock_settings.backend_timeout = 30
            mock_settings.backend_verify_ssl = True

            with patch("doc_server.mcp_server.get_api_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_client.health_check = AsyncMock(
                    return_value=MagicMock(
                        status="healthy",
                        components={"vector_store": "connected"},
                        timestamp=1234567890.0,
                    )
                )
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_get_client.return_value = mock_client

                result = await health_check.fn()

                mock_client.health_check.assert_called_once()
                assert result["status"] == "healthy"


class TestValidateServerRemoteMode:
    """Tests for validate_server in remote mode."""

    @pytest.mark.asyncio
    async def test_validate_server_uses_api_client_when_remote(self):
        """When mode is remote, validate_server should use APIClient."""
        with patch("doc_server.mcp_server.settings") as mock_settings:
            mock_settings.mode = "remote"
            mock_settings.backend_url = "http://localhost:8000"
            mock_settings.backend_api_key = "test-key"
            mock_settings.backend_timeout = 30
            mock_settings.backend_verify_ssl = True

            with patch("doc_server.mcp_server.get_api_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_client.health_check = AsyncMock(
                    return_value=MagicMock(
                        status="healthy",
                        components={"api": "ready"},
                        timestamp=1234567890.0,
                    )
                )
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_get_client.return_value = mock_client

                result = await validate_server.fn()

                mock_client.health_check.assert_called_once()
                assert result["status"] == "healthy"
