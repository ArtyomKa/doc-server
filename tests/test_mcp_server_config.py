"""
Tests for MCP server configuration and health check functions.
"""

from unittest.mock import patch, MagicMock

import pytest


class TestHealthCheckFunctions:
    """Test cases for health check and validation functions."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch("doc_server.mcp_server.settings") as mock:
            mock.storage_path.exists.return_value = True
            mock.storage_path.is_dir.return_value = True
            mock.storage_path.parent.is_dir.return_value = True
            mock.chroma_db_path.exists.return_value = True
            mock.models_path.exists.return_value = True
            mock.libraries_path.exists.return_value = True
            mock.log_level = "INFO"
            mock.log_format = "json"
            yield mock

    def test_validate_startup_returns_dict(self, mock_settings):
        """Test that validate_startup returns a dictionary."""
        from doc_server.mcp_server import validate_startup

        result = validate_startup()

        assert isinstance(result, dict)
        assert "status" in result
        assert "components" in result
        assert "timestamp" in result

    def test_get_health_status_returns_dict(self, mock_settings):
        """Test that get_health_status returns a dictionary."""
        from doc_server.mcp_server import get_health_status

        result = get_health_status()

        assert isinstance(result, dict)
        assert "status" in result
        assert "version" in result
        assert "timestamp" in result
        assert "components" in result

    def test_validate_startup_storage_check(self, mock_settings):
        """Test that validate_startup checks storage paths."""
        from doc_server.mcp_server import validate_startup

        with patch("doc_server.mcp_server.get_vector_store") as mock_vector_store:
            with patch("doc_server.mcp_server.get_hybrid_search") as mock_search:
                mock_vector_store_instance = MagicMock()
                mock_vector_store_instance.list_collections.return_value = []
                mock_vector_store.return_value = mock_vector_store_instance

                mock_search_instance = MagicMock()
                mock_search_instance.vector_weight = 0.7
                mock_search_instance.keyword_weight = 0.3
                mock_search.return_value = mock_search_instance

                result = validate_startup()

                assert "storage" in result["components"]

    def test_health_check_with_no_collections(self, mock_settings):
        """Test health check when no collections exist."""
        from doc_server.mcp_server import get_health_status

        with patch("doc_server.mcp_server.get_vector_store") as mock_vector_store:
            mock_vector_store_instance = MagicMock()
            mock_vector_store_instance.list_collections.return_value = []
            mock_vector_store.return_value = mock_vector_store_instance

            result = get_health_status()

            # Server is healthy even with no collections
            assert result["status"] == "healthy"
            assert result["components"]["vector_store"]["collection_count"] == 0


class TestStructuredLoggingIntegration:
    """Test integration of structured logging with MCP server."""

    def test_log_context_binds_library_id(self):
        """Test that LogContext can bind library_id."""
        from doc_server.logging_config import LogContext

        with LogContext(library_id="/test-library"):
            from doc_server.logging_config import get_logger

            logger = get_logger("test")
            logger.info("Test message")

    def test_log_context_binds_operation(self):
        """Test that LogContext can bind operation type."""
        from doc_server.logging_config import LogContext

        with LogContext(operation="search_docs"):
            from doc_server.logging_config import get_logger

            logger = get_logger("test")
            logger.info("Test message")
