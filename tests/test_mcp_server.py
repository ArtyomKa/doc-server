"""
Tests for MCP server implementation.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastmcp import FastMCP

from doc_server.mcp_server import (
    DocumentResult,
    _convert_search_result,
    mcp,
    search_docs,
    lifespan,
)
from doc_server.search.hybrid_search import SearchResult


class TestDocumentResult:
    """Tests for DocumentResult dataclass."""

    def test_document_result_creation(self):
        """Test creating a DocumentResult."""
        result = DocumentResult(
            content="Test content",
            file_path="/test/file.py",
            library_id="/test",
            relevance_score=0.95,
        )
        assert result.content == "Test content"
        assert result.file_path == "/test/file.py"
        assert result.library_id == "/test"
        assert result.relevance_score == 0.95
        assert result.line_numbers is None
        assert result.metadata == {}

    def test_document_result_with_line_numbers(self):
        """Test DocumentResult with line numbers."""
        result = DocumentResult(
            content="Test content",
            file_path="/test/file.py",
            library_id="/test",
            relevance_score=0.85,
            line_numbers=(10, 20),
        )
        assert result.line_numbers == (10, 20)

    def test_document_result_with_metadata(self):
        """Test DocumentResult with metadata."""
        metadata = {"author": "test", "version": "1.0"}
        result = DocumentResult(
            content="Test content",
            file_path="/test/file.py",
            library_id="/test",
            relevance_score=0.75,
            metadata=metadata,
        )
        assert result.metadata == metadata

    def test_to_dict(self):
        """Test to_dict method."""
        result = DocumentResult(
            content="Test content",
            file_path="/test/file.py",
            library_id="/test",
            relevance_score=0.95,
            line_numbers=(5, 10),
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert d["content"] == "Test content"
        assert d["file_path"] == "/test/file.py"
        assert d["library_id"] == "/test"
        assert d["relevance_score"] == 0.95
        assert d["line_numbers"] == (5, 10)
        assert d["metadata"] == {"key": "value"}

    def test_to_dict_without_optional_fields(self):
        """Test to_dict without optional fields."""
        result = DocumentResult(
            content="Test content",
            file_path="/test/file.py",
            library_id="/test",
            relevance_score=0.95,
        )
        d = result.to_dict()
        assert "line_numbers" not in d
        assert "metadata" not in d or d["metadata"] == {}


class TestConvertSearchResult:
    """Tests for _convert_search_result function."""

    def test_convert_search_result(self):
        """Test converting SearchResult to DocumentResult."""
        search_result = SearchResult(
            content="Test content",
            file_path="/test/file.py",
            library_id="/test",
            relevance_score=0.85,
            vector_score=0.9,
            keyword_score=0.8,
            line_numbers=(1, 5),
            metadata={"extra": "data"},
        )

        doc_result = _convert_search_result(search_result)

        assert isinstance(doc_result, DocumentResult)
        assert doc_result.content == "Test content"
        assert doc_result.file_path == "/test/file.py"
        assert doc_result.library_id == "/test"
        assert doc_result.relevance_score == 0.85
        assert doc_result.line_numbers == (1, 5)
        assert doc_result.metadata == {"extra": "data"}


class TestMCPServer:
    """Tests for MCP server instance and tool registration."""

    def test_mcp_server_exists(self):
        """Test that MCP server instance exists."""
        assert mcp is not None
        assert isinstance(mcp, FastMCP)

    def test_mcp_server_name(self):
        """Test MCP server has correct name."""
        assert mcp.name == "doc-server"


class TestSearchDocsFunction:
    """Tests for search_docs function (tests the underlying function logic)."""

    @pytest.fixture
    def mock_hybrid_search(self):
        """Create a mock HybridSearch instance."""
        with patch("doc_server.mcp_server.get_hybrid_search") as mock:
            search_instance = MagicMock()
            mock.return_value = search_instance
            yield search_instance

    def test_empty_query_raises_value_error(self, mock_hybrid_search):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_docs.fn(query="", library_id="/test")

    def test_whitespace_query_raises_value_error(self, mock_hybrid_search):
        """Test that whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search_docs.fn(query="   ", library_id="/test")

    def test_empty_library_id_raises_value_error(self, mock_hybrid_search):
        """Test that empty library_id raises ValueError."""
        with pytest.raises(ValueError, match="Library ID cannot be empty"):
            search_docs.fn(query="test", library_id="")

    def test_search_docs_normalizes_library_id(self, mock_hybrid_search):
        """Test that library_id is normalized."""
        mock_results = []
        mock_hybrid_search.search.return_value = mock_results

        search_docs.fn(query="test", library_id="pandas", limit=10)

        # Should normalize to /pandas
        mock_hybrid_search.search.assert_called_once_with(
            query="test",
            library_id="/pandas",
            n_results=10,
        )

    def test_search_docs_with_slash_prefix(self, mock_hybrid_search):
        """Test that library_id with slash is handled correctly."""
        mock_results = []
        mock_hybrid_search.search.return_value = mock_results

        search_docs.fn(query="test", library_id="/pandas", limit=10)

        mock_hybrid_search.search.assert_called_once_with(
            query="test",
            library_id="/pandas",
            n_results=10,
        )

    def test_search_docs_limit_enforced(self, mock_hybrid_search):
        """Test that limit is enforced (max 100)."""
        mock_results = []
        mock_hybrid_search.search.return_value = mock_results

        search_docs.fn(query="test", library_id="/test", limit=200)

        mock_hybrid_search.search.assert_called_once_with(
            query="test",
            library_id="/test",
            n_results=100,  # Should be capped at 100
        )

    def test_search_docs_limit_minimum(self, mock_hybrid_search):
        """Test that limit minimum is 10."""
        mock_results = []
        mock_hybrid_search.search.return_value = mock_results

        search_docs.fn(query="test", library_id="/test", limit=-5)

        mock_hybrid_search.search.assert_called_once_with(
            query="test",
            library_id="/test",
            n_results=10,  # Should default to 10
        )

    def test_search_docs_returns_results(self, mock_hybrid_search):
        """Test that search_docs returns results."""
        search_result = SearchResult(
            content="Test content",
            file_path="/test/file.py",
            library_id="/test",
            relevance_score=0.95,
            vector_score=0.9,
            keyword_score=0.8,
            line_numbers=(1, 5),
            metadata={},
        )
        mock_hybrid_search.search.return_value = [search_result]

        result = search_docs.fn(query="test", library_id="/test")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["content"] == "Test content"
        assert result[0]["file_path"] == "/test/file.py"
        assert result[0]["library_id"] == "/test"
        assert result[0]["relevance_score"] == 0.95

    def test_search_docs_handles_search_error(self, mock_hybrid_search):
        """Test that search errors are wrapped in RuntimeError."""
        mock_hybrid_search.search.side_effect = Exception("Search failed")

        with pytest.raises(RuntimeError, match="Search failed"):
            search_docs.fn(query="test", library_id="/test")

    def test_search_docs_multiple_results(self, mock_hybrid_search):
        """Test search with multiple results."""
        results = [
            SearchResult(
                content=f"Content {i}",
                file_path=f"/test/file{i}.py",
                library_id="/test",
                relevance_score=0.9 - i * 0.1,
                vector_score=0.9 - i * 0.1,
                keyword_score=0.8 - i * 0.1,
            )
            for i in range(5)
        ]
        mock_hybrid_search.search.return_value = results

        result = search_docs.fn(query="test", library_id="/test", limit=10)

        assert len(result) == 5
        assert result[0]["relevance_score"] == 0.9
        assert result[4]["relevance_score"] == 0.5

    def test_invalid_library_id_raises_error(self, mock_hybrid_search):
        """Test that invalid library_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid library ID"):
            search_docs.fn(query="test", library_id="/invalid<>/id")

    def test_search_docs_default_limit(self, mock_hybrid_search):
        """Test that default limit is 10."""
        mock_results = []
        mock_hybrid_search.search.return_value = mock_results

        search_docs.fn(query="test", library_id="/test")

        mock_hybrid_search.search.assert_called_once_with(
            query="test",
            library_id="/test",
            n_results=10,
        )


class TestLifespanManager:
    """Tests for the lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_search_service(self):
        """Test that lifespan initializes the hybrid search service."""
        from doc_server.mcp_server import lifespan

        with patch("doc_server.mcp_server.get_hybrid_search") as mock_get_search:
            mock_search = MagicMock()
            mock_get_search.return_value = mock_search

            async with lifespan(mcp):
                pass

            # Verify hybrid search was initialized
            mock_get_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_logs_startup(self):
        """Test that lifespan logs startup message."""
        from doc_server.mcp_server import lifespan

        with patch("doc_server.mcp_server.get_hybrid_search") as mock_get_search:
            mock_search = MagicMock()
            mock_get_search.return_value = mock_search

            with patch("doc_server.mcp_server.logger") as mock_logger:
                async with lifespan(mcp):
                    pass

                # Check that info logs were called
                assert mock_logger.info.called


class TestMainFunction:
    """Tests for the main function."""

    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        from doc_server.mcp_server import main

        assert callable(main)

    def test_main_uses_stdio_transport(self):
        """Test that main runs with stdio transport by default."""
        from doc_server.mcp_server import main
        import inspect

        source = inspect.getsource(main)
        # Verify that mcp.run() is called (uses stdio by default)
        assert "mcp.run()" in source

    def test_main_sets_up_signal_handlers(self):
        """Test that main sets up signal handlers."""
        from doc_server.mcp_server import main
        import inspect

        source = inspect.getsource(main)
        # Verify signal handlers are set up
        assert "signal.signal" in source
        assert "SIGTERM" in source or "SIGINT" in source
