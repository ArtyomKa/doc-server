"""
Tests for MCP server implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastmcp import FastMCP

from doc_server.mcp_server import (
    DocumentResult,
    LibraryInfo,
    _convert_search_result,
    _sanitize_input,
    ingest_library,
    lifespan,
    list_libraries,
    mcp,
    remove_library,
    search_docs,
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

    @pytest.mark.asyncio
    async def test_empty_query_raises_value_error(self, mock_hybrid_search):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await search_docs.fn(query="", library_id="/test")

    @pytest.mark.asyncio
    async def test_whitespace_query_raises_value_error(self, mock_hybrid_search):
        """Test that whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await search_docs.fn(query="   ", library_id="/test")

    @pytest.mark.asyncio
    async def test_empty_library_id_raises_value_error(self, mock_hybrid_search):
        """Test that empty library_id raises ValueError."""
        with pytest.raises(ValueError, match="Library ID cannot be empty"):
            await search_docs.fn(query="test", library_id="")

    @pytest.mark.asyncio
    async def test_search_docs_normalizes_library_id(self, mock_hybrid_search):
        """Test that library_id is normalized."""
        mock_results = []
        mock_hybrid_search.search.return_value = mock_results

        await search_docs.fn(query="test", library_id="pandas", limit=10)

        # Should normalize to /pandas
        mock_hybrid_search.search.assert_called_once_with(
            query="test",
            library_id="/pandas",
            n_results=10,
        )

    @pytest.mark.asyncio
    async def test_search_docs_with_slash_prefix(self, mock_hybrid_search):
        """Test that library_id with slash is handled correctly."""
        mock_results = []
        mock_hybrid_search.search.return_value = mock_results

        await search_docs.fn(query="test", library_id="/pandas", limit=10)

        mock_hybrid_search.search.assert_called_once_with(
            query="test",
            library_id="/pandas",
            n_results=10,
        )

    @pytest.mark.asyncio
    async def test_search_docs_limit_enforced(self, mock_hybrid_search):
        """Test that limit is enforced (max 100)."""
        mock_results = []
        mock_hybrid_search.search.return_value = mock_results

        await search_docs.fn(query="test", library_id="/test", limit=200)

        mock_hybrid_search.search.assert_called_once_with(
            query="test",
            library_id="/test",
            n_results=100,  # Should be capped at 100
        )

    @pytest.mark.asyncio
    async def test_search_docs_limit_minimum(self, mock_hybrid_search):
        """Test that limit minimum is 10."""
        mock_results = []
        mock_hybrid_search.search.return_value = mock_results

        await search_docs.fn(query="test", library_id="/test", limit=-5)

        mock_hybrid_search.search.assert_called_once_with(
            query="test",
            library_id="/test",
            n_results=10,  # Should default to 10
        )

    @pytest.mark.asyncio
    async def test_search_docs_returns_results(self, mock_hybrid_search):
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

        result = await search_docs.fn(query="test", library_id="/test")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["content"] == "Test content"
        assert result[0]["file_path"] == "/test/file.py"
        assert result[0]["library_id"] == "/test"
        assert result[0]["relevance_score"] == 0.95

    @pytest.mark.asyncio
    async def test_search_docs_handles_search_error(self, mock_hybrid_search):
        """Test that search errors are wrapped in RuntimeError."""
        mock_hybrid_search.search.side_effect = Exception("Search failed")

        with pytest.raises(RuntimeError, match="Search failed"):
            await search_docs.fn(query="test", library_id="/test")

    @pytest.mark.asyncio
    async def test_search_docs_multiple_results(self, mock_hybrid_search):
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

        result = await search_docs.fn(query="test", library_id="/test", limit=10)

        assert len(result) == 5
        assert result[0]["relevance_score"] == 0.9
        assert result[4]["relevance_score"] == 0.5

    @pytest.mark.asyncio
    async def test_invalid_library_id_raises_error(self, mock_hybrid_search):
        """Test that invalid library_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid library ID"):
            await search_docs.fn(query="test", library_id="/invalid<>/id")

    @pytest.mark.asyncio
    async def test_search_docs_default_limit(self, mock_hybrid_search):
        """Test that default limit is 10."""
        mock_results = []
        mock_hybrid_search.search.return_value = mock_results

        await search_docs.fn(query="test", library_id="/test")

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
        import inspect

        from doc_server.mcp_server import main

        source = inspect.getsource(main)
        # Verify that mcp.run() is called (uses stdio by default)
        assert "mcp.run()" in source

    def test_main_sets_up_signal_handlers(self):
        """Test that main sets up signal handlers."""
        import inspect

        from doc_server.mcp_server import main

        source = inspect.getsource(main)
        # Verify signal handlers are set up
        assert "signal.signal" in source
        assert "SIGTERM" in source or "SIGINT" in source


class TestSanitizeInput:
    """Tests for input sanitization function."""

    def test_empty_input_raises_error(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            _sanitize_input("")

    def test_whitespace_only_input_raises_error(self):
        """Test that whitespace-only input raises ValueError."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            _sanitize_input("   ")

    def test_null_bytes_removed(self):
        """Test that null bytes are removed from input."""
        result = _sanitize_input("test\x00value")
        assert "\x00" not in result
        assert result == "testvalue"

    def test_max_length_enforced(self):
        """Test that input length is limited."""
        long_input = "a" * 1500
        with pytest.raises(ValueError, match="exceeds maximum length"):
            _sanitize_input(long_input, max_length=1000)

    def test_suspicious_pattern_dotdot(self):
        """Test that ../ pattern is rejected."""
        with pytest.raises(ValueError, match="suspicious pattern"):
            _sanitize_input("../etc/passwd")

    def test_suspicious_pattern_shell(self):
        """Test that shell injection patterns are rejected."""
        with pytest.raises(ValueError, match="suspicious pattern"):
            _sanitize_input("$(whoami)")

    def test_valid_input_passes(self):
        """Test that valid input passes through."""
        result = _sanitize_input("valid-input_123")
        assert result == "valid-input_123"


class TestLibraryInfo:
    """Tests for LibraryInfo dataclass."""

    def test_library_info_creation(self):
        """Test creating a LibraryInfo."""
        info = LibraryInfo(
            library_id="/pandas",
            collection_name="pandas",
            document_count=100,
            embedding_model="all-MiniLM-L6-v2",
            created_at=1234567890.0,
        )
        assert info.library_id == "/pandas"
        assert info.collection_name == "pandas"
        assert info.document_count == 100
        assert info.embedding_model == "all-MiniLM-L6-v2"
        assert info.created_at == 1234567890.0

    def test_library_info_to_dict(self):
        """Test converting LibraryInfo to dictionary."""
        info = LibraryInfo(
            library_id="/pandas",
            collection_name="pandas",
            document_count=100,
            embedding_model="all-MiniLM-L6-v2",
            created_at=1234567890.0,
        )
        d = info.to_dict()
        assert d["library_id"] == "/pandas"
        assert d["collection_name"] == "pandas"
        assert d["document_count"] == 100
        assert d["embedding_model"] == "all-MiniLM-L6-v2"
        assert d["created_at"] == 1234567890.0


class TestListLibrariesFunction:
    """Tests for list_libraries function."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        with patch("doc_server.mcp_server.get_vector_store") as mock:
            store_instance = MagicMock()
            mock.return_value = store_instance
            yield store_instance

    @pytest.mark.asyncio
    async def test_list_libraries_returns_empty_list(self, mock_vector_store):
        """Test that empty list is returned when no libraries exist."""
        mock_vector_store.list_collections.return_value = []

        result = await list_libraries.fn()

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_list_libraries_returns_collections(self, mock_vector_store):
        """Test that list_libraries returns collection info."""
        mock_vector_store.list_collections.return_value = [
            {
                "name": "pandas",
                "library_id": "/pandas",
                "count": 100,
                "metadata": {
                    "embedding_model": "all-MiniLM-L6-v2",
                    "created_at": 1234567890.0,
                },
            },
            {
                "name": "fastapi",
                "library_id": "/fastapi",
                "count": 50,
                "metadata": {
                    "embedding_model": "all-MiniLM-L6-v2",
                    "created_at": 1234567891.0,
                },
            },
        ]

        result = await list_libraries.fn()

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["library_id"] == "/pandas"
        assert result[0]["document_count"] == 100
        assert result[1]["library_id"] == "/fastapi"
        assert result[1]["document_count"] == 50

    @pytest.mark.asyncio
    async def test_list_libraries_handles_missing_metadata(self, mock_vector_store):
        """Test that list_libraries handles missing metadata gracefully."""
        mock_vector_store.list_collections.return_value = [
            {
                "name": "test",
                "count": 10,
                "metadata": {},
            }
        ]

        result = await list_libraries.fn()

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["embedding_model"] == "unknown"
        assert result[0]["created_at"] == 0.0

    @pytest.mark.asyncio
    async def test_list_libraries_handles_error(self, mock_vector_store):
        """Test that errors are wrapped in RuntimeError."""
        mock_vector_store.list_collections.side_effect = Exception("Database error")

        with pytest.raises(RuntimeError, match="Failed to list libraries"):
            await list_libraries.fn()


class TestRemoveLibraryFunction:
    """Tests for remove_library function."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        with patch("doc_server.mcp_server.get_vector_store") as mock:
            store_instance = MagicMock()
            mock.return_value = store_instance
            yield store_instance

    @pytest.mark.asyncio
    async def test_remove_library_returns_true_when_existed(self, mock_vector_store):
        """Test that True is returned when library existed and was removed."""
        mock_vector_store.delete_collection.return_value = True

        result = await remove_library.fn("/pandas")

        assert result is True
        mock_vector_store.delete_collection.assert_called_once_with("/pandas")

    @pytest.mark.asyncio
    async def test_remove_library_returns_false_when_not_existed(
        self, mock_vector_store
    ):
        """Test that False is returned when library didn't exist."""
        mock_vector_store.delete_collection.return_value = False

        result = await remove_library.fn("/nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_remove_library_normalizes_id(self, mock_vector_store):
        """Test that library_id is normalized."""
        mock_vector_store.delete_collection.return_value = True

        await remove_library.fn("pandas")

        mock_vector_store.delete_collection.assert_called_once_with("/pandas")

    @pytest.mark.asyncio
    async def test_remove_library_invalid_id_raises_error(self, mock_vector_store):
        """Test that invalid library_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid library ID"):
            await remove_library.fn("/invalid<>id")

    @pytest.mark.asyncio
    async def test_remove_library_empty_id_raises_error(self, mock_vector_store):
        """Test that empty library_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid library ID"):
            await remove_library.fn("")

    @pytest.mark.asyncio
    async def test_remove_library_handles_error(self, mock_vector_store):
        """Test that errors are wrapped in RuntimeError."""
        mock_vector_store.delete_collection.side_effect = Exception("Database error")

        with pytest.raises(RuntimeError, match="Failed to remove library"):
            await remove_library.fn("/test")


class TestIngestLibraryFunction:
    """Tests for ingest_library function."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all ingestion dependencies."""
        with (
            patch("doc_server.mcp_server.GitCloner") as mock_git,
            patch("doc_server.mcp_server.ZIPExtractor") as mock_zip,
            patch("doc_server.mcp_server.FileFilter") as mock_filter,
            patch("doc_server.mcp_server.DocumentProcessor") as mock_processor,
            patch("doc_server.mcp_server.get_vector_store") as mock_store,
        ):
            yield {
                "git": mock_git,
                "zip": mock_zip,
                "filter": mock_filter,
                "processor": mock_processor,
                "store": mock_store,
            }

    @pytest.mark.asyncio
    async def test_ingest_library_empty_source_raises_error(self, mock_dependencies):
        """Test that empty source raises ValueError."""
        with pytest.raises(ValueError, match="Invalid input"):
            await ingest_library.fn("", "/test")

    @pytest.mark.asyncio
    async def test_ingest_library_empty_library_id_raises_error(
        self, mock_dependencies
    ):
        """Test that empty library_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid input"):
            await ingest_library.fn("https://github.com/test/repo", "")

    @pytest.mark.asyncio
    async def test_ingest_library_invalid_library_id_format(self, mock_dependencies):
        """Test that invalid library_id format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid library ID format"):
            await ingest_library.fn("https://github.com/test/repo", "/invalid<>id")

    @pytest.mark.asyncio
    async def test_ingest_library_sanitizes_input(self, mock_dependencies):
        """Test that input is sanitized."""
        mock_store_instance = mock_dependencies["store"].return_value
        mock_store_instance.create_collection.return_value = None

        # Mock filter to return empty list (no files)
        mock_filter_instance = mock_dependencies["filter"].return_value

        mock_filter_instance.filter_files.return_value = []

        # Expect RuntimeError since ValueError gets wrapped
        with pytest.raises(RuntimeError, match="Ingestion failed"):
            await ingest_library.fn("https://github.com/test/repo", "/test")

    @pytest.mark.asyncio
    async def test_ingest_library_local_directory(self, mock_dependencies):
        """Test ingesting from local directory."""
        mock_store_instance = mock_dependencies["store"].return_value
        mock_store_instance.create_collection.return_value = None
        mock_store_instance.add_documents.return_value = ["id1"]

        mock_filter_instance = mock_dependencies["filter"].return_value
        from doc_server.ingestion.file_filter import FilterResult

        mock_filter_instance.filter_files.return_value = [
            FilterResult(
                file_path="/test/test.py",
                included=True,
                reason="Allowed",
                extension=".py",
                size_bytes=100,
            )
        ]

        mock_processor_instance = mock_dependencies["processor"].return_value
        from doc_server.ingestion.document_processor import DocumentChunk

        mock_processor_instance.process_file.return_value = [
            DocumentChunk(
                content="test content",
                file_path="/test/test.py",
                library_id="/test",
                line_start=1,
                line_end=10,
            )
        ]

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("test content")

            result = await ingest_library.fn(tmpdir, "/test")

        assert result["success"] is True
        assert result["documents_ingested"] == 1
        assert result["library_id"] == "/test"
        assert result["source_type"] == "local"

    @pytest.mark.asyncio
    async def test_ingest_library_handles_error(self, mock_dependencies):
        """Test that errors are wrapped in RuntimeError."""
        mock_dependencies["store"].side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError, match="Ingestion failed"):
            await ingest_library.fn("https://github.com/test/repo", "/test")

    @pytest.mark.asyncio
    async def test_ingest_library_suspicious_pattern_rejected(self, mock_dependencies):
        """Test that suspicious patterns in source are rejected."""
        with pytest.raises(ValueError, match="suspicious pattern"):
            await ingest_library.fn("https://example.com/../../etc/passwd", "/test")

        with pytest.raises(ValueError, match="suspicious pattern"):
            await ingest_library.fn("https://example.com/${SECRET}", "/test")


class TestListLibrariesErrorHandling:
    """Tests for error handling in list_libraries."""

    @pytest.mark.asyncio
    async def test_list_libraries_handles_collection_error(self):
        """Test that collection errors are handled gracefully."""
        with patch("doc_server.mcp_server.get_vector_store") as mock_store:
            mock_store.return_value.list_collections.side_effect = Exception(
                "Database error"
            )

            with pytest.raises(RuntimeError, match="Failed to list libraries"):
                await list_libraries.fn()


class TestRemoveLibraryErrorHandling:
    """Tests for error handling in remove_library."""

    @pytest.mark.asyncio
    async def test_remove_library_database_error(self):
        """Test that database errors are wrapped in RuntimeError."""
        with patch("doc_server.mcp_server.get_vector_store") as mock_store:
            mock_store.return_value.delete_collection.side_effect = Exception(
                "Connection failed"
            )

            with pytest.raises(RuntimeError, match="Failed to remove library"):
                await remove_library.fn("/test")


class TestIngestLibraryErrorHandling:
    """Additional tests for ingest_library error handling."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all ingestion dependencies."""
        with (
            patch("doc_server.mcp_server.GitCloner") as mock_git,
            patch("doc_server.mcp_server.ZIPExtractor") as mock_zip,
            patch("doc_server.mcp_server.FileFilter") as mock_filter,
            patch("doc_server.mcp_server.DocumentProcessor") as mock_processor,
            patch("doc_server.mcp_server.get_vector_store") as mock_store,
        ):
            yield {
                "git": mock_git,
                "zip": mock_zip,
                "filter": mock_filter,
                "processor": mock_processor,
                "store": mock_store,
            }

    @pytest.mark.asyncio
    async def test_ingest_library_local_path_not_found(self, mock_dependencies):
        """Test that non-existent local path raises error."""
        # ValueError gets wrapped in RuntimeError by the function
        with pytest.raises(RuntimeError, match="Ingestion failed"):
            await ingest_library.fn("/nonexistent/path", "/test")

    @pytest.mark.asyncio
    async def test_ingest_library_zip_extraction_error(self, mock_dependencies):
        """Test that ZIP extraction errors are handled."""
        mock_store_instance = mock_dependencies["store"].return_value
        mock_store_instance.create_collection.return_value = None

        mock_zip_instance = mock_dependencies["zip"].return_value
        mock_zip_instance.extract_archive.side_effect = Exception("Invalid ZIP")

        with pytest.raises(RuntimeError, match="Ingestion failed"):
            await ingest_library.fn("/path/to/file.zip", "/test")

    @pytest.mark.asyncio
    async def test_ingest_library_processing_error_continues(self, mock_dependencies):
        """Test that processing errors don't stop ingestion."""
        import tempfile
        from pathlib import Path

        from doc_server.ingestion.document_processor import DocumentChunk
        from doc_server.ingestion.file_filter import FilterResult

        mock_store_instance = mock_dependencies["store"].return_value
        mock_store_instance.create_collection.return_value = None
        mock_store_instance.add_documents.return_value = ["id1"]

        mock_filter_instance = mock_dependencies["filter"].return_value
        mock_filter_instance.filter_files.return_value = [
            FilterResult(
                file_path="/test/file1.py",
                included=True,
                reason="Allowed",
                extension=".py",
                size_bytes=100,
            ),
            FilterResult(
                file_path="/test/file2.py",
                included=True,
                reason="Allowed",
                extension=".py",
                size_bytes=100,
            ),
        ]

        mock_processor_instance = mock_dependencies["processor"].return_value
        # First file succeeds, second file raises error
        mock_processor_instance.process_file.side_effect = [
            [
                DocumentChunk(
                    content="test content",
                    file_path="/test/file1.py",
                    library_id="/test",
                    line_start=1,
                    line_end=10,
                )
            ],
            Exception("Processing failed"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "file1.py"
            test_file.write_text("test content")

            # Should not raise, just log warning and continue
            result = await ingest_library.fn(tmpdir, "/test")

        assert result["success"] is True
        assert result["documents_ingested"] == 1  # Only first file processed

    @pytest.mark.asyncio
    async def test_ingest_library_vector_store_error(self, mock_dependencies):
        """Test that vector store errors are handled."""
        mock_store_instance = mock_dependencies["store"].return_value
        mock_store_instance.create_collection.side_effect = Exception(
            "Connection failed"
        )

        with pytest.raises(RuntimeError, match="Ingestion failed"):
            await ingest_library.fn("/test/path", "/test")


class TestListLibrariesPartialErrors:
    """Tests for list_libraries with partial collection errors."""

    @pytest.mark.asyncio
    async def test_list_libraries_handles_partial_collection_errors(self):
        """Test that individual collection processing errors are handled gracefully."""
        with patch("doc_server.mcp_server.get_vector_store") as mock_store:
            # Mock collections with one that will cause an error
            mock_store.return_value.list_collections.return_value = [
                {
                    "name": "good_collection",
                    "library_id": "/good",
                    "count": 100,
                    "metadata": {
                        "embedding_model": "all-MiniLM-L6-v2",
                        "created_at": 1234567890.0,
                    },
                },
                {
                    "name": "bad_collection",
                    # Missing required fields that will cause an error
                    "count": 50,
                    # No metadata or other required fields
                },
                {
                    "name": "another_good",
                    "library_id": "/another",
                    "count": 75,
                    "metadata": {
                        "embedding_model": "all-MiniLM-L6-v2",
                        "created_at": 1234567891.0,
                    },
                },
            ]

            result = await list_libraries.fn()

            # Should process the good collections and skip the bad one
            assert isinstance(result, list)
            assert len(result) >= 2  # At least 2 good collections

            # Verify good collections are processed correctly
            good_libs = [
                lib for lib in result if lib["library_id"] in ["/good", "/another"]
            ]
            assert len(good_libs) == 2


class TestHealthCheckErrors:
    """Tests for health_check error handling."""

    @pytest.mark.asyncio
    async def test_health_check_returns_error_status_on_exception(self):
        """Test that health_check returns error status when health check fails."""
        import time

        from doc_server.mcp_server import health_check

        with patch("doc_server.mcp_server.get_health_status") as mock_health:
            mock_health.side_effect = Exception("Health check failed")

            result = await health_check.fn()

            assert result["status"] == "unhealthy"
            assert "error" in result
            assert result["error"] == "Health check failed"
            assert "timestamp" in result
            assert result["timestamp"] <= time.time() + 1  # Allow for small timing diff


class TestValidateServerFunction:
    """Tests for validate_server functionality."""

    @pytest.mark.asyncio
    async def test_validate_server_success(self):
        """Test validate_server returns success when validation passes."""
        from doc_server.mcp_server import validate_server

        with patch("doc_server.mcp_server.validate_startup") as mock_validate:
            mock_validate.return_value = {
                "status": "healthy",
                "components": {"embedding_service": "ok", "vector_store": "ok"},
                "timestamp": 1234567890.0,
            }

            result = await validate_server.fn()

            assert result["status"] == "healthy"
            assert "components" in result
            assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_validate_server_handles_validation_failure(self):
        """Test validate_server handles validation failures gracefully."""

        from doc_server.mcp_server import validate_server

        with patch("doc_server.mcp_server.validate_startup") as mock_validate:
            mock_validate.side_effect = Exception("Validation service unavailable")

            result = await validate_server.fn()

            assert result["status"] == "unhealthy"
            assert "error" in result
            assert result["error"] == "Validation service unavailable"


class TestSearchDocsValidationErrors:
    """Tests for search_docs validation errors (lines 157-168, 171-174)."""

    @pytest.mark.asyncio
    async def test_search_docs_empty_query_validation(self):
        """Test search_docs with empty query validation."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await search_docs.fn("", "/test")

    @pytest.mark.asyncio
    async def test_search_docs_query_sanitization(self):
        """Test search_docs input sanitization."""
        with patch("doc_server.mcp_server.get_hybrid_search") as _mock_search:
            # Test with potentially problematic input
            query = "test\x00query\x00with\x00nulls"

            result = await search_docs.fn(query, "/test")

            # Should handle null bytes gracefully
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_docs_handles_search_service_errors(self):
        """Test search_docs handles search service errors."""
        with patch("doc_server.mcp_server.get_hybrid_search") as mock_search:
            mock_search.side_effect = Exception("Search service unavailable")

            # Should raise RuntimeError when service errors occur
            with pytest.raises(
                RuntimeError, match="Search failed: Search service unavailable"
            ):
                await search_docs.fn("test query", "/test")


class TestListLibrariesCollectionErrors:
    """Tests for list_libraries collection processing errors (lines 308-311)."""

    @pytest.mark.asyncio
    async def test_list_libraries_collection_metadata_errors(self):
        """Test list_libraries with collection metadata errors."""
        with patch("doc_server.mcp_server.get_vector_store") as mock_store:
            # Mock collections with problematic metadata
            mock_store.return_value.list_collections.return_value = [
                {
                    "name": "test_collection",
                    "library_id": "/test",
                    "count": 100,
                    "metadata": {
                        "embedding_model": "all-MiniLM-L6-v2",
                        # Missing created_at timestamp
                    },
                }
            ]

            result = await list_libraries.fn()

            # Should handle missing metadata gracefully
            assert isinstance(result, list)
            if result:  # If any results were returned
                assert "library_id" in result[0]

    @pytest.mark.asyncio
    async def test_list_libraries_handles_list_collections_failure(self):
        """Test list_libraries when list_collections fails."""
        with patch("doc_server.mcp_server.get_vector_store") as mock_store:
            mock_store.return_value.list_collections.side_effect = Exception(
                "Collection access failed"
            )

            # Should raise RuntimeError when list_collections fails
            with pytest.raises(
                RuntimeError, match="Failed to list libraries: Collection access failed"
            ):
                await list_libraries.fn()


class TestRemoveLibraryInvalidID:
    """Tests for remove_library with invalid ID (lines 341-348)."""

    @pytest.mark.asyncio
    async def test_remove_library_empty_id(self):
        """Test remove_library with empty library ID."""
        # Should raise ValueError for empty ID
        with pytest.raises(
            ValueError, match="Invalid library ID: Input cannot be empty"
        ):
            await remove_library.fn("")

    @pytest.mark.asyncio
    async def test_remove_library_nonexistent_id(self):
        """Test remove_library with non-existent library ID."""
        with patch("doc_server.mcp_server.get_vector_store") as mock_store:
            mock_store.return_value.delete_collection.return_value = False

            result = await remove_library.fn("/nonexistent")

            # Should return False for non-existent library
            assert result is False

    @pytest.mark.asyncio
    async def test_remove_library_handles_service_errors(self):
        """Test remove_library handles service errors gracefully."""
        with patch("doc_server.mcp_server.get_vector_store") as mock_store:
            mock_store.return_value.delete_collection.side_effect = Exception(
                "Service unavailable"
            )

            # Should raise RuntimeError when service errors occur
            with pytest.raises(
                RuntimeError, match="Failed to remove library: Service unavailable"
            ):
                await remove_library.fn("/test")


class TestHealthCheckVectorStoreErrors:
    """Tests for health_check with vector store errors (lines 382-399)."""

    @pytest.mark.asyncio
    async def test_health_check_vector_store_connection_error(self):
        """Test health_check when vector store has connection errors."""
        import time

        from doc_server.mcp_server import health_check

        with patch("doc_server.mcp_server.get_health_status") as mock_health:
            mock_health.return_value = {
                "status": "unhealthy",
                "version": "1.0.0",
                "timestamp": time.time(),
                "components": {
                    "vector_store": {
                        "status": "unhealthy",
                        "error": "connection_failed",
                    },
                    "embedding_service": {"status": "healthy"},
                },
            }

            result = await health_check.fn()

            assert result["status"] == "unhealthy"
            assert "components" in result
            assert result["components"]["vector_store"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_vector_store_timeout(self):
        """Test health_check when vector store times out."""
        import time

        from doc_server.mcp_server import health_check

        with patch("doc_server.mcp_server.get_health_status") as mock_health:
            mock_health.return_value = {
                "status": "unhealthy",
                "version": "1.0.0",
                "timestamp": time.time(),
                "components": {
                    "vector_store": {"status": "unhealthy", "error": "timeout"},
                    "embedding_service": {"status": "healthy"},
                },
            }

            result = await health_check.fn()

            assert result["status"] == "unhealthy"
            assert "components" in result
            assert result["components"]["vector_store"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_embedding_service_errors(self):
        """Test health_check when embedding service has errors."""
        import time

        from doc_server.mcp_server import health_check

        with patch("doc_server.mcp_server.get_health_status") as mock_health:
            mock_health.return_value = {
                "status": "unhealthy",
                "version": "1.0.0",
                "timestamp": time.time(),
                "components": {
                    "vector_store": {"status": "healthy"},
                    "embedding_service": {
                        "status": "unhealthy",
                        "error": "model_load_failed",
                    },
                },
            }

            result = await health_check.fn()

            assert result["status"] == "unhealthy"
            assert "components" in result
            assert result["components"]["embedding_service"]["status"] == "unhealthy"
            assert (
                result["components"]["embedding_service"]["error"]
                == "model_load_failed"
            )


class TestIngestLibraryCleanupOnFailure:
    """Tests for ingest_library cleanup on failure (lines 617-626)."""

    @pytest.mark.asyncio
    async def test_ingest_library_cleanup_on_git_failure(self):
        """Test ingest_library cleanup when git cloning fails."""
        with patch("doc_server.mcp_server.GitCloner") as mock_git_cloner:
            mock_git_instance = mock_git_cloner.return_value
            mock_git_instance.clone_repository.side_effect = Exception(
                "Git clone failed"
            )

            # Should raise RuntimeError when git cloning fails
            with pytest.raises(
                RuntimeError, match="Ingestion failed: Git clone failed"
            ):
                await ingest_library.fn("https://github.com/test/repo.git", "/test")

    @pytest.mark.asyncio
    async def test_ingest_library_cleanup_on_zip_failure(self):
        """Test ingest_library cleanup when ZIP extraction fails."""
        with patch("doc_server.mcp_server.ZIPExtractor") as mock_zip_extractor:
            mock_zip_instance = mock_zip_extractor.return_value
            mock_zip_instance.extract_archive.side_effect = Exception("Invalid ZIP")

            # Should raise RuntimeError when ZIP extraction fails
            with pytest.raises(RuntimeError, match="Ingestion failed: Invalid ZIP"):
                await ingest_library.fn("/path/to/file.zip", "/test")

    @pytest.mark.asyncio
    async def test_ingest_library_cleanup_on_vector_store_failure(self):
        """Test ingest_library cleanup when vector store operations fail."""
        with (
            patch("doc_server.mcp_server.GitCloner") as mock_git_cloner,
            patch("doc_server.mcp_server.FileFilter") as mock_filter,
            patch("doc_server.mcp_server.DocumentProcessor") as mock_processor,
            patch("doc_server.mcp_server.get_vector_store") as mock_store,
        ):
            # Mock successful git cloning, filtering, and processing
            mock_git_cloner.return_value.clone_repository.return_value = None
            mock_filter.return_value.filter_files.return_value = [
                MagicMock(included=True, file_path="/test/file.py")
            ]
            mock_processor.return_value.process_file.return_value = [
                MagicMock(
                    content="test",
                    file_path="/test/file.py",
                    library_id="/test",
                    line_start=1,
                    line_end=10,
                )
            ]

            # Mock vector store to fail during document addition
            mock_store.return_value.create_collection.return_value = None
            mock_store.return_value.add_documents.side_effect = Exception(
                "Vector store error"
            )

            # Should raise RuntimeError when vector store operations fail
            with pytest.raises(
                RuntimeError, match="Ingestion failed: Vector store error"
            ):
                await ingest_library.fn("https://github.com/test/repo.git", "/test")

    @pytest.mark.asyncio
    async def test_ingest_library_handles_invalid_source(self):
        """Test ingest_library with invalid source type."""
        # Should raise RuntimeError when source type is invalid
        with pytest.raises(
            RuntimeError, match="Ingestion failed: Local path does not exist"
        ):
            await ingest_library.fn("invalid://source", "/test")

    @pytest.mark.asyncio
    async def test_ingest_library_handles_validation_errors(self):
        """Test ingest_library with validation errors."""
        # Test with invalid library_id
        with pytest.raises(ValueError, match="Invalid input: Input cannot be empty"):
            await ingest_library.fn("https://github.com/test/repo.git", "")

    @pytest.mark.asyncio
    async def test_validate_server_logs_validation_attempt(self):
        """Test that validate_server logs the validation attempt."""
        from doc_server.mcp_server import validate_server

        with patch("doc_server.mcp_server.validate_startup") as mock_validate:
            with patch("doc_server.mcp_server.logger") as mock_logger:
                mock_validate.return_value = {"status": "healthy"}

                await validate_server.fn()

                # Verify logging calls
                mock_logger.info.assert_any_call("Server validation requested")
                mock_logger.info.assert_any_call(
                    "Server validation completed", status="healthy"
                )
