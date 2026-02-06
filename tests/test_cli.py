"""
Tests for CLI commands.

Tests all CLI commands: ingest, search, list, remove, serve, health.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from doc_server.cli import cli


@pytest.fixture
def runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestCLIHelp:
    """Test CLI help and documentation."""

    def test_main_help(self, runner):
        """Test main CLI help displays correctly."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Doc Server CLI" in result.output
        assert "ingest" in result.output
        assert "search" in result.output
        assert "list" in result.output
        assert "remove" in result.output
        assert "serve" in result.output

    def test_version_flag(self, runner):
        """Test version flag displays version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_verbose_flag(self, runner):
        """Test verbose flag is accepted."""
        result = runner.invoke(cli, ["--verbose", "list"])
        # list might fail due to no libraries, but verbose should be accepted
        assert result.exit_code in [0, 1]


class TestIngestCommand:
    """Test ingest command."""

    @patch("doc_server.cli.GitCloner")
    @patch("doc_server.cli.FileFilter")
    @patch("doc_server.cli.DocumentProcessor")
    @patch("doc_server.cli.get_vector_store")
    def test_ingest_git_source(
        self,
        mock_get_vector_store,
        mock_processor,
        mock_filter,
        mock_git_cloner,
        runner,
    ):
        """Test ingest command with git source."""
        # Setup mocks
        mock_git_instance = MagicMock()
        mock_git_cloner.return_value = mock_git_instance

        mock_filter_instance = MagicMock()
        mock_filter.return_value = mock_filter_instance
        mock_filter_instance.filter_files.return_value = [
            MagicMock(file_path="/tmp/test/file.md", included=True)
        ]

        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        mock_chunk = MagicMock()
        mock_chunk.content = "Test content"
        mock_chunk.file_path = "file.md"
        mock_chunk.library_id = "/test"
        mock_chunk.line_start = 1
        mock_chunk.line_end = 10
        mock_processor_instance.process_file.return_value = [mock_chunk]

        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.add_documents.return_value = ["id1"]

        result = runner.invoke(
            cli,
            [
                "ingest",
                "--source",
                "https://github.com/test/repo",
                "--library-id",
                "/test",
            ],
        )

        assert result.exit_code == 0
        assert (
            "Successfully ingested" in result.output
            or "Ingesting documentation" in result.output
        )

    @patch("doc_server.cli.Path")
    def test_ingest_local_source_not_exist(self, mock_path, runner):
        """Test ingest with non-existent local source."""
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.resolve.return_value = mock_path_instance
        mock_path_instance.exists.return_value = False

        result = runner.invoke(
            cli,
            ["ingest", "--source", "/nonexistent/path", "--library-id", "/test"],
        )

        assert result.exit_code == 1

    def test_ingest_missing_source(self, runner):
        """Test ingest fails without source."""
        result = runner.invoke(cli, ["ingest", "--library-id", "/test"])
        assert result.exit_code != 0
        assert (
            "--source" in result.output
            or "Missing option" in result.output
            or "required" in result.output.lower()
        )

    def test_ingest_missing_library_id(self, runner):
        """Test ingest fails without library_id."""
        result = runner.invoke(
            cli, ["ingest", "--source", "https://github.com/test/repo"]
        )
        assert result.exit_code != 0
        assert (
            "--library-id" in result.output
            or "Missing option" in result.output
            or "required" in result.output.lower()
        )


class TestSearchCommand:
    """Test search command."""

    @patch("doc_server.cli.get_hybrid_search")
    def test_search_basic(self, mock_get_search, runner):
        """Test basic search command."""
        mock_search = MagicMock()
        mock_get_search.return_value = mock_search

        # Create mock result with proper attributes
        mock_result = MagicMock()
        mock_result.content = "Test content about pandas"
        mock_result.file_path = "docs/pandas.md"
        mock_result.library_id = "/pandas"
        mock_result.relevance_score = 0.95
        mock_result.line_numbers = (10, 20)
        mock_result.metadata = {}
        mock_search.search.return_value = [mock_result]

        result = runner.invoke(
            cli, ["search", "--query", "pandas read_csv", "--library-id", "/pandas"]
        )

        assert result.exit_code == 0
        mock_search.search.assert_called_once()

    @patch("doc_server.cli.get_hybrid_search")
    def test_search_no_results(self, mock_get_search, runner):
        """Test search with no results."""
        mock_search = MagicMock()
        mock_get_search.return_value = mock_search
        mock_search.search.return_value = []

        result = runner.invoke(
            cli, ["search", "--query", "nonexistent", "--library-id", "/test"]
        )

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("doc_server.cli.get_hybrid_search")
    def test_search_json_format(self, mock_get_search, runner):
        """Test search with JSON output format."""
        mock_search = MagicMock()
        mock_get_search.return_value = mock_search

        mock_result = MagicMock()
        mock_result.content = "Test content"
        mock_result.file_path = "test.md"
        mock_result.library_id = "/test"
        mock_result.relevance_score = 0.8
        mock_result.line_numbers = None
        mock_result.metadata = {}
        mock_search.search.return_value = [mock_result]

        result = runner.invoke(
            cli,
            [
                "search",
                "--query",
                "test",
                "--library-id",
                "/test",
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0
        # Verify JSON output
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["file_path"] == "test.md"

    @patch("doc_server.cli.get_hybrid_search")
    def test_search_simple_format(self, mock_get_search, runner):
        """Test search with simple output format."""
        mock_search = MagicMock()
        mock_get_search.return_value = mock_search

        mock_result = MagicMock()
        mock_result.file_path = "test.md"
        mock_result.relevance_score = 0.8
        mock_result.content = "Test content"
        mock_search.search.return_value = [mock_result]

        result = runner.invoke(
            cli,
            [
                "search",
                "--query",
                "test",
                "--library-id",
                "/test",
                "--format",
                "simple",
            ],
        )

        assert result.exit_code == 0
        assert "test.md" in result.output

    def test_search_empty_query(self, runner):
        """Test search with empty query."""
        result = runner.invoke(cli, ["search", "--query", "", "--library-id", "/test"])

        assert result.exit_code == 1
        assert (
            "Validation error" in result.output
            or "cannot be empty" in result.output.lower()
        )


class TestListCommand:
    """Test list command."""

    @patch("doc_server.cli.get_vector_store")
    def test_list_basic(self, mock_get_vector_store, runner):
        """Test basic list command."""
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.list_collections.return_value = [
            {
                "library_id": "/pandas",
                "name": "pandas",
                "count": 150,
                "metadata": {
                    "embedding_model": "all-MiniLM-L6-v2",
                    "created_at": 1234567890.0,
                },
            },
            {
                "library_id": "/fastapi",
                "name": "fastapi",
                "count": 75,
                "metadata": {
                    "embedding_model": "all-MiniLM-L6-v2",
                    "created_at": 1234567891.0,
                },
            },
        ]

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "/pandas" in result.output or "libraries available" in result.output

    @patch("doc_server.cli.get_vector_store")
    def test_list_empty(self, mock_get_vector_store, runner):
        """Test list with no libraries."""
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.list_collections.return_value = []

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "No libraries found" in result.output

    @patch("doc_server.cli.get_vector_store")
    def test_list_json_format(self, mock_get_vector_store, runner):
        """Test list with JSON output format."""
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.list_collections.return_value = [
            {
                "library_id": "/test",
                "name": "test",
                "count": 10,
                "metadata": {"embedding_model": "model"},
            }
        ]

        result = runner.invoke(cli, ["list", "--format", "json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1

    @patch("doc_server.cli.get_vector_store")
    def test_list_simple_format(self, mock_get_vector_store, runner):
        """Test list with simple output format."""
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.list_collections.return_value = [
            {"library_id": "/test1", "name": "test1", "count": 10, "metadata": {}},
            {"library_id": "/test2", "name": "test2", "count": 20, "metadata": {}},
        ]

        result = runner.invoke(cli, ["list", "--format", "simple"])

        assert result.exit_code == 0


class TestRemoveCommand:
    """Test remove command."""

    @patch("doc_server.cli.get_vector_store")
    def test_remove_basic(self, mock_get_vector_store, runner):
        """Test basic remove command with force."""
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.delete_collection.return_value = True

        result = runner.invoke(cli, ["remove", "--library-id", "/test", "--force"])

        assert result.exit_code == 0
        assert "removed successfully" in result.output
        mock_vector_store.delete_collection.assert_called_once()

    @patch("doc_server.cli.get_vector_store")
    def test_remove_not_exist(self, mock_get_vector_store, runner):
        """Test remove library that doesn't exist."""
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.delete_collection.return_value = False

        result = runner.invoke(
            cli, ["remove", "--library-id", "/nonexistent", "--force"]
        )

        assert result.exit_code == 0
        assert "did not exist" in result.output

    @patch("doc_server.cli.get_vector_store")
    def test_remove_with_confirmation(self, mock_get_vector_store, runner):
        """Test remove with confirmation prompt."""
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.delete_collection.return_value = True

        result = runner.invoke(
            cli,
            ["remove", "--library-id", "/test"],
            input="y\n",
        )

        assert result.exit_code == 0

    @patch("doc_server.cli.get_vector_store")
    def test_remove_cancelled(self, mock_get_vector_store, runner):
        """Test remove with cancelled confirmation."""
        result = runner.invoke(
            cli,
            ["remove", "--library-id", "/test"],
            input="n\n",
        )

        assert result.exit_code == 0
        assert "cancelled" in result.output.lower()


class TestServeCommand:
    """Test serve command."""

    @patch("doc_server.cli.mcp_main")
    def test_serve_basic(self, mock_mcp_main, runner):
        """Test basic serve command."""
        mock_mcp_main.side_effect = KeyboardInterrupt()

        result = runner.invoke(cli, ["serve"])

        assert result.exit_code == 0
        assert "Starting doc-server MCP server" in result.output
        mock_mcp_main.assert_called_once()

    @patch("doc_server.cli.mcp_main")
    def test_serve_sse_transport(self, mock_mcp_main, runner):
        """Test serve with SSE transport."""
        mock_mcp_main.side_effect = KeyboardInterrupt()

        result = runner.invoke(cli, ["serve", "--transport", "sse"])

        assert result.exit_code == 0
        assert "sse" in result.output.lower()

    @patch("doc_server.cli.mcp_main")
    def test_serve_custom_host_port(self, mock_mcp_main, runner):
        """Test serve with custom host and port."""
        mock_mcp_main.side_effect = KeyboardInterrupt()

        result = runner.invoke(
            cli, ["serve", "--transport", "sse", "--host", "0.0.0.0", "--port", "9000"]
        )

        assert result.exit_code == 0
        assert "0.0.0.0" in result.output
        assert "9000" in result.output


class TestHealthCommand:
    """Test health command."""

    @patch("doc_server.cli.get_vector_store")
    def test_health_healthy(self, mock_get_vector_store, runner):
        """Test health command with healthy status."""
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.list_collections.return_value = [
            {"name": "test", "count": 10}
        ]

        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 0
        assert "healthy" in result.output.lower() or "Status:" in result.output

    @patch("doc_server.cli.get_vector_store")
    def test_health_unhealthy(self, mock_get_vector_store, runner):
        """Test health command with unhealthy status."""
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        mock_vector_store.list_collections.side_effect = Exception("Connection failed")

        result = runner.invoke(cli, ["health"])

        assert result.exit_code == 1
        assert "unhealthy" in result.output.lower() or "failed" in result.output.lower()


class TestErrorHandling:
    """Test CLI error handling."""

    def test_unknown_command(self, runner):
        """Test unknown command returns error."""
        result = runner.invoke(cli, ["unknown"])
        assert result.exit_code != 0
        assert "No such command" in result.output or "Usage:" in result.output


class TestColoredOutput:
    """Test colored output functionality."""

    @patch("doc_server.cli.get_hybrid_search")
    def test_search_output(self, mock_get_search, runner):
        """Test search output includes content."""
        mock_search = MagicMock()
        mock_get_search.return_value = mock_search

        mock_result = MagicMock()
        mock_result.content = "Test content"
        mock_result.file_path = "test.md"
        mock_result.library_id = "/test"
        mock_result.relevance_score = 0.8
        mock_result.line_numbers = (1, 10)
        mock_result.metadata = {}
        mock_search.search.return_value = [mock_result]

        result = runner.invoke(
            cli, ["search", "--query", "test", "--library-id", "/test"]
        )

        assert result.exit_code == 0
        assert "test.md" in result.output
