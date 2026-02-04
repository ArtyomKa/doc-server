"""
Integration tests for doc-server Phase 6.2.

Tests cover:
- End-to-end ingestion workflow
- Search accuracy validation
- Performance benchmarking
- Error scenarios and edge cases
- Multiple library handling
"""

import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from doc_server.config import Settings
from doc_server.ingestion.document_processor import DocumentProcessor, DocumentChunk
from doc_server.ingestion.file_filter import FileFilter
from doc_server.logging_config import configure_structlog
from doc_server.mcp_server import DocumentResult
from doc_server.search.vector_store import ChromaVectorStore


# Configure structlog for tests
configure_structlog()


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_storage_dir: Path) -> Settings:
    """Create Settings with temporary storage."""
    return Settings(storage_path=temp_storage_dir)


@pytest.fixture
def document_processor(config: Settings) -> DocumentProcessor:
    """Create DocumentProcessor instance."""
    return DocumentProcessor(config=config)


@pytest.fixture
def file_filter(config: Settings) -> FileFilter:
    """Create FileFilter instance."""
    return FileFilter(config=config)


class TestEndToEndIngestion:
    """Tests for end-to-end ingestion workflow (AC-6.2.1)."""

    def test_document_processor_file_processing(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test complete document processing workflow."""
        # Create test file
        test_file = temp_storage_dir / "test.md"
        test_file.write_text("# Test Document\n\nThis is test content for processing.")

        # Process file
        chunks = document_processor.process_file(
            file_path=test_file, library_id="/test-library"
        )

        assert len(chunks) > 0
        assert isinstance(chunks[0], DocumentChunk)
        assert "/test-library" in chunks[0].library_id
        assert "Test Document" in chunks[0].content

    def test_document_processor_with_large_file(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test document processing with large file (>2KB threshold)."""
        # Create large test file
        test_file = temp_storage_dir / "large.md"
        large_content = "# Large Document\n\n" + "x" * 5000
        test_file.write_text(large_content)

        # Process file - should create multiple chunks
        chunks = document_processor.process_file(
            file_path=test_file, library_id="/test-library"
        )

        # Large files should be chunked
        assert len(chunks) > 1

        # Each chunk should have valid line numbers
        for chunk in chunks:
            assert chunk.line_start < chunk.line_end

    def test_file_filter_allowed_extensions(
        self, file_filter: FileFilter, temp_storage_dir: Path
    ):
        """Test that file filter correctly handles allowed extensions."""
        # Create test files with different extensions
        test_files = {
            "test.py": "print('hello')",
            "test.md": "# Markdown",
            "test.txt": "Plain text",
            "test.cpp": "// C++ code",
            "test.bin": b"\x00\x01\x02",
            "test.jpg": b"\xff\xd8\xff",  # JPEG magic bytes
        }

        for filename, content in test_files.items():
            test_file = temp_storage_dir / filename
            if isinstance(content, bytes):
                test_file.write_bytes(content)
            else:
                test_file.write_text(content)

        # Test filtering
        results = []
        for filename in test_files.keys():
            test_file = temp_storage_dir / filename
            result = file_filter.should_include_file(str(test_file))
            results.append((filename, result))

        # Check results
        allowed = [name for name, included in results if included]

        # Should allow text files with allowed extensions
        assert "test.py" in allowed
        assert "test.md" in allowed
        assert "test.txt" in allowed
        assert "test.cpp" in allowed

        # Should reject binary files and disallowed extensions
        assert "test.bin" not in allowed
        assert "test.jpg" not in allowed

    def test_file_filter_binary_detection(
        self, file_filter: FileFilter, temp_storage_dir: Path
    ):
        """Test binary file detection."""
        # Create binary file with null bytes
        binary_file = temp_storage_dir / "binary.dat"
        binary_file.write_bytes(b"\x00\x01\x02\x03binary content")

        # Should be filtered out
        result = file_filter.should_include_file(str(binary_file))
        assert result is False

    def test_file_filter_size_limit(
        self, file_filter: FileFilter, temp_storage_dir: Path
    ):
        """Test file size limit enforcement."""
        # Create large file (>1MB limit)
        large_file = temp_storage_dir / "large.txt"
        large_file.write_text("x" * (1024 * 1024 + 1))

        # Should be filtered out
        result = file_filter.should_include_file(str(large_file))
        assert result is False

    def test_file_filter_directory(
        self, file_filter: FileFilter, temp_storage_dir: Path
    ):
        """Test filtering an entire directory."""
        # Create test directory structure
        test_dir = temp_storage_dir / "test_repo"
        test_dir.mkdir()

        # Create various files
        (test_dir / "README.md").write_text("# README\n\nContent")
        (test_dir / "main.py").write_text("print('hello')")
        (test_dir / "data.bin").write_bytes(b"\x00\x01\x02")
        (test_dir / "large.txt").write_text("x" * (1024 * 1024 + 1))

        # Filter directory
        results = file_filter.filter_directory(str(test_dir))

        # Should have filtered results
        assert len(results) >= 4

        # Check that allowed files were included
        included = [r for r in results if r.included]
        assert len(included) >= 2  # README.md and main.py should be included


class TestDocumentProcessorEdgeCases:
    """Additional edge case tests for document processor."""

    def test_empty_file(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test handling of empty files."""
        test_file = temp_storage_dir / "empty.md"
        test_file.write_text("")

        chunks = document_processor.process_file(
            file_path=test_file, library_id="/test"
        )

        assert len(chunks) == 0

    def test_single_line_file(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test handling of single line files."""
        test_file = temp_storage_dir / "single.md"
        test_file.write_text("# Title\n\nSingle line of content.")

        chunks = document_processor.process_file(
            file_path=test_file, library_id="/test"
        )

        assert len(chunks) == 1
        assert "Title" in chunks[0].content

    def test_metadata_header_preserved(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test that metadata headers are preserved in chunks."""
        test_file = temp_storage_dir / "metadata.md"
        test_file.write_text("# Document\n\nContent here.")

        chunks = document_processor.process_file(
            file_path=test_file, library_id="/my-library"
        )

        assert len(chunks) > 0
        chunk = chunks[0]
        assert "file_path:" in chunk.content
        assert "library_id: /my-library" in chunk.content
        assert "lines:" in chunk.content


class TestFileFilterEdgeCases:
    """Additional edge case tests for file filter."""

    def test_nonexistent_file(self, file_filter: FileFilter, temp_storage_dir: Path):
        """Test handling of nonexistent files."""
        result = file_filter.should_include_file(
            str(temp_storage_dir / "nonexistent.txt")
        )
        assert result is False

    def test_directory_path(self, file_filter: FileFilter, temp_storage_dir: Path):
        """Test that directories are filtered out."""
        result = file_filter.should_include_file(str(temp_storage_dir))
        assert result is False

    def test_unicode_content(self, file_filter: FileFilter, temp_storage_dir: Path):
        """Test handling of unicode content in files."""
        test_file = temp_storage_dir / "unicode.md"
        test_file.write_text("# Unicode Test\n\nCafé résumé 🌍 émojis")

        # Should be allowed (unicode text files are text, not binary)
        result = file_filter.should_include_file(str(test_file))
        # Note: This may or may not be allowed depending on null byte detection
        # Unicode files don't contain null bytes, so they should be allowed
        assert result is True

    def test_gitignore_patterns(self, file_filter: FileFilter, temp_storage_dir: Path):
        """Test gitignore pattern loading and filtering."""
        # Create a gitignore file
        gitignore_file = temp_storage_dir / ".gitignore"
        gitignore_file.write_text("*.log\nbuild/\n*.pyc\n")

        # Load gitignore
        gitignore = file_filter.load_gitignore(str(gitignore_file))

        assert gitignore is not None

        # Create test files
        (temp_storage_dir / "test.md").write_text("content")
        (temp_storage_dir / "debug.log").write_text("log")

        # Check if gitignore works - *.log should match debug.log
        assert gitignore.match_file("debug.log"), "*.log should match debug.log"
        assert not gitignore.match_file("README.md"), "*.log should not match README.md"


class TestErrorScenarios:
    """Tests for error handling and edge cases (AC-6.2.4)."""

    def test_document_result_error_handling(self):
        """Test DocumentResult handles edge cases."""
        # Test with very long content
        long_content = "x" * 100000

        result = DocumentResult(
            content=long_content,
            file_path="/test/long.py",
            library_id="/test",
            relevance_score=1.0,
        )

        assert result.content == long_content
        assert result.relevance_score == 1.0

        # Test to_dict conversion
        result_dict = result.to_dict()
        assert result_dict["content"] == long_content

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

        result_dict = result.to_dict()
        assert result_dict["line_numbers"] == (10, 20)

    def test_document_result_with_metadata(self):
        """Test DocumentResult with custom metadata."""
        metadata = {"author": "test", "version": "1.0"}
        result = DocumentResult(
            content="Test content",
            file_path="/test/file.py",
            library_id="/test",
            relevance_score=0.75,
            metadata=metadata,
        )

        assert result.metadata == metadata

        result_dict = result.to_dict()
        assert result_dict["metadata"] == metadata


class TestVectorStoreIntegration:
    """Integration tests for vector store."""

    def test_vector_store_initialization(
        self, config: Settings, temp_storage_dir: Path
    ):
        """Test vector store initialization."""
        with patch("doc_server.search.vector_store.get_embedding_service"):
            # Test that vector store can be initialized
            vector_store = ChromaVectorStore(persist_directory=temp_storage_dir)

            assert vector_store is not None
            assert vector_store.persist_directory == temp_storage_dir

    def test_collection_creation(self, config: Settings, temp_storage_dir: Path):
        """Test collection creation."""
        with patch(
            "doc_server.search.vector_store.get_embedding_service"
        ) as mock_get_emb:
            # Setup mock embedding service
            mock_emb = MagicMock()
            mock_emb.model_name = "test-model"
            mock_emb.embedding_dimension = 384
            mock_get_emb.return_value = mock_emb

            vector_store = ChromaVectorStore(persist_directory=temp_storage_dir)

            # Create collection
            collection = vector_store.create_collection(library_id="/test-lib")

            assert collection is not None

    def test_collection_listing(self, config: Settings, temp_storage_dir: Path):
        """Test listing collections."""
        with patch(
            "doc_server.search.vector_store.get_embedding_service"
        ) as mock_get_emb:
            mock_emb = MagicMock()
            mock_emb.model_name = "test-model"
            mock_emb.embedding_dimension = 384
            mock_get_emb.return_value = mock_emb

            vector_store = ChromaVectorStore(persist_directory=temp_storage_dir)

            # List collections (may be empty initially)
            collections = vector_store.list_collections()

            assert isinstance(collections, list)

    def test_collection_deletion(self, config: Settings, temp_storage_dir: Path):
        """Test collection deletion."""
        with patch(
            "doc_server.search.vector_store.get_embedding_service"
        ) as mock_get_emb:
            mock_emb = MagicMock()
            mock_emb.model_name = "test-model"
            mock_emb.embedding_dimension = 384
            mock_get_emb.return_value = mock_emb

            vector_store = ChromaVectorStore(persist_directory=temp_storage_dir)

            # Create collection
            vector_store.create_collection(library_id="/delete-test")

            # Delete collection
            result = vector_store.delete_collection(library_id="/delete-test")

            assert result is True


class TestPerformanceBenchmarks:
    """Performance benchmarking tests (AC-6.2.3)."""

    def test_document_processing_throughput(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test document processing performance."""
        num_docs = 50

        # Create test files
        test_files = []
        for i in range(num_docs):
            test_file = temp_storage_dir / f"test_{i}.md"
            test_file.write_text(
                f"# Test Document {i}\n\nThis is test content for benchmarking document {i}."
            )
            test_files.append(test_file)

        start_time = time.time()

        # Process all files
        total_chunks = 0
        for test_file in test_files:
            chunks = document_processor.process_file(
                file_path=test_file, library_id="/benchmark"
            )
            total_chunks += len(chunks)

        end_time = time.time()
        elapsed = end_time - start_time

        # Calculate throughput (docs per minute)
        throughput = (num_docs / elapsed) * 60

        # Should be well above baseline
        assert throughput > 10, f"Throughput {throughput:.1f} docs/min below baseline"

        print(f"Document processing throughput: {throughput:.1f} docs/min")

    def test_large_file_chunking(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test chunking performance with large files."""
        # Create a very large test file with many lines to trigger chunking
        lines = ["# Very Large Document\n"]
        for i in range(1000):
            lines.append(f"Line {i}: {'x' * 100}\n")
        large_content = "".join(lines)

        test_file = temp_storage_dir / "very_large.md"
        test_file.write_text(large_content)

        start_time = time.time()

        # Process large file
        chunks = document_processor.process_file(
            file_path=test_file, library_id="/benchmark"
        )

        end_time = time.time()
        elapsed = end_time - start_time

        # Should create multiple chunks
        assert len(chunks) > 1, f"Expected multiple chunks but got {len(chunks)}"

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0

        print(f"Large file chunking: {len(chunks)} chunks in {elapsed:.2f}s")

    def test_directory_filtering_performance(
        self, file_filter: FileFilter, temp_storage_dir: Path
    ):
        """Test directory filtering performance."""
        # Create many test files
        num_files = 100
        for i in range(num_files):
            test_file = temp_storage_dir / f"file_{i}.md"
            test_file.write_text(f"# File {i}\n\nContent for file {i}.")

        start_time = time.time()

        # Filter directory
        results = file_filter.filter_directory(str(temp_storage_dir))

        end_time = time.time()
        elapsed = end_time - start_time

        # Should process all files
        assert len(results) >= num_files

        # Should complete in reasonable time (< 10 seconds)
        assert elapsed < 10.0

        print(f"Directory filtering: {len(results)} files in {elapsed:.2f}s")


class TestSpecialCharacters:
    """Tests for special character handling."""

    def test_special_characters_in_content(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test handling of special characters in document content."""
        # Content with various special characters
        special_content = """
        # Special Characters Test
        
        ```python
        def test():
            print("Hello, World! 🌍")
            return "Café résumé"
        ```
        
        - Mathematical: ∑ ∏ ∫ √ π
        - Code: `var_name`, {dict_key}, [list_item]
        """

        # Create test file
        test_file = temp_storage_dir / "special.md"
        test_file.write_text(special_content)

        # Process file
        chunks = document_processor.process_file(
            file_path=test_file, library_id="/test"
        )

        # Content should be preserved
        assert len(chunks) > 0
        assert "Special Characters Test" in chunks[0].content

    def test_whitespace_handling(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test handling of excessive whitespace."""
        test_file = temp_storage_dir / "whitespace.md"
        test_file.write_text("Line 1\n\n\n\n\n\nLine 2\n\n\n\n\n\nLine 3")

        chunks = document_processor.process_file(
            file_path=test_file, library_id="/test"
        )

        # Should handle excessive whitespace
        assert len(chunks) > 0

    def test_tabs_and_indentation(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test handling of tabs and indentation."""
        test_file = temp_storage_dir / "indentation.py"
        test_file.write_text("""def test():
    if True:
        print("indented")
    else:
        return None
""")

        chunks = document_processor.process_file(
            file_path=test_file, library_id="/test"
        )

        # Should preserve indentation
        assert len(chunks) > 0
        assert "indented" in chunks[0].content


class TestEdgeCases:
    """Additional edge case tests."""

    def test_very_short_content(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test handling of very short document content."""
        # Very short content
        test_file = temp_storage_dir / "short.md"
        test_file.write_text("x")

        # Should still process correctly
        chunks = document_processor.process_file(
            file_path=test_file, library_id="/test"
        )

        assert len(chunks) > 0

    def test_very_long_content(
        self, document_processor: DocumentProcessor, temp_storage_dir: Path
    ):
        """Test handling of very long document content."""
        # Very long content (>2KB threshold for chunking)
        test_file = temp_storage_dir / "long.md"
        test_file.write_text("y" * 5000)

        # Should be chunked if exceeds threshold
        chunks = document_processor.process_file(
            file_path=test_file, library_id="/test"
        )

        assert len(chunks) > 0

    def test_nonexistent_library_operations(
        self, config: Settings, temp_storage_dir: Path
    ):
        """Test operations on nonexistent library."""
        with patch(
            "doc_server.search.vector_store.get_embedding_service"
        ) as mock_get_emb:
            mock_emb = MagicMock()
            mock_emb.model_name = "test-model"
            mock_emb.embedding_dimension = 384
            mock_get_emb.return_value = mock_emb

            vector_store = ChromaVectorStore(persist_directory=temp_storage_dir)

            # Querying nonexistent collection should raise CollectionNotFoundError
            with pytest.raises(Exception):
                vector_store.query_documents(
                    library_id="/non_existent", query_texts=["test"]
                )

    def test_empty_metadata(self, file_filter: FileFilter, temp_storage_dir: Path):
        """Test handling of empty metadata in file operations."""
        # Create a simple file
        test_file = temp_storage_dir / "simple.md"
        test_file.write_text("# Simple\n\nContent")

        # Filter file with no special metadata
        result = file_filter.should_include_file(str(test_file))

        # Should be included
        assert result is True


class TestMultipleLibraryHandling:
    """Tests for handling multiple libraries."""

    def test_multiple_collection_operations(
        self, config: Settings, temp_storage_dir: Path
    ):
        """Test creating and managing multiple collections."""
        with patch(
            "doc_server.search.vector_store.get_embedding_service"
        ) as mock_get_emb:
            mock_emb = MagicMock()
            mock_emb.model_name = "test-model"
            mock_emb.embedding_dimension = 384
            mock_get_emb.return_value = mock_emb

            vector_store = ChromaVectorStore(persist_directory=temp_storage_dir)

            # Create collections for different libraries
            vector_store.create_collection(library_id="/pandas")
            vector_store.create_collection(library_id="/fastapi")
            vector_store.create_collection(library_id="/algorithms")

            # List collections
            collections = vector_store.list_collections()

            # Should have at least 3 collections
            assert len(collections) >= 3

            # Verify we can find our collections
            collection_names = [c["name"] for c in collections]
            lib_ids = [c.get("library_id", "") for c in collections]

            # At least some should match our libraries
            found_count = sum(
                1
                for lib_id in lib_ids
                if "pandas" in lib_id or "fastapi" in lib_id or "algorithms" in lib_id
            )
            assert found_count >= 2
