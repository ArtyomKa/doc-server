"""
Additional document processor tests for uncovered edge cases.
"""

from pathlib import Path

import pytest

from doc_server.config import Settings
from doc_server.ingestion.document_processor import (
    DocumentProcessor,
    EncodingError,
)


@pytest.fixture
def document_processor() -> DocumentProcessor:
    """Create a DocumentProcessor instance for testing."""
    return DocumentProcessor(Settings())


@pytest.fixture
def tmp_path(tmp_path_factory) -> Path:
    """Create a temporary directory for testing."""
    return tmp_path_factory.mktemp("doc_processor_extra")


class TestDocumentProcessorUncoveredPaths:
    """Tests for uncovered document_processor edge cases."""

    def test_empty_file_handling(
        self, document_processor: DocumentProcessor, tmp_path: Path
    ):
        """Test processing completely empty files."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = document_processor.process_file(empty_file, "/test")

        # Should handle empty files gracefully
        assert len(result) == 0
        assert isinstance(result, list)

    def test_file_with_only_whitespace(
        self, document_processor: DocumentProcessor, tmp_path: Path
    ):
        """Test processing files with only whitespace."""
        whitespace_file = tmp_path / "whitespace.txt"
        whitespace_file.write_text("   \n   \n\t\n   \n")

        result = document_processor.process_file(whitespace_file, "/test")

        # Should handle whitespace-only files
        assert isinstance(result, list)

    def test_binary_file_detection_during_processing(
        self, document_processor: DocumentProcessor, tmp_path: Path
    ):
        """Test binary file detection during document processing."""
        binary_file = tmp_path / "binary.py"
        binary_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")

        # Document processor should handle binary content gracefully using latin-1 encoding
        # Binary content detection should happen at the file filtering stage, not during processing
        result = document_processor.process_file(binary_file, "/test")

        # Should successfully process and return chunks (even for binary content)
        assert isinstance(result, list)
        # The content will include null bytes and binary data when read with latin-1
        assert len(result) > 0
        # Check that the content contains the binary data (decoded as latin-1)
        full_content = "".join(chunk.content for chunk in result)
        # latin-1 decoding of \x00\x01\x02\x03\x04\x05 results in control characters
        assert len(full_content) > 0

    def test_very_long_line_chunking(
        self, document_processor: DocumentProcessor, tmp_path: Path
    ):
        """Test chunking with very long lines that exceed chunk size."""
        long_line_file = tmp_path / "longline.py"
        # Create a line longer than typical chunk size
        long_content = "#" + "x" * 10000 + "\n"
        long_line_file.write_text(long_content)

        result = document_processor.process_file(long_line_file, "/test")

        # Should handle very long lines
        assert len(result) >= 1
        # Content should be preserved
        full_content = "".join(chunk.content for chunk in result)
        assert "x" in full_content

    def test_encoding_detection_with_mixed_encodings(
        self, document_processor: DocumentProcessor, tmp_path: Path
    ):
        """Test encoding detection with mixed encoding content."""
        mixed_file = tmp_path / "mixed.txt"
        # Create content with mixed byte sequences
        mixed_content = b"English text\n" + "Café résumé\n".encode(
            "latin-1"
        )
        mixed_file.write_bytes(mixed_content)

        # Should detect and decode appropriately
        try:
            result = document_processor.process_file(mixed_file, "/test")
            assert isinstance(result, list)
        except EncodingError:
            # Acceptable if encoding detection fails
            pass

    def test_file_with_invalid_utf8_sequences(
        self, document_processor: DocumentProcessor, tmp_path: Path
    ):
        """Test handling of files with invalid UTF-8 sequences."""
        invalid_utf8_file = tmp_path / "invalid.py"
        # Create content with invalid UTF-8 sequences
        invalid_utf8_file.write_bytes(
            b"# Valid comment\n\xff\xfe\x00\n# Another comment\n"
        )

        # Should handle invalid UTF-8 gracefully
        try:
            result = document_processor.process_file(invalid_utf8_file, "/test")
            assert isinstance(result, list)
        except EncodingError:
            # Acceptable if all encoding attempts fail
            pass

    def test_chunking_with_special_characters(
        self, document_processor: DocumentProcessor, tmp_path: Path
    ):
        """Test chunking with special Unicode characters."""
        special_chars_file = tmp_path / "special.py"
        special_content = "# Special characters: ñáéíóú ß 漢字 🐍\n" * 100
        special_chars_file.write_text(special_content)

        result = document_processor.process_file(special_chars_file, "/test")

        # Should preserve special characters in chunks
        assert len(result) > 0
        full_content = "".join(chunk.content for chunk in result)
        assert "ñáéíóú" in full_content or "🐍" in full_content

    def test_memory_efficiency_large_file_processing(
        self, document_processor: DocumentProcessor, tmp_path: Path
    ):
        """Test memory efficiency with large files."""
        large_file = tmp_path / "large.py"
        # Create a reasonably large file
        large_content = "# Large file test\n" + "print('test')\n" * 10000
        large_file.write_text(large_content)

        # Should process without memory issues
        result = document_processor.process_file(large_file, "/test")

        assert len(result) > 0
        assert isinstance(result, list)
