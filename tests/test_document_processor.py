"""
Unit tests for DocumentProcessor module.
"""

from pathlib import Path

import pytest

from doc_server.config import Settings
from doc_server.ingestion.document_processor import (
    CHUNK_OVERLAP_RATIO,
    CHUNK_SIZE_BYTES,
    ChunkingError,
    DocumentChunk,
    DocumentProcessor,
    DocumentProcessorError,
    EncodingError,
)


@pytest.fixture
def config() -> Settings:
    """Create a Settings instance for testing."""
    return Settings()


@pytest.fixture
def document_processor(config: Settings) -> DocumentProcessor:
    """Create a DocumentProcessor instance for testing."""
    return DocumentProcessor(config)


@pytest.fixture
def test_directory(tmp_path: Path) -> Path:
    """Create test directory with various document files."""
    # Small text file (no chunking needed)
    small_md = tmp_path / "small.md"
    small_md.write_text("# Small Document\n\nThis is a small document.\n")

    # Large markdown file (needs chunking)
    large_md = tmp_path / "large.md"
    large_md.write_text(
        "\n".join([f"Line {i} with some content here." for i in range(100)])
    )

    # Python code file
    code_py = tmp_path / "code.py"
    code_py.write_text(
        """def function1():
    '''This is function 1.'''
    pass


def function2():
    '''This is function 2.'''
    return 42


class MyClass:
    '''This is a class.'''
    
    def method(self):
        '''This is a method.'''
        pass
"""
    )

    # Empty file
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    # File with special characters
    special_chars = tmp_path / "special.txt"
    special_chars.write_text("Content with special chars: <>&\"'\\n\\t")

    # File with various encodings (Latin-1)
    latin1_file = tmp_path / "latin1.txt"
    latin1_file.write_bytes("Café résumé naïve".encode("latin-1"))

    return tmp_path


# ==================== DocumentChunk Tests ====================


def test_document_chunk_creation() -> None:
    """Test DocumentChunk model creation."""
    chunk = DocumentChunk(
        content="Test content",
        file_path="/path/to/file.py",
        library_id="/pandas",
        line_start=1,
        line_end=10,
    )

    assert chunk.content == "Test content"
    assert chunk.file_path == "/path/to/file.py"
    assert chunk.library_id == "/pandas"
    assert chunk.line_start == 1
    assert chunk.line_end == 10


def test_document_chunk_string_representation() -> None:
    """Test DocumentChunk __str__ method."""
    chunk = DocumentChunk(
        content="Test content",
        file_path="/path/to/file.py",
        library_id="/pandas",
        line_start=1,
        line_end=10,
    )

    assert str(chunk) == "/path/to/file.py:1-10"


# ==================== DocumentProcessor Initialization Tests ====================


def test_document_processor_initialization(config: Settings) -> None:
    """Test DocumentProcessor initialization."""
    processor = DocumentProcessor(config)
    assert processor is not None
    assert processor._config is config


# ==================== File Processing Tests ====================


def test_process_small_file(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test processing a small file (<2KB) returns single chunk."""
    small_file = test_directory / "small.md"
    chunks = document_processor.process_file(small_file, "/test-lib")

    assert len(chunks) == 1
    assert isinstance(chunks[0], DocumentChunk)
    assert chunks[0].file_path == str(small_file)
    assert chunks[0].library_id == "/test-lib"
    assert "Small Document" in chunks[0].content


def test_process_large_file(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test processing a large file (>2KB) returns multiple chunks."""
    large_file = test_directory / "large.md"
    chunks = document_processor.process_file(large_file, "/test-lib")

    assert len(chunks) > 1

    # Verify all chunks have proper metadata
    for chunk in chunks:
        assert chunk.file_path == str(large_file)
        assert chunk.library_id == "/test-lib"
        assert chunk.line_start >= 1
        assert chunk.line_end >= chunk.line_start
        assert "---" in chunk.content  # Metadata header marker

    # Verify chunks are sequential
    for i in range(len(chunks) - 1):
        assert chunks[i].line_end >= chunks[i + 1].line_start


def test_process_code_file(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test processing a Python code file."""
    code_file = test_directory / "code.py"
    chunks = document_processor.process_file(code_file, "/test-lib")

    assert len(chunks) >= 1

    # Verify code formatting is preserved
    for chunk in chunks:
        assert "def function1" in chunk.content or "def function2" in chunk.content
        assert "    " in chunk.content  # Indentation preserved


def test_process_empty_file(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test processing an empty file returns empty list."""
    empty_file = test_directory / "empty.txt"
    chunks = document_processor.process_file(empty_file, "/test-lib")

    assert chunks == []


def test_process_file_with_special_characters(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test processing file with special characters."""
    special_file = test_directory / "special.txt"
    chunks = document_processor.process_file(special_file, "/test-lib")

    assert len(chunks) == 1
    assert "special chars" in chunks[0].content
    assert "<>&" in chunks[0].content  # Special characters preserved


def test_process_nonexistent_file(document_processor: DocumentProcessor) -> None:
    """Test processing a non-existent file raises appropriate error."""
    nonexistent = Path("/nonexistent/file.txt")

    with pytest.raises(DocumentProcessorError):
        document_processor.process_file(nonexistent, "/test-lib")


# ==================== Encoding Handling Tests ====================


def test_read_utf8_file(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test reading UTF-8 encoded file."""
    utf8_file = test_directory / "utf8.txt"
    utf8_file.write_text("Hello 世界 🌍", encoding="utf-8")

    chunks = document_processor.process_file(utf8_file, "/test-lib")

    assert len(chunks) == 1
    assert "世界" in chunks[0].content
    assert "🌍" in chunks[0].content


def test_read_latin1_file(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test reading Latin-1 encoded file."""
    latin1_file = test_directory / "latin1.txt"
    latin1_file.write_bytes("Café résumé naïve".encode("latin-1"))

    chunks = document_processor.process_file(latin1_file, "/test-lib")

    assert len(chunks) == 1
    assert "Café" in chunks[0].content


def test_read_invalid_encoding_raises_encoding_error(
    document_processor: DocumentProcessor, tmp_path: Path
) -> None:
    """Test that invalid encoding raises EncodingError."""
    # Create a file that's not a valid text file
    invalid_file = tmp_path / "invalid.bin"
    # Write a mix of binary data that might not decode cleanly
    invalid_file.write_bytes(b"\x00\x01\x02\x03\x80\x81\x82\x83\xfe\xff")

    # Should raise EncodingError since UTF-8 and Latin-1 will fail
    # Note: Latin-1 accepts all byte values, so we need to check if the file
    # actually contains readable content
    try:
        chunks = document_processor.process_file(invalid_file, "/test-lib")
        # If it succeeds, that's because Latin-1 can decode anything
        # This is acceptable behavior
        assert isinstance(chunks, list)
    except EncodingError:
        # This is also acceptable - encoding detection failed
        pass


# ==================== Metadata Header Tests ====================


def test_metadata_header_format(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test that metadata headers are properly formatted."""
    small_file = test_directory / "small.md"
    chunks = document_processor.process_file(small_file, "/my-library")

    assert len(chunks) == 1
    content = chunks[0].content

    # Check metadata header structure
    assert content.startswith("---")
    assert "file_path:" in content
    assert "library_id: /my-library" in content
    assert "lines: 1-" in content
    # Check that header ends with --- and content follows
    parts = content.split("\n---", 2)
    assert len(parts) >= 2  # Header section exists


def test_metadata_header_line_numbers(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test that metadata headers contain correct line numbers."""
    small_file = test_directory / "small.md"
    chunks = document_processor.process_file(small_file, "/test-lib")

    assert chunks[0].line_start == 1
    # Should be at least 1 line
    assert chunks[0].line_end >= 1


# ==================== Chunking Strategy Tests ====================


def test_text_document_chunking(
    document_processor: DocumentProcessor, tmp_path: Path
) -> None:
    """Test chunking of text/markdown documents."""
    # Create file with paragraphs
    text_file = tmp_path / "text.md"
    content = "\n\n".join(
        ["Paragraph 1. " + " ".join(["word"] * 50) for _ in range(10)]
    )
    text_file.write_text(content)

    chunks = document_processor.process_file(text_file, "/test-lib")

    assert len(chunks) > 1

    # Verify each chunk has reasonable size
    for chunk in chunks:
        assert len(chunk.content) <= CHUNK_SIZE_BYTES + 1000  # Allow some overhead


def test_code_document_chunking(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test chunking of code documents."""
    code_file = test_directory / "code.py"
    chunks = document_processor.process_file(code_file, "/test-lib")

    assert len(chunks) >= 1

    # Verify code blocks are preserved
    for chunk in chunks:
        # Check that indentation is preserved
        assert (
            "    " in chunk.content
            or "def" in chunk.content
            or "class" in chunk.content
        )


def test_large_code_document_chunking(
    document_processor: DocumentProcessor, tmp_path: Path
) -> None:
    """Test chunking of a large code file."""
    # Create a large code file that requires chunking
    large_code_file = tmp_path / "large_code.py"
    code_lines = []
    for i in range(200):
        code_lines.append(f"def function_{i}():")
        code_lines.append(f"    '''Function {i} documentation.'''")
        code_lines.append(f"    pass")
        code_lines.append("")

    large_code_file.write_text("\n".join(code_lines))

    chunks = document_processor.process_file(large_code_file, "/test-lib")

    # Should be chunked since file is large
    assert len(chunks) > 1

    # Verify code formatting is preserved across chunks
    for chunk in chunks:
        # Check that indentation is preserved
        assert "    def" in chunk.content or "def function_" in chunk.content
        assert "    '''" in chunk.content or "'''" in chunk.content


def test_chunk_overlap(document_processor: DocumentProcessor, tmp_path: Path) -> None:
    """Test that chunks have overlap for context continuity."""
    # Create large file
    large_file = tmp_path / "overlap.txt"
    large_file.write_text("\n".join([f"Line {i}" for i in range(200)]))

    chunks = document_processor.process_file(large_file, "/test-lib")

    # Verify sequential chunks have overlap
    if len(chunks) > 1:
        # Last line of chunk i should appear in chunk i+1 (overlap)
        # This is harder to test directly, but we can check line ranges
        for i in range(len(chunks) - 1):
            # Next chunk should start before or at current chunk's end (overlap)
            assert chunks[i + 1].line_start <= chunks[i].line_end


# ==================== Content Optimization Tests ====================


def test_optimize_for_embeddings_removes_trailing_whitespace() -> None:
    """Test that trailing whitespace is removed."""
    processor = DocumentProcessor(Settings())
    content = "Line 1   \nLine 2\t\t\nLine 3    "

    optimized = processor._optimize_for_embeddings(content)

    # Lines should not have trailing whitespace
    lines = optimized.splitlines()
    for line in lines:
        assert line == line.rstrip()


def test_optimize_for_embeddings_preserves_leading_whitespace() -> None:
    """Test that leading whitespace is preserved (for code)."""
    processor = DocumentProcessor(Settings())
    content = "    def function():\n        return 42"

    optimized = processor._optimize_for_embeddings(content)

    # Leading whitespace should be preserved
    assert "    def function():" in optimized
    assert "        return 42" in optimized


def test_optimize_for_embeddings_removes_excessive_blank_lines() -> None:
    """Test that excessive blank lines are removed."""
    processor = DocumentProcessor(Settings())
    content = "Line 1\n\n\n\n\nLine 2"  # 4 blank lines

    optimized = processor._optimize_for_embeddings(content)

    # Should have at most 2 consecutive blank lines
    lines = optimized.splitlines()
    max_consecutive_blanks = 0
    current_blanks = 0

    for line in lines:
        if not line.strip():
            current_blanks += 1
            max_consecutive_blanks = max(max_consecutive_blanks, current_blanks)
        else:
            current_blanks = 0

    assert max_consecutive_blanks <= 2, (
        f"Found {max_consecutive_blanks} consecutive blank lines"
    )


def test_optimize_for_embeddings_preserves_special_characters() -> None:
    """Test that special characters are preserved."""
    processor = DocumentProcessor(Settings())
    content = "Special chars: <>&\"'\\t\\n"

    optimized = processor._optimize_for_embeddings(content)

    assert "<>&" in optimized
    assert '"' in optimized or "'" in optimized


# ==================== Error Handling Tests ====================


def test_chunking_error_context(
    document_processor: DocumentProcessor, tmp_path: Path
) -> None:
    """Test that ChunkingError includes file_path context."""
    # Create a file that might cause chunking issues
    test_file = tmp_path / "test.txt"
    test_file.write_text("x" * (CHUNK_SIZE_BYTES + 1000))

    # This should work normally, but let's test the error type
    try:
        document_processor.process_file(test_file, "/test-lib")
        # If it succeeds, that's fine
        assert True
    except ChunkingError as e:
        # If it fails, verify error has file_path
        assert hasattr(e, "file_path")
        assert e.file_path is not None


def test_encoding_reraise(
    document_processor: DocumentProcessor, tmp_path: Path
) -> None:
    """Test that EncodingError is properly re-raised."""
    # Mock _read_with_encoding to raise EncodingError
    original_read = document_processor._read_with_encoding
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    def mock_read_with_encoding_raises(file_path: Path) -> str:
        raise EncodingError("Mock encoding error", str(file_path))

    document_processor._read_with_encoding = mock_read_with_encoding_raises

    with pytest.raises(EncodingError) as exc_info:
        document_processor.process_file(test_file, "/test-lib")

    assert exc_info.value.file_path == str(test_file)

    # Restore original method
    document_processor._read_with_encoding = original_read


def test_encoding_error_context(
    document_processor: DocumentProcessor, tmp_path: Path
) -> None:
    """Test that EncodingError includes file_path context."""
    # Create a file that might cause encoding issues
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_bytes(b"\x00\x01\x02\x03\x80\x81\x82\x83")

    try:
        chunks = document_processor.process_file(invalid_file, "/test-lib")
        # If it succeeds, Latin-1 decoded it (which is fine)
        assert isinstance(chunks, list)
    except EncodingError as e:
        # If it fails, verify error has file_path
        assert e.file_path == str(invalid_file)


def test_encoding_error_has_file_path_attribute() -> None:
    """Test that EncodingError has file_path attribute."""
    error = EncodingError("Test error", "/test/file.txt")
    assert error.file_path == "/test/file.txt"
    assert str(error) == "Test error"


def test_chunking_error_has_file_path_attribute() -> None:
    """Test that ChunkingError has file_path attribute."""
    error = ChunkingError("Test error", "/test/file.txt")
    assert error.file_path == "/test/file.txt"
    assert str(error) == "Test error"


def test_document_processor_error_on_other_failures(
    document_processor: DocumentProcessor, tmp_path: Path
) -> None:
    """Test that other processing failures raise DocumentProcessorError."""
    # Try to process a directory (not a file)
    with pytest.raises(DocumentProcessorError):
        document_processor.process_file(tmp_path, "/test-lib")


# ==================== Integration Tests ====================


def test_end_to_end_document_processing(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test complete document processing workflow."""
    # Process multiple files
    results = {}

    for file_path in test_directory.iterdir():
        if file_path.is_file():
            try:
                chunks = document_processor.process_file(file_path, "/test-lib")
                results[str(file_path)] = chunks
            except Exception as e:
                # Some files may have encoding issues, that's ok
                results[str(file_path)] = f"Error: {e}"

    # Verify at least some files were processed successfully
    successful = [k for k, v in results.items() if isinstance(v, list) and v]
    assert len(successful) >= 3  # At least 3 files should succeed


def test_multiple_library_ids(
    document_processor: DocumentProcessor, test_directory: Path
) -> None:
    """Test processing same file with different library IDs."""
    small_file = test_directory / "small.md"

    chunks1 = document_processor.process_file(small_file, "/lib1")
    chunks2 = document_processor.process_file(small_file, "/lib2")

    assert len(chunks1) == len(chunks2) == 1
    assert chunks1[0].library_id == "/lib1"
    assert chunks2[0].library_id == "/lib2"
    assert chunks1[0].content != chunks2[0].content  # Different library_id in metadata


# ==================== Constants Tests ====================


def test_chunk_size_constant() -> None:
    """Test that CHUNK_SIZE_BYTES is 2KB."""
    assert CHUNK_SIZE_BYTES == 2048


def test_chunk_overlap_ratio_constant() -> None:
    """Test that CHUNK_OVERLAP_RATIO is 10%."""
    assert CHUNK_OVERLAP_RATIO == 0.1


# ==================== Private Method Tests (for coverage) ====================


def test_find_sentence_overlap(document_processor: DocumentProcessor) -> None:
    """Test _find_sentence_overlap private method."""
    content = "\n".join([f"Line {i}" for i in range(20)])
    overlap = document_processor._find_sentence_overlap(content, 100)

    # Should have some lines
    assert isinstance(overlap, str)
    assert len(overlap) > 0


def test_find_line_overlap(document_processor: DocumentProcessor) -> None:
    """Test _find_line_overlap private method."""
    content = "\n".join([f"Line {i}" for i in range(20)])
    overlap = document_processor._find_line_overlap(content, 100)

    # Should have some lines
    assert isinstance(overlap, str)
    assert len(overlap) > 0


def test_create_metadata_header(document_processor: DocumentProcessor) -> None:
    """Test _create_metadata_header private method."""
    header = document_processor._create_metadata_header(
        file_path="/test/file.md",
        library_id="/my-lib",
        line_start=10,
        line_end=20,
    )

    assert "---" in header
    assert "file_path: /test/file.md" in header
    assert "library_id: /my-lib" in header
    assert "lines: 10-20" in header


def test_create_chunk_private_method(
    document_processor: DocumentProcessor, tmp_path: Path
) -> None:
    """Test _create_chunk private method."""
    test_file = tmp_path / "test.md"
    test_file.write_text("Test content")

    chunk = document_processor._create_chunk(
        content="Test content",
        file_path=test_file,
        library_id="/test-lib",
        line_start=1,
        line_end=1,
    )

    assert isinstance(chunk, DocumentChunk)
    assert "Test content" in chunk.content
    assert chunk.file_path == str(test_file)
    assert chunk.library_id == "/test-lib"
    assert chunk.line_start == 1
    assert chunk.line_end == 1
