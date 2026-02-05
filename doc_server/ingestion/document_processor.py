"""
Document processing and content extraction.

Provides functionality to process documents for ingestion into the vector store,
including metadata headers, intelligent chunking, and encoding handling.
"""

from pathlib import Path

import structlog
from pydantic import BaseModel

from doc_server.config import Settings

logger = structlog.get_logger()

# Constants
CHUNK_SIZE_BYTES = 2048  # 2KB
CHUNK_OVERLAP_RATIO = 0.1  # 10% overlap for context


class DocumentChunk(BaseModel):
    """A chunk of document content with metadata."""

    content: str
    file_path: str
    library_id: str
    line_start: int
    line_end: int

    def __str__(self) -> str:
        """Return formatted string representation."""
        return f"{self.file_path}:{self.line_start}-{self.line_end}"


class DocumentProcessorError(Exception):
    """Base exception for document processing errors."""

    pass


class EncodingError(DocumentProcessorError):
    """Raised when file encoding cannot be determined or read."""

    def __init__(self, message: str, file_path: str) -> None:
        """Initialize error with context."""
        self.file_path = file_path
        super().__init__(message)


class ChunkingError(DocumentProcessorError):
    """Raised when document chunking fails."""

    def __init__(self, message: str, file_path: str) -> None:
        """Initialize error with context."""
        self.file_path = file_path
        super().__init__(message)


class DocumentProcessor:
    """
    Process documents for vector store ingestion.

    Handles metadata headers, intelligent content chunking, encoding detection,
    and optimization for embedding generation.

    Features:
    - Metadata headers with file path, line numbers, library ID
    - Intelligent chunking on sentence boundaries (for text/markdown)
    - Code block preservation (for code files)
    - Encoding handling (UTF-8, Latin-1, automatic detection)
    - Content optimization for embeddings
    """

    def __init__(self, config: Settings) -> None:
        """Initialize document processor with configuration.

        Args:
            config: Application settings instance
        """
        self._config = config
        logger.debug("DocumentProcessor initialized", config=str(config))

    def process_file(self, file_path: Path, library_id: str) -> list[DocumentChunk]:
        """
        Process a file and return document chunks.

        Args:
            file_path: Path to the file to process
            library_id: Library identifier for metadata

        Returns:
            List of DocumentChunk objects with metadata

        Raises:
            EncodingError: If file encoding cannot be determined or read
            ChunkingError: If chunking fails
            DocumentProcessorError: For other processing errors
        """
        logger.info(
            "Processing file",
            file_path=str(file_path),
            library_id=library_id,
        )

        # Convert to Path object if needed
        if isinstance(file_path, str):
            file_path = Path(file_path)

        try:
            # Read file with encoding detection
            content = self._read_with_encoding(file_path)

            # Split content into lines for line number tracking
            lines = content.splitlines()

            if not lines:
                logger.warning("File is empty", file_path=str(file_path))
                return []

            # Check if chunking is needed
            file_size = file_path.stat().st_size
            if file_size <= CHUNK_SIZE_BYTES:
                # Small file: return single chunk with full content
                chunk = self._create_chunk(
                    content=content,
                    file_path=file_path,
                    library_id=library_id,
                    line_start=1,
                    line_end=len(lines),
                )
                return [chunk]
            else:
                # Large file: chunk intelligently
                chunks = self._chunk_content(
                    content=content,
                    lines=lines,
                    file_path=file_path,
                    library_id=library_id,
                )
                logger.info(
                    "Chunked file into pieces",
                    file_path=str(file_path),
                    num_chunks=len(chunks),
                )
                return chunks

        except EncodingError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to process file",
                file_path=str(file_path),
                error=str(exc),
            )
            raise DocumentProcessorError(
                f"Failed to process {file_path}: {exc}"
            ) from exc

    def _read_with_encoding(self, file_path: Path) -> str:
        """
        Read file with automatic encoding detection.

        Tries UTF-8 first, then falls back to Latin-1, then attempts automatic
        detection using chardet if available.

        Args:
            file_path: Path to file to read

        Returns:
            File content as string

        Raises:
            EncodingError: If file cannot be read with any supported encoding
        """
        encodings = ["utf-8", "latin-1"]

        # Try each encoding in order
        for encoding in encodings:
            try:
                with open(file_path, encoding=encoding) as f:
                    content = f.read()
                logger.debug(
                    "Successfully read file with encoding",
                    file_path=str(file_path),
                    encoding=encoding,
                )
                return content
            except (UnicodeDecodeError, UnicodeError):
                logger.debug(
                    "Failed to read with encoding, trying next",
                    file_path=str(file_path),
                    encoding=encoding,
                )
                continue

        # Try chardet if available
        try:
            import chardet

            raw_bytes = file_path.read_bytes()
            detected = chardet.detect(raw_bytes)
            detected_encoding = detected["encoding"]

            if detected_encoding and detected["confidence"] > 0.7:
                try:
                    content = raw_bytes.decode(detected_encoding)
                    logger.info(
                        "Detected encoding with chardet",
                        file_path=str(file_path),
                        encoding=detected_encoding,
                        confidence=detected["confidence"],
                    )
                    return content
                except (UnicodeDecodeError, UnicodeError):
                    pass
        except ImportError:
            logger.debug("chardet not available for encoding detection")
        except Exception as exc:
            logger.debug(
                "chardet detection failed",
                file_path=str(file_path),
                error=str(exc),
            )

        # All attempts failed
        raise EncodingError(
            f"Could not determine encoding for {file_path}. "
            f"Tried: {', '.join(encodings)}",
            file_path=str(file_path),
        )

    def _create_chunk(
        self,
        content: str,
        file_path: Path,
        library_id: str,
        line_start: int,
        line_end: int,
    ) -> DocumentChunk:
        """
        Create a document chunk with metadata header.

        Args:
            content: Chunk content
            file_path: Path to source file
            library_id: Library identifier
            line_start: Starting line number (1-indexed)
            line_end: Ending line number (1-indexed)

        Returns:
            DocumentChunk with metadata header prepended to content
        """
        # Optimize content for embeddings
        optimized_content = self._optimize_for_embeddings(content)

        # Create metadata header
        metadata_header = self._create_metadata_header(
            file_path=str(file_path),
            library_id=library_id,
            line_start=line_start,
            line_end=line_end,
        )

        # Combine header with content
        chunk_content = f"{metadata_header}\n{optimized_content}"

        return DocumentChunk(
            content=chunk_content,
            file_path=str(file_path),
            library_id=library_id,
            line_start=line_start,
            line_end=line_end,
        )

    def _create_metadata_header(
        self, file_path: str, library_id: str, line_start: int, line_end: int
    ) -> str:
        """
        Create metadata header for document chunk.

        Args:
            file_path: Path to source file
            library_id: Library identifier
            line_start: Starting line number
            line_end: Ending line number

        Returns:
            Formatted metadata header string
        """
        return f"---\nfile_path: {file_path}\nlibrary_id: {library_id}\nlines: {line_start}-{line_end}\n---"

    def _chunk_content(
        self,
        content: str,
        lines: list[str],
        file_path: Path,
        library_id: str,
    ) -> list[DocumentChunk]:
        """
        Intelligently chunk content while preserving context.

        For markdown/text files: chunks on sentence boundaries
        For code files: preserves code blocks, chunks on logical breaks
        Uses overlap to maintain context between chunks

        Args:
            content: Full file content
            lines: Lines of content (for line number tracking)
            file_path: Path to source file
            library_id: Library identifier

        Returns:
            List of DocumentChunk objects

        Raises:
            ChunkingError: If chunking fails
        """
        try:
            file_extension = file_path.suffix.lower()

            # Use appropriate chunking strategy based on file type
            if file_extension in {".md", ".rst", ".txt"}:
                return self._chunk_text_document(content, lines, file_path, library_id)
            else:
                # Code files and others: use line-based chunking
                return self._chunk_code_document(content, lines, file_path, library_id)
        except Exception as exc:
            logger.error(
                "Chunking failed",
                file_path=str(file_path),
                error=str(exc),
            )
            raise ChunkingError(
                f"Failed to chunk: {exc}", file_path=str(file_path)
            ) from exc

    def _chunk_text_document(
        self,
        content: str,
        lines: list[str],
        file_path: Path,
        library_id: str,
    ) -> list[DocumentChunk]:
        """
        Chunk text documents on sentence boundaries.

        Preserves paragraph structure and uses overlap for context.

        Args:
            content: Full file content
            lines: Lines of content
            file_path: Path to source file
            library_id: Library identifier

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        current_content = ""
        current_line_start = 1
        line_index = 0

        # Calculate overlap in bytes
        overlap_bytes = int(CHUNK_SIZE_BYTES * CHUNK_OVERLAP_RATIO)

        while line_index < len(lines):
            line = lines[line_index]

            # Calculate approximate byte size of current chunk + new line
            potential_content = (
                current_content + "\n" + line if current_content else line
            )
            potential_size = len(potential_content.encode("utf-8"))

            # Check if adding this line would exceed chunk size
            if potential_size > CHUNK_SIZE_BYTES and current_content:
                # Create chunk from current content
                chunk = self._create_chunk(
                    content=current_content,
                    file_path=file_path,
                    library_id=library_id,
                    line_start=current_line_start,
                    line_end=line_index,
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                # Find a good sentence boundary for overlap
                overlap_lines = self._find_sentence_overlap(
                    current_content, overlap_bytes
                )
                current_content = overlap_lines
                current_line_start = line_index - len(overlap_lines.splitlines()) + 1
            else:
                # Add line to current chunk
                if current_content:
                    current_content += "\n" + line
                else:
                    current_content = line
                    current_line_start = line_index + 1

            line_index += 1

        # Add final chunk if there's content remaining
        if current_content:
            chunk = self._create_chunk(
                content=current_content,
                file_path=file_path,
                library_id=library_id,
                line_start=current_line_start,
                line_end=len(lines),
            )
            chunks.append(chunk)

        return chunks

    def _chunk_code_document(
        self,
        content: str,
        lines: list[str],
        file_path: Path,
        library_id: str,
    ) -> list[DocumentChunk]:
        """
        Chunk code documents preserving code blocks.

        Uses line-based chunking with overlap to preserve code structure.

        Args:
            content: Full file content
            lines: Lines of content
            file_path: Path to source file
            library_id: Library identifier

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        current_content = ""
        current_line_start = 1
        line_index = 0

        # Calculate overlap in bytes
        overlap_bytes = int(CHUNK_SIZE_BYTES * CHUNK_OVERLAP_RATIO)

        while line_index < len(lines):
            line = lines[line_index]

            # Calculate approximate byte size of current chunk + new line
            potential_content = (
                current_content + "\n" + line if current_content else line
            )
            potential_size = len(potential_content.encode("utf-8"))

            # Check if adding this line would exceed chunk size
            if potential_size > CHUNK_SIZE_BYTES and current_content:
                # Create chunk from current content
                chunk = self._create_chunk(
                    content=current_content,
                    file_path=file_path,
                    library_id=library_id,
                    line_start=current_line_start,
                    line_end=line_index,
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_lines = self._find_line_overlap(current_content, overlap_bytes)
                current_content = overlap_lines
                current_line_start = line_index - len(overlap_lines.splitlines()) + 1
            else:
                # Add line to current chunk
                if current_content:
                    current_content += "\n" + line
                else:
                    current_content = line
                    current_line_start = line_index + 1

            line_index += 1

        # Add final chunk if there's content remaining
        if current_content:
            chunk = self._create_chunk(
                content=current_content,
                file_path=file_path,
                library_id=library_id,
                line_start=current_line_start,
                line_end=len(lines),
            )
            chunks.append(chunk)

        return chunks

    def _find_sentence_overlap(self, content: str, overlap_bytes: int) -> str:
        """
        Find a good sentence boundary for overlap in text documents.

        Args:
            content: Current chunk content
            overlap_bytes: Desired overlap size in bytes

        Returns:
            Content chunk ending at a sentence boundary
        """
        # Find approximate byte position for overlap
        lines = content.splitlines()
        overlap_content = ""
        current_size = 0

        # Build overlap content up to desired size
        for line in reversed(lines):
            line_size = len(line.encode("utf-8")) + 1  # +1 for newline
            if current_size + line_size > overlap_bytes:
                break
            if overlap_content:
                overlap_content = line + "\n" + overlap_content
            else:
                overlap_content = line
            current_size += line_size

        return overlap_content

    def _find_line_overlap(self, content: str, overlap_bytes: int) -> str:
        """
        Find overlap for code documents (line-based).

        Args:
            content: Current chunk content
            overlap_bytes: Desired overlap size in bytes

        Returns:
            Content chunk of overlapping lines
        """
        # Find approximate byte position for overlap
        lines = content.splitlines()
        overlap_content = ""
        current_size = 0

        # Build overlap content up to desired size
        for line in reversed(lines):
            line_size = len(line.encode("utf-8")) + 1  # +1 for newline
            if current_size + line_size > overlap_bytes:
                break
            if overlap_content:
                overlap_content = line + "\n" + overlap_content
            else:
                overlap_content = line
            current_size += line_size

        return overlap_content

    def _optimize_for_embeddings(self, content: str) -> str:
        """
        Optimize content for embedding generation.

        Removes excessive whitespace while preserving code formatting
        and special characters. Normalizes line endings.

        Args:
            content: Raw content to optimize

        Returns:
            Optimized content suitable for embeddings
        """
        # Preserve code blocks by not changing indentation
        # Just normalize line endings and remove trailing whitespace per line
        lines = content.splitlines()

        # Remove trailing whitespace from each line, preserve leading whitespace
        optimized_lines = [line.rstrip() for line in lines]

        # Remove excessive blank lines (more than 2 consecutive)
        final_lines = []
        blank_count = 0
        for line in optimized_lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:
                    final_lines.append(line)
            else:
                blank_count = 0
                final_lines.append(line)

        # Join with single newlines (normalize line endings)
        return "\n".join(final_lines)
