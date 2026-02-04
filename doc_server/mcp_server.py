"""
MCP Server implementation for doc-server.

Provides AI-powered documentation search via MCP protocol with
FastMCP framework, supporting stdio transport for MCP clients.
"""

import logging
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastmcp import FastMCP

from .config import settings
from .ingestion.document_processor import DocumentProcessor
from .ingestion.file_filter import FileFilter
from .ingestion.git_cloner import GitCloner
from .ingestion.zip_extractor import ZIPExtractor
from .search.hybrid_search import get_hybrid_search
from .search.vector_store import get_vector_store

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    """
    Document search result for MCP responses.

    Attributes:
        content: Document content text
        file_path: Path to the source file
        library_id: Library identifier
        relevance_score: Overall relevance score (0-1)
        line_numbers: Optional tuple of (start_line, end_line)
        metadata: Additional metadata dictionary
    """

    content: str
    file_path: str
    library_id: str
    relevance_score: float
    line_numbers: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "content": self.content,
            "file_path": self.file_path,
            "library_id": self.library_id,
            "relevance_score": self.relevance_score,
        }
        if self.line_numbers is not None:
            result["line_numbers"] = self.line_numbers
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Manage server startup and shutdown."""
    logger.info("Starting doc-server MCP...")
    try:
        # Initialize hybrid search service
        search = get_hybrid_search()
        logger.info(
            f"HybridSearch initialized: vector_weight={search.vector_weight}, "
            f"keyword_weight={search.keyword_weight}"
        )
        yield
    finally:
        logger.info("Shutting down doc-server MCP...")


# Create FastMCP server instance
mcp = FastMCP(
    name="doc-server",
    instructions="AI-powered documentation search server. Use search_docs to search documentation.",
    lifespan=lifespan,
)


def _convert_search_result(result) -> DocumentResult:
    """Convert SearchResult from hybrid_search to DocumentResult for MCP."""
    return DocumentResult(
        content=result.content,
        file_path=result.file_path,
        library_id=result.library_id,
        relevance_score=result.relevance_score,
        line_numbers=result.line_numbers,
        metadata=result.metadata,
    )


@mcp.tool
def search_docs(query: str, library_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Search through ingested documentation.

    Args:
        query: Search query string (required)
        library_id: Library identifier to search within (required)
        limit: Maximum number of results (default 10, max 100)

    Returns:
        List of search results with content, metadata, and relevance scores

    Example:
        >>> search_docs("pandas read_csv", "/pandas")
        >>> search_docs("fastapi routing", "/fastapi", limit=5)
    """
    # Validate inputs
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if not library_id or not library_id.strip():
        raise ValueError("Library ID cannot be empty")

    # Normalize library_id
    try:
        normalized_library_id = settings.normalize_library_id(library_id)
    except ValueError as e:
        raise ValueError(f"Invalid library ID: {e}") from e

    # Validate limit
    if limit < 1:
        limit = 10
    elif limit > 100:
        limit = 100

    logger.info(
        f"Searching docs: query='{query[:100]}...', library_id={normalized_library_id}, limit={limit}"
    )

    try:
        # Get hybrid search service
        search = get_hybrid_search()

        # Perform search
        results = search.search(
            query=query,
            library_id=normalized_library_id,
            n_results=limit,
        )

        # Convert to DocumentResult and then to dict
        document_results = [_convert_search_result(r) for r in results]
        output = [r.to_dict() for r in document_results]

        logger.info(f"Found {len(output)} results for query")
        return output

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise RuntimeError(f"Search failed: {e}") from e


@dataclass
class LibraryInfo:
    """Information about an ingested library."""

    library_id: str
    collection_name: str
    document_count: int
    embedding_model: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "library_id": self.library_id,
            "collection_name": self.collection_name,
            "document_count": self.document_count,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
        }


def _sanitize_input(input_str: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        input_str: Input string to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized string

    Raises:
        ValueError: If input is invalid after sanitization
    """
    if not input_str or not input_str.strip():
        raise ValueError("Input cannot be empty")

    # Remove null bytes and other control characters
    sanitized = input_str.replace("\x00", "")

    # Limit length
    if len(sanitized) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length} characters")

    # Check for suspicious patterns
    suspicious_patterns = ["../", "..\\", "${", "`", "$(", "|"]
    for pattern in suspicious_patterns:
        if pattern in sanitized:
            raise ValueError(f"Input contains suspicious pattern: {pattern}")

    return sanitized


@mcp.tool
def list_libraries() -> list[dict[str, Any]]:
    """
    List all available libraries that have been ingested.

    Returns:
        List of library information dictionaries with library_id, document_count, and metadata

    Example:
        >>> list_libraries()
        [{"library_id": "/pandas", "document_count": 150, ...}, ...]
    """
    logger.info("Listing all libraries")

    try:
        vector_store = get_vector_store()
        collections = vector_store.list_collections()

        libraries = []
        for collection in collections:
            try:
                info = LibraryInfo(
                    library_id=collection.get("library_id", collection.get("name", "")),
                    collection_name=collection.get("name", ""),
                    document_count=collection.get("count", 0),
                    embedding_model=collection.get("metadata", {}).get(
                        "embedding_model", "unknown"
                    ),
                    created_at=collection.get("metadata", {}).get("created_at", 0.0),
                )
                libraries.append(info.to_dict())
            except Exception as e:
                logger.warning(f"Error processing collection info: {e}")
                continue

        logger.info(f"Found {len(libraries)} libraries")
        return libraries

    except Exception as e:
        logger.error(f"Failed to list libraries: {e}", exc_info=True)
        raise RuntimeError(f"Failed to list libraries: {e}") from e


@mcp.tool
def remove_library(library_id: str) -> bool:
    """
    Remove a library from the index and delete all its documents.

    Args:
        library_id: Library identifier to remove (required)

    Returns:
        True if library was removed, False if it didn't exist

    Example:
        >>> remove_library("/pandas")
        True
        >>> remove_library("/nonexistent")
        False
    """
    # Sanitize and validate library_id
    try:
        sanitized_id = _sanitize_input(library_id)
    except ValueError as e:
        raise ValueError(f"Invalid library ID: {e}") from e

    # Normalize library_id
    try:
        normalized_library_id = settings.normalize_library_id(sanitized_id)
    except ValueError as e:
        raise ValueError(f"Invalid library ID format: {e}") from e

    logger.info(f"Removing library: {normalized_library_id}")

    try:
        vector_store = get_vector_store()
        deleted = vector_store.delete_collection(normalized_library_id)

        if deleted:
            logger.info(f"Library '{normalized_library_id}' removed successfully")
        else:
            logger.info(f"Library '{normalized_library_id}' did not exist")

        return deleted

    except Exception as e:
        logger.error(f"Failed to remove library: {e}", exc_info=True)
        raise RuntimeError(f"Failed to remove library: {e}") from e


@mcp.tool
def ingest_library(source: str, library_id: str) -> dict[str, Any]:
    """
    Ingest documentation from a git repository, ZIP archive, or local folder.

    Args:
        source: Source to ingest from (git URL, path to ZIP file, or local directory path)
        library_id: Library identifier for this ingestion (required)

    Returns:
        Dictionary with ingestion status, document count, and any errors

    Example:
        >>> ingest_library("https://github.com/pandas-dev/pandas", "/pandas")
        {"success": True, "documents_ingested": 150, "library_id": "/pandas"}

        >>> ingest_library("/path/to/docs.zip", "/my-docs")
        {"success": True, "documents_ingested": 50, "library_id": "/my-docs"}
    """
    import tempfile
    from pathlib import Path

    # Sanitize and validate inputs
    try:
        sanitized_source = _sanitize_input(source, max_length=2000)
        sanitized_library_id = _sanitize_input(library_id)
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}") from e

    # Normalize library_id
    try:
        normalized_library_id = settings.normalize_library_id(sanitized_library_id)
    except ValueError as e:
        raise ValueError(f"Invalid library ID format: {e}") from e

    logger.info(
        f"Ingesting library: source='{sanitized_source[:100]}...', library_id={normalized_library_id}"
    )

    # Track temporary resources for cleanup
    temp_dir: Path | None = None
    extraction_path: Path | None = None

    try:
        # Determine source type and prepare extraction path
        source_lower = sanitized_source.lower()

        if source_lower.startswith(("http://", "https://", "git://", "ssh://")):
            # Git repository
            source_type = "git"
            temp_dir = Path(tempfile.mkdtemp(prefix="doc-server-git-"))
            extraction_path = temp_dir / "repo"
        elif sanitized_source.endswith(".zip"):
            # ZIP archive
            source_type = "zip"
            extraction_path = Path(sanitized_source).parent / "extracted"
        else:
            # Local directory
            source_type = "local"
            extraction_path = Path(sanitized_source).resolve()

        # Initialize components
        git_cloner = GitCloner(settings)
        zip_extractor = ZIPExtractor(settings)
        file_filter = FileFilter(settings)
        document_processor = DocumentProcessor(settings)
        vector_store = get_vector_store()

        # Step 1: Get source content
        logger.info(f"Step 1: Fetching content from {source_type}")

        if source_type == "git":
            git_cloner.clone_repository(sanitized_source, destination=extraction_path)
            logger.info(f"Cloned repository to {extraction_path}")
        elif source_type == "zip":
            extraction_path = zip_extractor.extract_archive(
                sanitized_source, destination=extraction_path
            )
            logger.info(f"Extracted ZIP to {extraction_path}")
        elif source_type == "local":
            if not extraction_path.exists():
                raise ValueError(f"Local path does not exist: {extraction_path}")
            logger.info(f"Using local directory: {extraction_path}")

        # Step 2: Filter files
        logger.info("Step 2: Filtering files")
        all_files = list(extraction_path.rglob("*"))
        files = [f for f in all_files if f.is_file()]

        filtered_files = file_filter.filter_files(files, base_path=extraction_path)
        included_files = [f for f in filtered_files if f.included]

        logger.info(
            f"Filtered {len(files)} files to {len(included_files)} included files"
        )

        if not included_files:
            raise ValueError("No files found matching criteria")

        # Step 3: Process documents
        logger.info("Step 3: Processing documents")
        all_chunks = []

        for i, file_result in enumerate(included_files):
            try:
                chunks = document_processor.process_file(
                    file_path=file_result.file_path,
                    library_id=normalized_library_id,
                )
                all_chunks.extend(chunks)

                # Progress feedback
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(included_files)} files")

            except Exception as e:
                logger.warning(f"Error processing {file_result.file_path}: {e}")
                continue

        logger.info(f"Created {len(all_chunks)} document chunks")

        # Step 4: Add to vector store
        logger.info("Step 4: Adding documents to vector store")

        # Create or get collection
        vector_store.create_collection(normalized_library_id, get_or_create=True)

        # Prepare documents for batch insertion
        documents = [chunk.content for chunk in all_chunks]
        metadatas = [
            {
                "file_path": chunk.file_path,
                "library_id": normalized_library_id,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
            }
            for chunk in all_chunks
        ]
        ids = [
            f"{chunk.library_id}_{chunk.file_path}_{chunk.line_start}_{chunk.line_end}"
            for chunk in all_chunks
        ]

        # Add in batches
        batch_size = settings.embedding_batch_size
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            added_ids = vector_store.add_documents(
                library_id=normalized_library_id,
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas,
                batch_size=len(batch_docs),
            )
            total_added += len(added_ids)

        logger.info(
            f"Added {total_added} documents to library '{normalized_library_id}'"
        )

        # Clean up temporary resources
        if temp_dir and temp_dir.exists():
            import shutil

            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

        result = {
            "success": True,
            "documents_ingested": total_added,
            "library_id": normalized_library_id,
            "source_type": source_type,
        }

        logger.info(f"Ingestion complete: {result}")
        return result

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)

        # Clean up temporary resources on failure
        if temp_dir and temp_dir.exists():
            import shutil

            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

        raise RuntimeError(f"Ingestion failed: {e}") from e


def main():
    """Main entry point for the MCP server."""
    # Configure logging based on debug setting
    if settings.mcp_debug:
        logging.getLogger("doc_server").setLevel(logging.DEBUG)
        logging.getLogger("fastmcp").setLevel(logging.DEBUG)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run with STDIO transport (default)
    logger.info("Starting doc-server MCP with stdio transport")
    mcp.run()
