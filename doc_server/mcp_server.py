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
from .search.hybrid_search import HybridSearch, get_hybrid_search

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
