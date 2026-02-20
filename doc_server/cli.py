"""
Command Line Interface for doc-server.

Provides CLI commands for documentation management:
- ingest: Ingest documentation from git/zip/local sources
- search: Search through ingested documentation
- list: List all available libraries
- remove: Remove a library from the index
- serve: Start the MCP server
- health: Check health status
"""

import shutil
import sys
import tempfile
import time
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any

import click
from click import style

from .config import settings

try:
    __version__ = get_version("doc-server")
except Exception:
    __version__ = "0.1.0"

from .ingestion.document_processor import DocumentProcessor
from .ingestion.file_filter import FileFilter
from .ingestion.git_cloner import GitCloner
from .ingestion.zip_extractor import ZIPExtractor
from .logging_config import configure_structlog, get_logger
from .mcp_server import main as mcp_main
from .search.hybrid_search import get_hybrid_search
from .search.vector_store import get_vector_store

# Configure logging for CLI
configure_structlog()
logger = get_logger(__name__)


def _run_remote_ingest(
    source: str, library_id: str, batch_size: int | None, verbose: bool
) -> None:
    """Run ingestion via remote backend API."""
    import asyncio

    from .api_client import APIClient

    click.echo(
        style("📚 Ingesting documentation (remote mode)...", fg="cyan", bold=True)
    )
    click.echo(style(f"   Source: {source}", fg="white"))
    click.echo(style(f"   Library: {library_id}", fg="white"))
    click.echo(style(f"   Backend: {settings.backend_url}", fg="white"))
    click.echo("")

    async def _do_ingest():
        async with APIClient(
            base_url=settings.backend_url,
            api_key=settings.backend_api_key,
            timeout=settings.backend_timeout,
            verify_ssl=settings.backend_verify_ssl,
        ) as client:
            result = await client.ingest(
                source=source,
                library_id=library_id,
                version=None,
                batch_size=batch_size or 32,
            )
            click.echo("")
            click.echo(
                style(
                    f"✓ Successfully ingested {result.documents_ingested} documents",
                    fg="green",
                    bold=True,
                )
            )

    try:
        asyncio.run(_do_ingest())
    except Exception as e:
        click.echo("")
        click.echo(style(f"❌ Remote ingestion failed: {e}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo("")
            click.echo(traceback.format_exc())
        sys.exit(1)


def _run_remote_search(query: str, library_id: str, limit: int, verbose: bool) -> None:
    """Run search via remote backend API."""
    import asyncio

    from .api_client import APIClient

    click.echo(
        style("🔍 Searching documentation (remote mode)...", fg="cyan", bold=True)
    )
    click.echo(style(f"   Query: {query}", fg="white"))
    click.echo(style(f"   Library: {library_id}", fg="white"))
    click.echo(style(f"   Backend: {settings.backend_url}", fg="white"))
    click.echo("")

    async def _do_search():
        async with APIClient(
            base_url=settings.backend_url,
            api_key=settings.backend_api_key,
            timeout=settings.backend_timeout,
            verify_ssl=settings.backend_verify_ssl,
        ) as client:
            results = await client.search(query, library_id, limit)

            if not results:
                click.echo(style("No results found.", fg="yellow"))
                return

            click.echo(style(f"Found {len(results)} results:\n", fg="green"))
            for i, result in enumerate(results, 1):
                click.echo(f"{i}. {result.file_path} (score: {result.score:.2f})")
                # Show first 200 chars of content
                content_preview = result.content[:200].replace("\n", " ")
                click.echo(f"   {content_preview}...")
                click.echo("")

    try:
        asyncio.run(_do_search())
    except Exception as e:
        click.echo("")
        click.echo(style(f"❌ Remote search failed: {e}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo("")
            click.echo(traceback.format_exc())
        sys.exit(1)


# Custom click group for better help formatting
class DocServerGroup(click.Group):
    """Custom click group with formatted help."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Format help with custom styling."""
        formatter.write_text(
            style("AI-powered documentation management system", fg="cyan", bold=True)
        )
        formatter.write_text("")
        super().format_help(ctx, formatter)


@click.group(cls=DocServerGroup, invoke_without_command=False)
@click.version_option(version=__version__, prog_name="doc-server")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    Doc Server CLI - Manage and search documentation libraries.

    Commands:
        ingest    Ingest documentation from git/zip/local sources
        search    Search through ingested documentation
        list      List all available libraries
        remove    Remove a library from the index
        serve     Start the MCP server
        health    Check health status
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


def _sanitize_input(input_str: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent injection attacks."""
    if not input_str or not input_str.strip():
        raise ValueError("Input cannot be empty")

    sanitized = input_str.replace("\x00", "")

    if len(sanitized) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length} characters")

    suspicious_patterns = ["../", "..\\", "${", "`", "$(", "|"]
    for pattern in suspicious_patterns:
        if pattern in sanitized:
            raise ValueError(f"Input contains suspicious pattern: {pattern}")

    return sanitized


@cli.command()
@click.option(
    "--source",
    "-s",
    required=True,
    help="Source to ingest from (git URL, ZIP file path, or local directory)",
)
@click.option(
    "--library-id",
    "-l",
    required=True,
    help="Library identifier (e.g., /pandas, /fastapi)",
)
@click.option(
    "--batch-size",
    "-b",
    default=None,
    type=int,
    help="Batch size for document processing (default: from config)",
)
@click.pass_context
def ingest(
    ctx: click.Context, source: str, library_id: str, batch_size: int | None
) -> None:
    """
    Ingest documentation from a git repository, ZIP archive, or local folder.

    Examples:
        doc-server ingest -s https://github.com/pandas-dev/pandas -l /pandas
        doc-server ingest -s ./docs.zip -l /my-docs
        doc-server ingest -s ./local-docs -l /project-docs
    """
    verbose = ctx.obj.get("verbose", False)

    # Check if remote mode is enabled
    if settings.mode == "remote":
        _run_remote_ingest(source, library_id, batch_size, verbose)
        return

    click.echo(style("📚 Ingesting documentation...", fg="cyan", bold=True))
    click.echo(style(f"   Source: {source}", fg="white"))
    click.echo(style(f"   Library: {library_id}", fg="white"))
    click.echo("")

    temp_dir: Path | None = None

    try:
        # Sanitize inputs
        sanitized_source = _sanitize_input(source, max_length=2000)
        sanitized_library_id = _sanitize_input(library_id)
        normalized_library_id = settings.normalize_library_id(sanitized_library_id)

        # Override batch size if provided
        if batch_size:
            settings.embedding_batch_size = batch_size

        # Determine source type
        source_lower = sanitized_source.lower()
        if source_lower.startswith(("http://", "https://", "git://", "ssh://")):
            source_type = "git"
            temp_dir = Path(tempfile.mkdtemp(prefix="doc-server-git-"))
            extraction_path = temp_dir / "repo"
        elif sanitized_source.endswith(".zip"):
            source_type = "zip"
            extraction_path = Path(sanitized_source).parent / "extracted"
        else:
            source_type = "local"
            extraction_path = Path(sanitized_source).resolve()

        # Initialize components
        git_cloner = GitCloner(settings)
        zip_extractor = ZIPExtractor(settings)
        file_filter = FileFilter(settings)
        document_processor = DocumentProcessor(settings)
        vector_store = get_vector_store()

        # Step 1: Get source content
        click.echo(style("Step 1: Fetching content...", fg="yellow"))
        if source_type == "git":
            git_cloner.clone_repository(sanitized_source, destination=extraction_path)
        elif source_type == "zip":
            extraction_path = zip_extractor.extract_archive(
                sanitized_source, destination=extraction_path
            )
        elif source_type == "local":
            if not extraction_path.exists():
                raise ValueError(f"Local path does not exist: {extraction_path}")

        # Step 2: Filter files
        click.echo(style("Step 2: Filtering files...", fg="yellow"))
        all_files = list(extraction_path.rglob("*"))
        files = [f for f in all_files if f.is_file()]
        filtered_files = file_filter.filter_files(
            [str(f) for f in files], base_path=str(extraction_path)
        )
        included_files = [f for f in filtered_files if f.included]

        if not included_files:
            raise ValueError("No files found matching criteria")

        click.echo(
            style(f"   Found {len(included_files)} files to process", fg="green")
        )

        # Step 3: Process documents with progress bar
        click.echo(style("Step 3: Processing documents...", fg="yellow"))
        all_chunks = []

        with click.progressbar(
            included_files,
            label=style("Processing", fg="cyan"),
            fill_char=style("█", fg="green"),
            empty_char=style("░", fg="white"),
        ) as bar:
            for file_result in bar:
                try:
                    chunks = document_processor.process_file(
                        file_path=Path(file_result.file_path),
                        library_id=normalized_library_id,
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    if verbose:
                        click.echo(
                            style(
                                f"   Warning: Error processing {file_result.file_path}: {e}",
                                fg="yellow",
                            )
                        )
                    continue

        click.echo(style(f"   Created {len(all_chunks)} document chunks", fg="green"))

        # Step 4: Add to vector store with progress bar
        click.echo(style("Step 4: Adding to vector store...", fg="yellow"))

        vector_store.create_collection(normalized_library_id, get_or_create=True)

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

        batch_size = settings.embedding_batch_size
        total_added = 0

        with click.progressbar(
            range(0, len(documents), batch_size),
            label=style("Indexing", fg="cyan"),
            fill_char=style("█", fg="green"),
            empty_char=style("░", fg="white"),
        ) as bar:
            for i in bar:
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

        # Clean up temporary resources
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)

        click.echo("")
        click.echo(
            style(
                f"✅ Successfully ingested {total_added} documents",
                fg="green",
                bold=True,
            )
        )
        click.echo(style(f"   Library ID: {normalized_library_id}", fg="white"))
        click.echo(style(f"   Source type: {source_type}", fg="white"))

    except ValueError as e:
        click.echo("")
        click.echo(style(f"❌ Validation error: {e}", fg="red", bold=True))
        sys.exit(1)
    except Exception as e:
        click.echo("")
        click.echo(style(f"❌ Ingestion failed: {e}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo("")
            click.echo(traceback.format_exc())
        sys.exit(1)
    finally:
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


@cli.command()
@click.option(
    "--query",
    "-q",
    required=True,
    help="Search query string",
)
@click.option(
    "--library-id",
    "-l",
    required=True,
    help="Library identifier to search within",
)
@click.option(
    "--limit",
    "-n",
    default=10,
    type=int,
    help="Maximum number of results (default: 10, max: 100)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json", "simple"]),
    default="table",
    help="Output format (default: table)",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    library_id: str,
    limit: int,
    output_format: str,
) -> None:
    """
    Search through ingested documentation.

    Examples:
        doc-server search -q "pandas read_csv" -l /pandas
        doc-server search -q "fastapi routing" -l /fastapi -n 5
        doc-server search -q "binary search" -l /algorithms -f json
    """
    verbose = ctx.obj.get("verbose", False)

    # Check if remote mode is enabled
    if settings.mode == "remote":
        _run_remote_search(query, library_id, limit, verbose)
        return

    try:
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not library_id or not library_id.strip():
            raise ValueError("Library ID cannot be empty")

        normalized_library_id = settings.normalize_library_id(library_id)

        if limit < 1:
            limit = 10
        elif limit > 100:
            limit = 100

        if verbose:
            click.echo(style("🔍 Searching documentation...", fg="cyan"))
            click.echo(style(f"   Query: {query}", fg="white"))
            click.echo(style(f"   Library: {library_id}", fg="white"))
            click.echo("")

        # Perform search
        search_service = get_hybrid_search()
        results = search_service.search(
            query=query,
            library_id=normalized_library_id,
            n_results=limit,
        )

        if not results:
            click.echo(style("No results found.", fg="yellow"))
            return

        if output_format == "json":
            import json

            output = []
            for r in results:
                output.append(
                    {
                        "content": r.content,
                        "file_path": r.file_path,
                        "library_id": r.library_id,
                        "relevance_score": r.relevance_score,
                        "line_numbers": r.line_numbers,
                        "metadata": r.metadata,
                    }
                )
            click.echo(json.dumps(output, indent=2))
        elif output_format == "simple":
            for i, result in enumerate(results, 1):
                click.echo(
                    f"{i}. {result.file_path} (score: {result.relevance_score:.3f})"
                )
                content_preview = result.content[:200].replace("\n", " ")
                click.echo(f"   {content_preview}...")
                click.echo("")
        else:  # table format
            click.echo("")
            click.echo(style(f"Found {len(results)} results", fg="green", bold=True))
            click.echo("")

            for i, result in enumerate(results, 1):
                header = f"{i}. {result.file_path}"
                click.echo(style(header, fg="cyan", bold=True))
                click.echo(
                    style(f"   Relevance: {result.relevance_score:.3f}", fg="yellow")
                )

                if result.line_numbers:
                    line_info = (
                        f"   Lines: {result.line_numbers[0]}-{result.line_numbers[1]}"
                    )
                    click.echo(style(line_info, fg="white"))

                content = result.content
                if len(content) > 300:
                    content = content[:300] + "..."

                lines = content.split("\n")
                for line in lines[:10]:
                    if line.strip():
                        click.echo(f"   {line}")

                if len(lines) > 10:
                    click.echo(
                        style(f"   ... ({len(lines) - 10} more lines)", fg="white")
                    )

                click.echo("")

    except ValueError as e:
        click.echo(style(f"❌ Validation error: {e}", fg="red", bold=True))
        sys.exit(1)
    except Exception as e:
        click.echo(style(f"❌ Search failed: {e}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo("")
            click.echo(traceback.format_exc())
        sys.exit(1)


@cli.command(name="list")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json", "simple"]),
    default="table",
    help="Output format (default: table)",
)
@click.pass_context
def list_cmd(ctx: click.Context, output_format: str) -> None:
    """
    List all available libraries that have been ingested.

    Examples:
        doc-server list
        doc-server list -f json
    """
    verbose = ctx.obj.get("verbose", False)

    if verbose:
        click.echo(style("📋 Listing libraries...", fg="cyan"))

    try:
        vector_store = get_vector_store()
        collections = vector_store.list_collections()

        libraries = []
        for collection in collections:
            try:
                libraries.append(
                    {
                        "library_id": collection.get(
                            "library_id", collection.get("name", "")
                        ),
                        "collection_name": collection.get("name", ""),
                        "document_count": collection.get("count", 0),
                        "embedding_model": collection.get("metadata", {}).get(
                            "embedding_model", "unknown"
                        ),
                        "created_at": collection.get("metadata", {}).get(
                            "created_at", 0.0
                        ),
                    }
                )
            except Exception:
                continue

        if not libraries:
            click.echo(
                style(
                    "No libraries found. Use 'ingest' to add documentation.",
                    fg="yellow",
                )
            )
            return

        if output_format == "json":
            import json

            click.echo(json.dumps(libraries, indent=2))
        elif output_format == "simple":
            for lib in libraries:
                click.echo(f"{lib['library_id']}: {lib['document_count']} documents")
        else:  # table format
            click.echo("")
            click.echo(
                style(f"📚 {len(libraries)} libraries available", fg="green", bold=True)
            )
            click.echo("")

            max_id_len = max(len(lib["library_id"]) for lib in libraries)
            max_count_len = max(len(str(lib["document_count"])) for lib in libraries)

            header = f"{'Library ID':<{max_id_len}}  {'Documents':>{max_count_len}}  {'Model':<20}"
            click.echo(style(header, fg="cyan", bold=True))
            click.echo(style("-" * (max_id_len + max_count_len + 25), fg="white"))

            for lib in libraries:
                row = (
                    f"{lib['library_id']:<{max_id_len}}  "
                    f"{lib['document_count']:>{max_count_len}}  "
                    f"{lib['embedding_model']:<20}"
                )
                click.echo(row)

            click.echo("")

    except Exception as e:
        click.echo(style(f"❌ Failed to list libraries: {e}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo("")
            click.echo(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option(
    "--library-id",
    "-l",
    required=True,
    help="Library identifier to remove",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Skip confirmation prompt",
)
@click.pass_context
def remove(ctx: click.Context, library_id: str, force: bool) -> None:
    """
    Remove a library from the index and delete all its documents.

    Examples:
        doc-server remove -l /pandas
        doc-server remove -l /fastapi --force
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        # Sanitize and validate
        sanitized_id = _sanitize_input(library_id)
        normalized_library_id = settings.normalize_library_id(sanitized_id)

        if verbose:
            click.echo(
                style(f"🗑️  Removing library: {normalized_library_id}", fg="cyan")
            )

        # Confirmation prompt
        if not force:
            click.echo("")
            click.echo(
                style(
                    f"⚠️  This will permanently delete library '{normalized_library_id}'",
                    fg="yellow",
                    bold=True,
                )
            )
            click.echo(
                style("   All documents will be removed from the index.", fg="yellow")
            )
            click.echo("")

            if not click.confirm("Do you want to continue?"):
                click.echo(style("Operation cancelled.", fg="white"))
                return

        vector_store = get_vector_store()
        deleted = vector_store.delete_collection(normalized_library_id)

        if deleted:
            click.echo("")
            click.echo(
                style(
                    f"✅ Library '{normalized_library_id}' removed successfully",
                    fg="green",
                    bold=True,
                )
            )
        else:
            click.echo("")
            click.echo(
                style(
                    f"ℹ️  Library '{normalized_library_id}' did not exist", fg="yellow"
                )
            )

    except ValueError as e:
        click.echo(style(f"❌ Validation error: {e}", fg="red", bold=True))
        sys.exit(1)
    except Exception as e:
        click.echo(style(f"❌ Failed to remove library: {e}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo("")
            click.echo(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option(
    "--transport",
    "-t",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type (default: stdio)",
)
@click.option(
    "--host",
    "-h",
    default="127.0.0.1",
    help="Host to bind to for SSE transport (default: 127.0.0.1)",
)
@click.option(
    "--port",
    "-p",
    default=8080,
    type=int,
    help="Port to bind to for SSE transport (default: 8080)",
)
@click.pass_context
def serve(ctx: click.Context, transport: str, host: str, port: int) -> None:
    """
    Start the MCP server.

    This starts the MCP server that can be used by MCP clients like Claude Desktop.

    Examples:
        doc-server serve
        doc-server serve -t stdio
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo(style("🚀 Starting doc-server MCP server...", fg="cyan", bold=True))
    click.echo("")

    # Validate startup
    try:
        validation = _validate_startup()
        if validation["status"] != "healthy":
            click.echo(style("⚠️  Server validation warnings:", fg="yellow", bold=True))
            for component, status in validation.get("components", {}).items():
                if isinstance(status, dict) and "error" in status:
                    click.echo(
                        style(f"   - {component}: {status['error']}", fg="yellow")
                    )
            click.echo("")
    except Exception as e:
        click.echo(style(f"⚠️  Startup validation failed: {e}", fg="yellow", bold=True))
        click.echo("")

    click.echo(style(f"Transport: {transport}", fg="white"))
    if transport == "sse":
        click.echo(style(f"Host: {host}", fg="white"))
        click.echo(style(f"Port: {port}", fg="white"))
    click.echo("")
    click.echo(style("Press Ctrl+C to stop the server", fg="yellow"))
    click.echo("")

    try:
        import os

        os.environ["MCP_TRANSPORT"] = transport
        if transport == "sse":
            os.environ["MCP_HOST"] = host
            os.environ["MCP_PORT"] = str(port)

        mcp_main()
    except KeyboardInterrupt:
        click.echo("")
        click.echo(style("👋 Server stopped", fg="green"))
    except Exception as e:
        click.echo("")
        click.echo(style(f"❌ Server error: {e}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo("")
            click.echo(traceback.format_exc())
        sys.exit(1)


@click.command("backend")
@click.option(
    "--host",
    "-h",
    default="0.0.0.0",
    help="Host to bind to (default: 0.0.0.0)",
)
@click.option(
    "--port",
    "-p",
    default=8000,
    type=int,
    help="Port to bind to (default: 8000)",
)
@click.option(
    "--workers",
    "-w",
    default=1,
    type=int,
    help="Number of worker processes (default: 1)",
)
@click.pass_context
def backend(ctx: click.Context, host: str, port: int, workers: int) -> None:
    """
    Start the Doc Server backend server.

    This starts the REST API server that can be used by remote clients.

    Examples:
        doc-server backend
        doc-server backend -h 0.0.0.0 -p 8000
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo(style("🚀 Starting doc-server backend server...", fg="cyan", bold=True))
    click.echo("")

    if not settings.backend_api_key:
        click.echo(
            style(
                "⚠️  Warning: No API key configured. Set DOC_SERVER_API_KEY environment variable.",
                fg="yellow",
                bold=True,
            )
        )
        click.echo("")

    click.echo(style(f"Host: {host}", fg="white"))
    click.echo(style(f"Port: {port}", fg="white"))
    click.echo(style(f"Workers: {workers}", fg="white"))
    click.echo("")
    click.echo(style("Press Ctrl+C to stop the server", fg="yellow"))
    click.echo("")

    try:
        import uvicorn

        uvicorn.run(
            "doc_server.api_server:app",
            host=host,
            port=port,
            workers=workers,
            reload=False,
        )
    except KeyboardInterrupt:
        click.echo("")
        click.echo(style("👋 Server stopped", fg="green"))
    except Exception as e:
        click.echo("")
        click.echo(style(f"❌ Server error: {e}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo("")
            click.echo(traceback.format_exc())
        sys.exit(1)


def _validate_startup() -> dict[str, Any]:
    """Validate startup dependencies."""
    checks: dict[str, Any] = {
        "status": "healthy",
        "components": {},
        "timestamp": time.time(),
    }

    # Check storage paths
    try:
        storage_checks: dict[str, Any] = {
            "storage_path_exists": settings.storage_path.exists(),
            "storage_path_writable": settings.storage_path.is_dir()
            or settings.storage_path.parent.is_dir(),
        }

        for path_prop in ["chroma_db_path", "models_path", "libraries_path"]:
            path = getattr(settings, path_prop)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.mkdir(parents=True, exist_ok=True)

        storage_checks["directories_created"] = True
        checks["components"]["storage"] = storage_checks

    except Exception as exc:
        checks["components"]["storage"] = {"error": str(exc)}
        checks["status"] = "degraded"

    # Check vector store
    try:
        vector_store = get_vector_store()
        collections = vector_store.list_collections()
        checks["components"]["vector_store"] = {
            "initialized": True,
            "collection_count": len(collections),
        }
    except Exception as exc:
        checks["components"]["vector_store"] = {"error": str(exc)}
        checks["status"] = "degraded"

    # Check hybrid search
    try:
        search = get_hybrid_search()
        checks["components"]["hybrid_search"] = {
            "initialized": True,
            "vector_weight": search.vector_weight,
            "keyword_weight": search.keyword_weight,
        }
    except Exception as exc:
        checks["components"]["hybrid_search"] = {"error": str(exc)}
        checks["status"] = "degraded"

    return checks


def _get_health_status() -> dict[str, Any]:
    """Get health status of the server."""
    # Check if remote mode is enabled
    if settings.mode == "remote":
        return _get_remote_health_status()

    health: dict[str, Any] = {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time(),
        "components": {},
    }

    try:
        vector_store = get_vector_store()
        collections = vector_store.list_collections()
        health["components"]["vector_store"] = {
            "status": "healthy",
            "collection_count": len(collections),
        }

        collection_details = []
        for collection in collections:
            try:
                collection_details.append(
                    {
                        "name": collection.get("name", "unknown"),
                        "count": collection.get("count", 0),
                    }
                )
            except Exception:
                collection_details.append(
                    {
                        "name": collection.get("name", "unknown"),
                        "error": "failed to get count",
                    }
                )

        health["components"]["collections"] = {
            "status": "healthy" if collections else "empty",
            "details": collection_details,
        }

    except Exception as exc:
        health["components"]["vector_store"] = {
            "status": "unhealthy",
            "error": str(exc),
        }
        health["status"] = "unhealthy"

    return health


def _get_remote_health_status() -> dict[str, Any]:
    """Get health status from remote backend API."""
    import asyncio

    from .api_client import APIClient

    async def _do_health_check() -> dict[str, Any]:
        async with APIClient(
            base_url=settings.backend_url,
            api_key=settings.backend_api_key,
            timeout=settings.backend_timeout,
            verify_ssl=settings.backend_verify_ssl,
        ) as client:
            result = await client.health_check()
            return {
                "status": result.status,
                "version": "remote",
                "timestamp": result.timestamp or time.time(),
                "components": result.components,
            }

    try:
        return asyncio.run(_do_health_check())
    except Exception as exc:
        return {
            "status": "unhealthy",
            "version": "unknown",
            "timestamp": time.time(),
            "components": {"remote": {"status": "unhealthy", "error": str(exc)}},
        }


@cli.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """
    Check the health status of the doc-server.

    Examples:
        doc-server health
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        health_status = _get_health_status()

        status_color = "green" if health_status["status"] == "healthy" else "red"
        click.echo("")
        click.echo(
            style(f"Status: {health_status['status']}", fg=status_color, bold=True)
        )
        click.echo(style(f"Version: {health_status['version']}", fg="white"))
        click.echo("")

        if "components" in health_status:
            click.echo(style("Components:", fg="cyan", bold=True))
            for component, info in health_status["components"].items():
                if isinstance(info, dict):
                    if "status" in info:
                        comp_color = "green" if info["status"] == "healthy" else "red"
                        click.echo(
                            style(f"  {component}: {info['status']}", fg=comp_color)
                        )
                    elif "error" in info:
                        click.echo(
                            style(f"  {component}: error - {info['error']}", fg="red")
                        )
                    else:
                        click.echo(f"  {component}: {info}")
                else:
                    click.echo(f"  {component}: {info}")

        if verbose:
            click.echo("")
            click.echo(style("Full status:", fg="cyan"))
            import json

            click.echo(json.dumps(health_status, indent=2))

        click.echo("")

        if health_status["status"] != "healthy":
            sys.exit(1)

    except Exception as e:
        click.echo(style(f"❌ Health check failed: {e}", fg="red", bold=True))
        if verbose:
            import traceback

            click.echo("")
            click.echo(traceback.format_exc())
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    cli()


# Register backend command explicitly (in case decorator didn't work)
if "backend" not in cli.commands:
    cli.add_command(backend)


if __name__ == "__main__":
    main()
