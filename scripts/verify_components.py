#!/usr/bin/env python
"""Development script to verify doc-server components work correctly."""

import asyncio
import tempfile
from pathlib import Path

from doc_server.config import Settings
from doc_server.ingestion.document_processor import DocumentProcessor
from doc_server.ingestion.file_filter import FileFilter
from doc_server.search.embedding_service import EmbeddingService


async def verify_components():
    print("Verifying doc-server components...\n")

    # 1. Test configuration
    print("1. Configuration")
    settings = Settings()
    print(f"   Storage path: {settings.storage_path}")
    print(f"   Embedding model: {settings.embedding_model}")
    print("   ✓ Config OK\n")

    # 2. Test file filter
    print("2. File Filter")
    filter = FileFilter(settings)
    test_dir = Path("/home/artyom/devel/doc-server")
    result = filter.filter_directory(str(test_dir))
    print(f"   Found {len(result)} files to process")
    for f in result[:3]:
        print(f"   - {f.file_path}")
    print("   ✓ Filter OK\n")

    # 3. Test document processor
    print("3. Document Processor")
    processor = DocumentProcessor(settings)
    with tempfile.TemporaryDirectory() as tmp:
        test_file = Path(tmp) / "test.py"
        test_file.write_text(
            "def hello():\n    print('Hello, world!')\n    return 42\n"
        )
        chunks = processor.process_file(test_file, "/test-lib")
        print(f"   Created {len(chunks)} chunks from test file")
        print("   ✓ Processor OK\n")

    # 4. Test embedding service
    print("4. Embedding Service")
    with tempfile.TemporaryDirectory() as tmp:
        service = EmbeddingService(cache_dir=tmp, enable_cache=False)
        embeddings = service.get_embeddings(["test sentence"])
        print(f"   Embedding shape: {embeddings.shape}")
        print("   ✓ Embeddings OK\n")

    print("=" * 50)
    print("All components working! MCP server is ready.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(verify_components())
