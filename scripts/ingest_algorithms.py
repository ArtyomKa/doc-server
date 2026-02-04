#!/usr/bin/env python
"""
Direct ingestion script to bypass MCP caching for the algorithms library.

Purpose:
- Provides a command-line interface for ingesting documentation from GitHub repositories
- Bypasses the MCP (Model Context Protocol) server caching layer
- Enables direct testing of the ingestion pipeline

What it tests:
- Git repository cloning with shallow clone
- File filtering and .gitignore parsing
- Document processing and chunking
- Embedding generation via EmbeddingService
- ChromaDB vector storage and collection management

Usage:
    python scripts/ingest_algorithms.py

Input:
    - GitHub URL: https://github.com/keon/algorithms
    - Library ID: /algorithms

Output:
    - Success: Ingestion result dictionary with document count
    - Failure: Exception with full traceback

Note: This is a development utility, not part of the automated test suite.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from doc_server.mcp_server import ingest_library


async def main():
    print("Starting algorithms library ingestion...")
    try:
        result = await ingest_library.fn(
            "https://github.com/keon/algorithms", "/algorithms"
        )
        print(f"✅ Ingestion successful: {result}")
        return 0
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
