#!/usr/bin/env python
"""Direct ingestion script to bypass MCP caching."""

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
