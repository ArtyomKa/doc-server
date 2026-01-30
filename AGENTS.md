# AGENTS.md

Development guide for agentic coding agents working on the doc-server repository.

## Project Overview

Doc Server is an AI-powered documentation management system with intelligent search capabilities. It provides MCP (Model Context Protocol) server implementation and REST API for ingesting, processing, and searching technical documentation.

**Technology Stack**: Python 3.10+, FastAPI, MCP, ChromaDB/FAISS, OpenAI embeddings, Pydantic

---

## Build, Lint, and Test Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Install in development mode with dev dependencies
uv pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_module_name.py

# Run with coverage
pytest --cov=doc_server

# Run with specific options (configured in pyproject.toml)
pytest -v --tb=short
```

### Code Formatting and Linting
```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Run comprehensive linting with ruff
ruff check .

# Fix auto-fixable ruff issues
ruff check --fix .
```

### Type Checking
```bash
# Run MyPy type checking
mypy doc_server

# Check specific module
mypy doc_server/module_name.py
```

### Running the Server
```bash
# MCP server mode
uv run python -m doc_server.mcp_server

# HTTP API server (development)
uv run uvicorn doc_server.mcp_server:app --reload

# HTTP API server (production)
uv run uvicorn doc_server.mcp_server:app --host 0.0.0.0 --port 8000

# Run any Python script
uv run python your_script.py
```

---

## Code Style Guidelines

### Formatting Standards
- **Line Length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **String Quotes**: Double quotes for docstrings, single quotes for string literals
- **Trailing Commas**: Required in multi-line function calls and data structures

### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Local imports
from doc_server.config import Settings
from doc_server.utils import helper_function
```

Use `isort` to maintain import order automatically.

### Type Hints & Naming
- **Required**: All public functions need type hints (Python 3.10+ syntax)
- **Return Types**: Always specify return types, even for None returns
- **Variables/Functions**: `snake_case` • **Classes**: `PascalCase` • **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Prefix with single underscore `_private_method`

```python
from typing import Optional

def process_document(file_path: Path, options: Optional[dict] = None) -> dict[str, any]:
    """Process a document and return structured data."""
    return {"processed": True, "content": "..."}

class DocumentProcessor:
    MAX_FILE_SIZE = 10_000_000  # 10MB
    
    def __init__(self, config: Settings) -> None:
        self._config = config
```

### Error Handling
- **Specific Exceptions**: Use most specific exception types, never bare `except`
- **Custom Exceptions**: Create domain-specific exception classes
- **Logging**: Use `structlog` for structured logging

```python
import structlog
logger = structlog.get_logger()

class DocumentProcessingError(Exception):
    """Raised when document processing fails."""

def process_document(file_path: Path) -> dict:
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        return process_content(file_path.read_text())
    except FileNotFoundError as exc:
        logger.error("Document not found", path=str(file_path))
        raise DocumentProcessingError(f"Failed to process {file_path}") from exc
```

### Async/Await Patterns
- **Use AsyncIO**: For I/O-bound operations (HTTP requests, database queries)
- **Avoid Mixing**: Don't mix sync and async code unnecessarily

```python
async def fetch_document(url: str) -> Optional[str]:
    """Fetch document content from URL."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError as exc:
        logger.error("Failed to fetch document", url=url, error=str(exc))
        return None
```

### Testing Guidelines
- **Location**: `tests/` directory with `test_` prefix
- **Structure**: Arrange-Act-Assert pattern with pytest fixtures
- **Async Tests**: Use `pytest-asyncio` for async functions

```python
import pytest
from doc_server.search.hybrid_search import HybridSearch

@pytest.fixture
def search_engine():
    return HybridSearch(test_config)

async def test_hybrid_search_returns_results(search_engine):
    query = "python async programming"
    results = await search_engine.search(query, top_k=5)
    assert len(results) <= 5
    assert all(isinstance(r, dict) for r in results)
```

---

## Development Workflow

1. **Type check**: `mypy doc_server` before commits
2. **Format**: `black . && isort .` 
3. **Lint**: `ruff check --fix .`
4. **Test**: `pytest` for functionality
5. **Single file**: Use specific paths for faster iteration

---

## Repository Structure

- **Main Package**: `doc_server/`
- **Entry Point**: `doc_server/mcp_server.py`
- **Config**: `doc_server/config.py` with Pydantic settings
- **Modules**: `search/` and `ingestion/` subpackages

---

## Tools Configuration

- **Black**: 88-char line length, Python 3.10+ target
- **isort**: Black profile, `doc_server` as first-party
- **MyPy**: Strict type checking, no untyped definitions
- **Ruff**: E, W, F, I, B, C4, UP rules
- **pytest**: Auto asyncio mode, verbose output, short tracebacks