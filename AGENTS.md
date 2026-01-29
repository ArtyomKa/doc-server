# AGENTS.md

Development guide for agentic coding agents working on the doc-server repository.

## Project Overview

Doc Server is an AI-powered documentation management system with intelligent search capabilities. It provides MCP (Model Context Protocol) server implementation and REST API for ingesting, processing, and searching technical documentation.

**Technology Stack**: Python 3.10+, FastAPI, MCP, ChromaDB/FAISS, OpenAI embeddings, Pydantic

---

## Build, Lint, and Test Commands

### Environment Setup
```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install from requirements
pip install -r requirements.txt
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
python -m doc_server.mcp_server

# HTTP API server (development)
uvicorn doc_server.mcp_server:app --reload

# HTTP API server (production)
uvicorn doc_server.mcp_server:app --host 0.0.0.0 --port 8000
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

### Type Hints
- **Required**: All public functions and methods must have type hints
- **Style**: Use modern type hints (Python 3.10+ syntax)
- **Return Types**: Always specify return types, even for None returns

```python
from typing import List, Optional, Union

def process_document(
    file_path: Path,
    content_type: str,
    options: Optional[dict] = None,
) -> dict[str, any]:
    """Process a document and return structured data."""
    return {"processed": True, "content": "..."}
```

### Naming Conventions
- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Prefix with single underscore `_private_method`
- **Dunder methods**: Only use standard dunder methods

```python
class DocumentProcessor:
    MAX_FILE_SIZE = 10_000_000  # 10MB
    
    def __init__(self, config: Settings) -> None:
        self._config = config
    
    def _validate_file(self, file_path: Path) -> bool:
        return file_path.stat().st_size <= self.MAX_FILE_SIZE
```

### Error Handling
- **Specific Exceptions**: Use most specific exception types
- **Custom Exceptions**: Create domain-specific exception classes
- **Logging**: Use `structlog` for structured logging
- **Never Catch Bare `except`**: Always specify exception types

```python
import structlog

logger = structlog.get_logger()

class DocumentProcessingError(Exception):
    """Raised when document processing fails."""

def process_document(file_path: Path) -> dict:
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Processing logic
        return process_content(file_path.read_text())
    
    except FileNotFoundError as exc:
        logger.error("Document not found", path=str(file_path))
        raise DocumentProcessingError(f"Failed to process {file_path}") from exc
```

### Documentation Standards
- **Module Docstrings**: Triple quotes, brief description + purpose
- **Function Docstrings**: Google-style or NumPy-style
- **Type Hints in Docstrings**: Include in parameter descriptions

```python
def hybrid_search(
    query: str,
    top_k: int = 10,
    filters: Optional[dict] = None,
) -> list[dict]:
    """Perform hybrid search combining keyword and semantic matching.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        filters: Optional filtering criteria
        
    Returns:
        List of search results with relevance scores
        
    Raises:
        SearchError: When search service is unavailable
    """
    pass
```

### Code Organization
- **Single Responsibility**: Each function/class has one clear purpose
- **Small Functions**: Keep functions under 30 lines when possible
- **Dependency Injection**: Pass dependencies via constructor or parameters
- **Configuration**: Use Pydantic settings for configuration management

### Async/Await Patterns
- **Use AsyncIO**: For I/O-bound operations (HTTP requests, database queries)
- **Avoid Mixing**: Don't mix sync and async code unnecessarily
- **Error Handling**: Handle exceptions in async contexts properly

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
- **Test Location**: Place tests in `tests/` directory with `test_` prefix
- **Test Structure**: Arrange-Act-Assert pattern
- **Fixtures**: Use pytest fixtures for common setup
- **Async Tests**: Use `pytest-asyncio` for async function tests

```python
import pytest
from doc_server.search.hybrid_search import HybridSearch

@pytest.fixture
def search_engine():
    return HybridSearch(test_config)

async def test_hybrid_search_returns_results(search_engine):
    # Arrange
    query = "python async programming"
    
    # Act
    results = await search_engine.search(query, top_k=5)
    
    # Assert
    assert len(results) <= 5
    assert all(isinstance(r, dict) for r in results)
```

---

## Development Workflow

1. **Always run type checker**: `mypy doc_server` before commits
2. **Format code**: `black . && isort .` automatically
3. **Lint**: `ruff check --fix .` to fix issues
4. **Tests**: `pytest` to ensure functionality
5. **Check single file**: Use specific paths for faster iteration

---

## Repository Structure Notes

- **Main Package**: `doc_server/`
- **Entry Point**: `doc_server/mcp_server.py`
- **Configuration**: `doc_server/config.py` with Pydantic settings
- **Modular Design**: Separate `search/` and `ingestion/` subpackages
- **Version**: Managed in `doc_server/__init__.py` and `pyproject.toml`

---

## Tools Configuration Summary

- **Black**: 88-character line length, Python 3.10+ target
- **isort**: Black profile, first-party `doc_server` recognition
- **MyPy**: Strict type checking, untyped definitions disallowed
- **Ruff**: E, W, F, I, B, C4, UP rule sets (pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade)
- **pytest**: Auto asyncio mode, verbose output, short tracebacks