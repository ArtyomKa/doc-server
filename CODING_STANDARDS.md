# Coding Standards

Code style guidelines for doc-server development.

## Formatting Standards

- **Line Length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **String Quotes**: Double quotes for docstrings, single quotes for string literals
- **Trailing Commas**: Required in multi-line function calls and data structures

## Import Organization

```python
# Standard library imports
import os
from pathlib import Path

# Third-party imports
import structlog
from git import Repo
from pydantic_settings import BaseSettings

# Local imports
from doc_server.config import Settings
from doc_server.ingestion.git_cloner import GitCloner
```

Use `isort` to maintain import order automatically.

## Type Hints & Naming

- **Required**: All public functions need type hints (Python 3.10+ syntax: `list[str]`, `dict[str, int]`)
- **Return Types**: Always specify return types, even for None returns
- **Variables/Functions**: `snake_case` • **Classes**: `PascalCase` • **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Prefix with single underscore `_private_method`
- **Union types**: Use `X | Y` syntax instead of `Union[X, Y]`

```python
from pathlib import Path

def process_document(file_path: Path, options: dict[str, str] | None = None) -> dict[str, any]:
    """Process a document and return structured data."""
    if options is None:
        options = {}
    return {"processed": True, "content": "..."}

class DocumentProcessor:
    MAX_FILE_SIZE = 10_000_000  # 10MB

    def __init__(self, config: Settings) -> None:
        self._config = config
```

## Error Handling

- **Specific Exceptions**: Use most specific exception types, never bare `except`
- **Custom Exceptions**: Create domain-specific exception classes inheriting from Exception
- **Logging**: Use `structlog` for structured logging with context
- **Exception Chaining**: Always use `raise ... from exc` to preserve traceback

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

## Testing Guidelines

- **Location**: `tests/` directory with `test_` prefix
- **Structure**: Arrange-Act-Assert pattern with pytest fixtures
- **Mocking**: Use `unittest.mock` for external dependencies
- **Async Tests**: Use `pytest-asyncio` for async functions

```python
import pytest
from doc_server.ingestion.git_cloner import GitCloner

@pytest.fixture
def git_cloner():
    return GitCloner(Settings())

def test_clone_validates_url(git_cloner):
    git_cloner._validate_url("https://github.com/test/repo.git")

def test_clone_with_invalid_url_raises_error(git_cloner):
    with pytest.raises(InvalidURLError):
        git_cloner._validate_url("invalid-url")
```

## Docstring Guidelines

- Use Google-style docstrings for clarity
- Include sections for Args, Returns, and Raises when applicable
- Keep descriptions concise but informative

```python
def clone_repository(
    self,
    clone_url: str,
    destination: Path | None = None,
    branch: str | None = None,
    shallow: bool = True,
    depth: int = 1,
) -> Repo:
    """
    Clone a git repository with optional shallow clone.

    Args:
        clone_url: URL of the git repository to clone
        destination: Path where repository should be cloned. If None,
                    creates a temporary directory that needs to be cleaned up.
        branch: Specific branch to clone. If None, clones default branch.
        shallow: Whether to perform a shallow clone (no history)
        depth: Depth of shallow clone. 1 means only latest commit.

    Returns:
        Repo: GitPython Repo object for the cloned repository

    Raises:
        InvalidURLError: If the clone URL is invalid
        GitCloneError: If cloning fails for any reason
    """
    # Implementation...
```

## Logging Guidelines

- Use `structlog` for structured logging
- Include relevant context in log messages
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)

```python
import structlog
logger = structlog.get_logger()

def process_document(file_path: Path):
    logger.info(
        "Starting document processing",
        file_path=str(file_path),
        file_size=file_path.stat().st_size,
    )

    try:
        # Processing logic...
        logger.info("Document processed successfully", file_path=str(file_path))
    except Exception as exc:
        logger.error(
            "Failed to process document",
            file_path=str(file_path),
            error=str(exc),
        )
        raise
```
