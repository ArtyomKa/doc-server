# Development Workflow

Development workflow and commands for doc-server.

## Environment Setup

```bash
# Install dependencies using uv
uv sync

# Install in development mode with dev dependencies
uv pip install -e ".[dev]"
```

## Quick Start with Make

The Makefile provides convenient shortcuts for common development tasks:

```bash
# Quick validation (pre-commit)
make ci

# Run specific commands
make format       # Format code
make lint         # Check code style
make test         # Run tests
make test-cov     # Run tests with coverage
make serve        # Run server
```

## Running Tests

> **Note**: The test suite takes a long time to run. Use increased timeout: `uv run pytest --timeout=300`

```bash
# Run all tests (with increased timeout)
uv run pytest --timeout=300

# Run specific test file
uv run pytest tests/test_git_cloner.py
uv run pytest tests/test_config.py

# Run single test
uv run pytest tests/test_git_cloner.py::TestGitClonerValidation::test_validate_url_valid_https

# Run with coverage
uv run pytest --cov=doc_server

# Run with specific options (configured in pyproject.toml)
uv run pytest -v --tb=short
```



## Quick Validation (Pre-commit)

```bash
# Run all checks at once (recommended)
make ci

# Or run manually
uv run black . && uv run isort . && uv run ruff check --fix . && uv run mypy doc_server && uv run pytest
```

## Common Patterns

### Running specific test classes
```bash
uv run pytest tests/test_git_cloner.py::TestGitClonerValidation
```

### Running specific test methods
```bash
uv run pytest tests/test_git_cloner.py::TestGitClonerValidation::test_validate_url_valid_https
```

### Running tests with verbose output for a specific file
```bash
uv run pytest -v tests/test_git_cloner.py
```

### Running tests with coverage for a specific module
```bash
uv run pytest --cov=doc_server.ingestion.git_cloner tests/test_git_cloner.py
```

## Manual Commands (Advanced)

For advanced users or when the Makefile is not available, you can use the manual uv commands directly:

### Code Formatting and Linting

```bash
# Format code with Black
uv run black .

# Sort imports with isort
uv run isort .

# Run comprehensive linting with ruff
uv run ruff check .

# Fix auto-fixable ruff issues
uv run ruff check --fix .
```

### Type Checking

```bash
# Run MyPy type checking
uv run mypy doc_server

# Check specific module
uv run mypy doc_server/ingestion/git_cloner.py
```

### Running the Server

```bash
# MCP server mode
uv run python -m doc_server.mcp_server

# HTTP API server (development)
uv run uvicorn doc_server.mcp_server:app --reload

# HTTP API server (production)
uv run uvicorn doc_server.mcp_server:app --host 0.0.0.0 --port 8000
```
