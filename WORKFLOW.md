# Development Workflow

Development workflow and commands for doc-server.

## Environment Setup

```bash
# Install dependencies using uv
uv sync

# Install in development mode with dev dependencies
uv pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
uv run pytest

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

## Code Formatting and Linting

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

## Type Checking

```bash
# Run MyPy type checking
uv run mypy doc_server

# Check specific module
uv run mypy doc_server/ingestion/git_cloner.py
```

## Running the Server

```bash
# MCP server mode
uv run python -m doc_server.mcp_server

# HTTP API server (development)
uv run uvicorn doc_server.mcp_server:app --reload

# HTTP API server (production)
uv run uvicorn doc_server.mcp_server:app --host 0.0.0.0 --port 8000
```

## Development Workflow

1. **Type check**: `uv run mypy doc_server` before commits
2. **Format**: `uv run black . && uv run isort .`
3. **Lint**: `uv run ruff check --fix .`
4. **Test**: `uv run pytest` for functionality
5. **Iterate**: Use `uv run pytest tests/test_module.py::test_function` for faster feedback

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
