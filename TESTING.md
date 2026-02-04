# Testing Guide

This document provides comprehensive information about running and extending tests for the doc-server project.

## Quick Start

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=doc_server --cov-report=term-missing

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::test_default_configuration_loading
```

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_config.py           # Configuration tests
├── test_git_cloner.py      # Git cloning tests
├── test_zip_extractor.py    # ZIP extraction tests
├── test_file_filter.py      # File filtering tests
├── test_document_processor.py  # Document processing tests
├── test_embedding_service.py   # Embedding service tests
├── test_vector_store.py     # Vector storage tests
├── test_hybrid_search.py   # Search functionality tests
├── test_mcp_server.py      # MCP server tests
├── test_mcp_server_config.py  # Server configuration tests
├── test_logging_config.py  # Logging configuration tests
└── test_integration.py      # End-to-end integration tests
```

## Running Tests

### By Category

```bash
# Unit tests (default)
pytest tests/ -v

# Integration tests
pytest tests/test_integration.py -v

# Performance tests
pytest tests/ -k "performance" -v
```

### With Markers

```bash
# Run unit tests only
pytest -m unit -v

# Run integration tests only
pytest -m integration -v

# Run performance tests only
pytest -m performance -v

# Run MCP protocol tests
pytest -m mcp -v

# Run security tests
pytest -m security -v
```

### Coverage Requirements

The project requires >90% coverage for critical path components:

```bash
# Check coverage
pytest --cov=doc_server --cov-fail-under=90

# Generate HTML coverage report
pytest --cov=doc_server --cov-report=html
open htmlcov/index.html
```

### Performance Benchmarks

```bash
# Run all performance tests
pytest -k "performance" --verbose

# Specific benchmark
pytest tests/ -k "benchmark" --verbose
```

## Test Fixtures

The `conftest.py` provides these fixtures:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `config` | function | Settings instance for testing |
| `temp_storage_dir` | function | Temporary storage directory |
| `sample_repository` | function | Sample Python repository |
| `algorithms_repository` | function | Algorithms repository for search testing |
| `test_documents` | function | Test documents for search |
| `mock_embedding_model` | function | Mock embedding model |
| `mock_chromadb_collection` | function | Mock ChromaDB collection |
| `performance_monitor` | function | Performance benchmarking helper |
| `isolated_environment` | function | Isolated test environment |

### Using Fixtures

```python
def test_with_repository(sample_repository):
    # sample_repository provides a Path to a temporary repo
    assert (sample_repository / "README.md").exists()

def test_with_documents(test_documents):
    assert len(test_documents) == 5
    assert test_documents[0]["id"] == "doc1"

def test_performance(performance_monitor):
    def expensive_operation():
        return sum(range(1000))

    result, elapsed = performance_monitor.measure("sum", expensive_operation)
    assert elapsed < 1.0  # Should complete in under 1 second
```

## Writing New Tests

### Test Structure

```python
import pytest
from doc_server.module import MyClass

class TestMyClass:
    """Test suite for MyClass."""

    @pytest.fixture
    def my_instance(self):
        """Create a MyClass instance for testing."""
        return MyClass(param="test")

    def test_method_behavior(self, my_instance):
        """Test that method returns expected result."""
        result = my_instance.method()
        assert result == expected_value

    def test_edge_case(self, my_instance):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            my_instance.invalid_method()
```

### Test Naming

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<behavior>`

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/test.yml`) runs:

1. **Lint** - Black, isort, Ruff, MyPy
2. **Test** - Pytest on Python 3.10, 3.11, 3.12
3. **Coverage** - Verify >90% coverage
4. **Performance** - Run benchmarks on main branch
5. **Integration** - Test on pull requests
6. **Security** - Bandit and Safety scans

### Local CI Simulation

```bash
# Run linting
black --check doc_server/ tests/
isort --check-only doc_server/ tests/
ruff check doc_server/ tests/
mypy doc_server/ tests/ --ignore-missing-imports

# Run security scan
bandit -r doc_server/
safety check -r requirements.txt
```

## Continuous Integration locally

```bash
# Install pre-commit hooks
uv pip install pre-commit
pre-commit install

# Run all checks manually
pre-commit run --all-files
```

## Troubleshooting

### Tests timing out

```bash
# Increase timeout for slow tests
pytest --timeout=300 tests/

# Run specific slow tests with longer timeout
pytest tests/test_embedding_service.py -v --timeout=600
```

### Coverage not reaching 90%

```bash
# See which lines are missing coverage
pytest --cov=doc_server --cov-report=term-missing

# Generate detailed HTML report
pytest --cov=doc_server --cov-report=html
```

### Memory issues with ChromaDB tests

```bash
# Run tests sequentially
pytest tests/test_vector_store.py -v -p no:randomly

# Clear ChromaDB cache
rm -rf ~/.cache/chroma
```

## Performance Targets

| Metric | Target | Test Location |
|--------|--------|--------------|
| Test suite runtime | <30 seconds | All tests |
| Search latency | <100ms | test_hybrid_search.py |
| Ingestion throughput | >100 files/min | test_integration.py |
| Memory usage | <500MB | performance markers |

## Adding New Test Categories

1. Add marker to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "performance: Performance tests",
    "security: Security tests",
    "mcp: MCP protocol tests",
    "new_category: Description here",
]
```

2. Write tests with the new marker:

```python
@pytest.mark.new_category
def test_new_feature():
    pass
```

3. Run with marker:

```bash
pytest -m new_category -v
```
