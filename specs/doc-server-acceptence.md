# doc-server Acceptance Criteria & Test Specifications

## Overview

This document provides comprehensive acceptance criteria and test specifications for each phase of the doc-server project. Tests are designed following MCP best practices, ChromaDB hybrid search patterns, and production-ready vector database standards.

## Testing Strategy

- **Unit Tests**: Individual component validation (pytest)
- **Integration Tests**: End-to-end workflow validation  
- **Performance Tests**: Vector search and ingestion benchmarks
- **MCP Protocol Tests**: FastMCP compliance validation
- **Security Tests**: Input validation and sanitization
- **Coverage Target**: >90% for critical path components

---

## Phase 1: Core Infrastructure Setup

### Acceptance Criteria

#### 1.1 Project Structure Setup
**AC-1.1.1**: Project follows Python packaging standards with `pyproject.toml`
**AC-1.1.2**: All modules have proper `__init__.py` files with version exports
**AC-1.1.3**: Directory structure follows FastMCP conventions with:
   - `doc_server/` package with `__init__.py` for proper imports
   - Entry point file: `mcp_server.py` at package root
   - Modular subpackages: `ingestion/` and `search/` with `__init__.py` files
   - Optional provider structure: `tools/`, `resources/` for `FileSystemProvider` discovery
   - Expected structure: `doc_server/{__init__.py,mcp_server.py,ingestion/__init__.py,search/__init__.py}`
**AC-1.1.4**: Entry points are properly configured in `pyproject.toml` with:
    - MCP server entry point: `[project.scripts] doc-server = "doc_server.mcp_server:main"`
    - Entry point is callable after `pip install -e .`
    - MCP server contains FastMCP server instance with main() function
**AC-1.1.5**: Development environment is reproducible with requirements.txt

#### 1.2 Dependencies & Environment
**AC-1.2.1**: All core dependencies install without conflicts on Python 3.10+
**AC-1.2.2**: FastMCP integration works with stdio transport
**AC-1.2.3**: sentence-transformers model all-MiniLM-L6-v2 downloads successfully
**AC-1.2.4**: ChromaDB persistent storage initializes without errors
**AC-1.2.5**: GitPython operations work for shallow cloning
**AC-1.2.6**: Development tools (pytest, black, ruff, mypy) run successfully

#### 1.3 Configuration System
**AC-1.3.1**: Configuration loads from environment variables with defaults
**AC-1.3.2**: Default storage path `~/.doc-server/` is created automatically
**AC-1.3.3**: Configuration validation prevents invalid settings
**AC-1.3.4**: All configuration options have type hints and validation
**AC-1.3.5**: Configuration changes persist across server restarts

### Test Specifications

#### Unit Tests
```python
# tests/test_config.py
def test_default_configuration_loading():
    """Test that default config loads with expected values"""
    
def test_environment_variable_override():
    """Test that environment variables override defaults"""
    
def test_configuration_validation():
    """Test that invalid config raises appropriate errors"""
    
def test_storage_path_creation():
    """Test that storage directory is created if missing"""

# tests/test_dependencies.py  
def test_fastmcp_import():
    """Test FastMCP imports and initializes correctly"""
    
def test_chromadb_connection():
    """Test ChromaDB client connection and collection creation"""
    
def test_embedding_model_availability():
    """Test sentence-transformers model loads successfully"""
```

#### Integration Tests
```python
# tests/test_project_setup.py
def test_package_installation():
    """Test package installs in development mode"""
    
def test_entry_point_execution():
    """Test CLI entry points are accessible"""
    
def test_python_version_compatibility():
    """Test compatibility with Python 3.10+"""

def test_directory_structure():
    """Test directory structure follows FastMCP conventions"""
    from pathlib import Path
    assert (Path("doc_server/__init__.py")).exists()
    assert (Path("doc_server/mcp_server.py")).exists()
    assert (Path("doc_server/cli.py")).exists()
    assert (Path("doc_server/ingestion/__init__.py")).exists()
    assert (Path("doc_server/search/__init__.py")).exists()

def test_entry_points_configuration():
    """Test entry points are properly configured"""
    # Test MCP server can be imported
    from doc_server.mcp_server import mcp
    assert mcp is not None
```

#### Performance Tests
```python
# tests/test_performance.py
def test_dependency_import_time():
    """Measure time to import all dependencies"""
    
def test_configuration_load_time():
    """Measure configuration loading performance"""
```

---

## Phase 2: Ingestion System

### Acceptance Criteria

#### 2.1 Git Cloner Module
**AC-2.1.1**: Successfully clones repositories via HTTPS/SSH with depth=1
**AC-2.1.2**: Extracts repository metadata (name, description, last commit)
**AC-2.1.3**: Handles invalid URLs with proper error messages
**AC-2.1.4**: Cleans up temporary clones after processing
**AC-2.1.5**: Works with private repositories using SSH keys

#### 2.2 ZIP Extractor Module
**AC-2.2.1**: Securely extracts ZIP archives without path traversal
**AC-2.2.2**: Extracts archive metadata (file count, compression type)
**AC-2.2.3**: Handles password-protected ZIPs appropriately
**AC-2.2.4**: Validates archive integrity before extraction
**AC-2.2.5**: Preserves original file permissions when possible

#### 2.3 File Filter Module
**AC-2.3.1**: Parses .gitignore files correctly with all pattern types
**AC-2.3.2**: Enforces allowlist for file types: md, rst, txt, py, pyi, cpp, h, hpp, c
**AC-2.3.3**: Skips files >1MB with logging
**AC-2.3.4**: Detects and excludes binary files (null bytes)
**AC-2.3.5**: Preserves relative file paths for metadata

#### 2.4 Document Processor Module
**AC-2.4.1**: Adds metadata headers with file path, line numbers, library ID
**AC-2.4.2**: Chunks large files (>2KB) intelligently on sentence boundaries
**AC-2.4.3**: Preserves code formatting and special characters
**AC-2.4.4**: Optimizes content for embedding generation
**AC-2.4.5**: Handles different file encodings (UTF-8, Latin-1)

### Test Specifications

#### Unit Tests
```python
# tests/test_git_cloner.py
def test_shallow_clone_https():
    """Test shallow clone via HTTPS"""
    
def test_shallow_clone_ssh():
    """Test shallow clone via SSH"""
    
def test_invalid_url_handling():
    """Test error handling for invalid URLs"""
    
def test_repository_metadata_extraction():
    """Test extraction of repo metadata"""
    
def test_cleanup_temporary_clones():
    """Test cleanup of temporary directories"""

# tests/test_zip_extractor.py
def test_secure_zip_extraction():
    """Test ZIP extraction without path traversal"""
    
def test_archive_metadata_extraction():
    """Test extraction of archive metadata"""
    
def test_password_protected_zip():
    """Test handling of password-protected archives"""
    
def test_corrupted_zip_handling():
    """Test error handling for corrupted ZIPs"""

# tests/test_file_filter.py
def test_gitignore_parsing():
    """Test .gitignore pattern parsing"""
    
def test_allowlist_enforcement():
    """Test file type allowlist enforcement"""
    
def test_large_file_skip():
    """Test skipping files >1MB"""
    
def test_binary_file_detection():
    """Test binary file detection via null bytes"""
    
def test_relative_path_preservation():
    """Test preservation of relative paths"""

# tests/test_document_processor.py
def test_metadata_headers():
    """Test addition of metadata headers"""
    
def test_intelligent_chunking():
    """Test intelligent file chunking"""
    
def test_code_formatting_preservation():
    """Test preservation of code formatting"""
    
def test_embedding_optimization():
    """Test content optimization for embeddings"""
    
def test_encoding_handling():
    """Test handling of different file encodings"""
```

#### Integration Tests
```python
# tests/test_ingestion_integration.py
def test_end_to_end_git_ingestion():
    """Test complete git repository ingestion"""
    
def test_end_to_end_zip_ingestion():
    """Test complete ZIP archive ingestion"""
    
def test_large_repository_ingestion():
    """Test ingestion of large repository"""
    
def test_error_recovery():
    """Test recovery from ingestion errors"""
```

#### Performance Tests
```python
# tests/test_ingestion_performance.py
def test_ingestion_throughput():
    """Benchmark files/second ingestion rate"""
    
def test_memory_usage():
    """Monitor memory usage during ingestion"""
    
def test_large_file_handling():
    """Test performance with large files"""
```

---

## Phase 3: Search Infrastructure

### Acceptance Criteria

#### 3.1 Embedding Service
**AC-3.1.1**: Generates embeddings using all-MiniLM-L6-v2 consistently
**AC-3.1.2**: Processes embeddings in batches (default 32 docs) efficiently
**AC-3.1.3**: Caches embeddings with configurable TTL (default 24h)
**AC-3.1.4**: Warms up model on startup (<5 seconds)
**AC-3.1.5**: Handles embedding generation failures gracefully

#### 3.2 Vector Store Module
**AC-3.2.1**: Manages ChromaDB collections by library ID correctly
**AC-3.2.2**: Provides persistent storage across server restarts
**AC-3.2.3**: Implements collection CRUD operations
**AC-3.2.4**: Configures HNSW parameters for optimal performance
**AC-3.2.5**: Handles collection size limits and scaling

#### 3.3 Hybrid Search Module
**AC-3.3.1**: Combines vector similarity with keyword matching using a ranking/fusion algorithm (e.g., RRF, weighted sum, or similar)
**AC-3.3.2**: Boosts exact term matches in search results
**AC-3.3.3**: Implements configurable search weights (vector: 0.7, keyword: 0.3)
**AC-3.3.4**: Enriches results with metadata (scores, line numbers, file paths)
**AC-3.3.5**: Handles empty queries and invalid search parameters

### Test Specifications

#### Unit Tests
```python
# tests/test_embedding_service.py
def test_embedding_generation():
    """Test embedding generation with sample text"""
    
def test_batch_processing():
    """Test batch embedding processing"""
    
def test_embedding_caching():
    """Test embedding cache functionality"""
    
def test_model_warmup():
    """Test model warmup on startup"""
    
def test_embedding_failure_handling():
    """Test graceful handling of embedding failures"""

# tests/test_vector_store.py
def test_collection_creation():
    """Test ChromaDB collection creation"""
    
def test_persistence_across_restarts():
    """Test data persistence across server restarts"""
    
def test_collection_crud_operations():
    """Test create, read, update, delete operations"""
    
def test_hnsw_configuration():
    """Test HNSW parameter configuration"""
    
def test_scaling_behavior():
    """Test behavior with large collections"""

# tests/test_hybrid_search.py
def test_vector_similarity_search():
    """Test pure vector similarity search"""
    
def test_keyword_search():
    """Test pure keyword/BM25 search"""
    
def test_hybrid_rrf_fusion():
    """Test hybrid search with RRF fusion"""
    
def test_exact_term_boost():
    """Test boosting of exact term matches"""
    
def test_search_result_metadata():
    """Test search result metadata enrichment"""
```

#### Integration Tests
```python
# tests/test_search_integration.py
def test_end_to_end_search():
    """Test complete search pipeline"""
    
def test_multi_library_search():
    """Test search across multiple libraries"""
    
def test_search_consistency():
    """Test search result consistency"""
    
def test_search_performance():
    """Benchmark search response times"""
```

#### Performance Tests
```python
# tests/test_search_performance.py
def test_search_latency():
    """Measure search response times (<100ms target)"""
    
def test_concurrent_searches():
    """Test concurrent search requests"""
    
def test_memory_efficiency():
    """Monitor memory usage during search"""
    
def test_index_rebuilding():
    """Test index rebuilding performance"""
```

#### Search Quality Tests
```python
# tests/test_search_quality.py
def test_semantic_relevance():
    """Test semantic search relevance using sample queries"""
    
def test_exact_term_precision():
    """Test precision for exact term searches"""
    
def test_recall_metrics():
    """Measure recall for known documents"""
```

---

## Phase 4: MCP Server Implementation

### Acceptance Criteria

#### 4.1 Core MCP Server
**AC-4.1.1**: Implements FastMCP server with stdio transport
**AC-4.1.2**: Exposes `search_docs(query, library_id)` tool with proper validation
**AC-4.1.3**: Handles JSON-RPC 2.0 protocol correctly
**AC-4.1.4**: Implements proper error handling and logging
**AC-4.1.5**: Supports graceful shutdown on SIGTERM/SIGINT

#### 4.2 Library Management Tools
**AC-4.2.1**: Implements `ingest_library(source, library_id)` with input validation
**AC-4.2.2**: Implements `list_libraries()` returning all available library IDs
**AC-4.2.3**: Implements `remove_library(library_id)` with cascade deletion
**AC-4.2.4**: Sanitizes all inputs to prevent injection attacks
**AC-4.2.5**: Provides progress feedback for long-running operations

#### 4.3 Server Configuration
**AC-4.3.1**: Configures structured logging with appropriate levels
**AC-4.3.2**: Validates all dependencies on startup
**AC-4.3.3**: Implements health check for monitoring
**AC-4.3.4**: Supports performance monitoring hooks
**AC-4.3.5**: Compatible with major MCP clients (Claude Desktop, etc.)

### Test Specifications

#### MCP Protocol Tests
```python
# tests/test_mcp_protocol.py
def test_json_rpc_compliance():
    """Test JSON-RPC 2.0 message format compliance"""
    
def test_tool_registration():
    """Test proper tool registration with FastMCP"""
    
def test_stdio_transport():
    """Test stdio transport communication"""
    
def test_error_code_mapping():
    """Test MCP error code compliance"""
    
def test_capabilities_advertisement():
    """Test server capabilities advertisement"""
```

#### Unit Tests
```python
# tests/test_mcp_server.py
def test_search_docs_tool():
    """Test search_docs tool functionality"""
    
def test_ingest_library_tool():
    """Test ingest_library tool functionality"""
    
def test_list_libraries_tool():
    """Test list_libraries tool functionality"""
    
def test_remove_library_tool():
    """Test remove_library tool functionality"""
    
def test_input_validation():
    """Test input validation and sanitization"""

# tests/test_server_configuration.py
def test_logging_configuration():
    """Test structured logging setup"""
    
def test_startup_validation():
    """Test startup dependency validation"""
    
def test_health_check():
    """Test health check endpoint"""
    
def test_graceful_shutdown():
    """Test graceful shutdown handling"""
```

#### Integration Tests
```python
# tests/test_mcp_integration.py
def test_client_server_communication():
    """Test communication with MCP clients"""
    
def test_tool_execution_workflow():
    """Test complete tool execution workflows"""
    
def test_error_propagation():
    """Test error propagation to clients"""
    
def test_concurrent_requests():
    """Test handling concurrent client requests"""
```

#### Security Tests
```python
# tests/test_mcp_security.py
def test_input_sanitization():
    """Test input sanitization against injection"""
    
def test_library_id_validation():
    """Test library ID format validation"""
    
def test_query_parameter_validation():
    """Test query parameter validation"""
    
def test_authentication_scope():
    """Test operation scope boundaries"""
```

---

## Phase 8: CLI Interface

### Acceptance Criteria

#### 8.1 Command Line Interface
**AC-5.1.1**: Implements all commands: ingest, search, list, remove, serve
**AC-5.1.2**: Provides comprehensive help text and usage examples
**AC-5.1.3**: Supports configuration via command-line options
**AC-5.1.4**: Validates command arguments with helpful error messages
**AC-5.1.5**: Integrates with shell completion where possible

#### 8.2 User Experience
**AC-5.2.1**: Shows progress bars for operations >2 seconds
**AC-5.2.2**: Formats search results for readable terminal output
**AC-5.2.3**: Uses colored output for better readability
**AC-5.2.4**: Provides clear error messages with suggested solutions
**AC-5.2.5**: Maintains consistent command patterns and argument names

### Test Specifications

#### Unit Tests
```python
# tests/test_cli_commands.py
def test_ingest_command():
    """Test CLI ingest command"""
    
def test_search_command():
    """Test CLI search command"""
    
def test_list_command():
    """Test CLI list command"""
    
def test_remove_command():
    """Test CLI remove command"""
    
def test_serve_command():
    """Test CLI serve command"""

# tests/test_cli_ux.py
def test_progress_bars():
    """Test progress bar display"""
    
def test_result_formatting():
    """Test search result formatting"""
    
def test_colored_output():
    """Test colored output functionality"""
    
def test_error_messages():
    """Test error message clarity"""
```

#### Integration Tests
```python
# tests/test_cli_integration.py
def test_complete_cli_workflow():
    """Test complete CLI workflow"""
    
def test_command_help_system():
    """Test help system functionality"""
    
def test_shell_completion():
    """Test shell completion integration"""
```

---

## Phase 6: Testing & Validation

### Acceptance Criteria

#### 6.1 Unit Test Suite
**AC-6.1.1**: Achieves >90% code coverage for critical path components
**AC-6.1.2**: All tests pass on Python 3.10+ across platforms
**AC-6.1.3**: Tests execute in <30 seconds total
**AC-6.1.4**: Tests use proper fixtures and setup/teardown
**AC-6.1.5**: Test data is isolated and reproducible

##### 6.1.1 ChromaDB Compatibility & Search Implementation
**AC-6.1.1.1**: ChromaEmbeddingFunction implements all required ChromaDB interface methods
**AC-6.1.1.2**: Vector store operations work without type errors
**AC-6.1.1.3**: ChromaDB collection creation and management functions correctly
**AC-6.1.1.4**: Proper error handling for ChromaDB operation failures
**AC-6.1.1.5**: Compatibility with current ChromaDB API version
**AC-6.1.1.6**: Algorithms library ingestion completes successfully with >400 documents
**AC-6.1.1.7**: Search queries return relevant results from algorithms documentation
**AC-6.1.1.8**: Search results include proper metadata (file paths, relevance scores, line numbers)
**AC-6.1.1.9**: Hybrid search combines vector similarity and keyword matching effectively
**AC-6.1.1.10**: Search performance meets <500ms response time targets
**AC-6.1.1.11**: End-to-end ingestion workflow works from git clone to vector storage
**AC-6.1.1.12**: Document persistence verified across server restarts
**AC-6.1.1.13**: MCP server search tool responds correctly to queries
**AC-6.1.1.14**: Search functionality tested with sample queries (binary search, sorting algorithms)
**AC-6.1.1.15**: Error scenarios handled gracefully with informative messages

#### 6.2 Integration Tests
**AC-6.2.1**: End-to-end ingestion works with pandas and fastapi repositories
**AC-6.2.2**: Search accuracy validated against known document sets
**AC-6.2.3**: Performance benchmarks meet targets (ingestion: 100 files/min, search: <100ms)
**AC-6.2.4**: Error scenarios handled gracefully with proper logging
**AC-6.2.5**: Multiple MCP clients can connect and operate simultaneously

#### 6.3 Test Infrastructure
**AC-6.3.1**: Pytest configured with appropriate markers and fixtures
**AC-6.3.2**: Test data fixtures for repositories and documents
**AC-6.3.3**: CI/CD pipeline runs all tests automatically
**AC-6.3.4**: Test documentation covers running and extending tests
**AC-6.3.5**: Performance regression tests prevent degradations

### Test Specifications

#### Coverage Tests
```python
# tests/test_coverage.py
def test_critical_path_coverage():
    """Verify >90% coverage for critical modules"""
    
def test_edge_case_coverage():
    """Verify coverage of error handling paths"""
```

#### End-to-End Tests
```python
# tests/test_e2e.py
def test_pandas_repository_ingestion():
    """Test ingestion of pandas documentation"""
    
def test_fastapi_search_accuracy():
    """Test search accuracy on FastAPI docs"""
    
def test_mcp_client_compatibility():
    """Test compatibility with various MCP clients"""
```

#### Benchmark Tests
```python
# tests/test_benchmarks.py
def test_ingestion_throughput_benchmark():
    """Benchmark ingestion performance"""
    
def test_search_latency_benchmark():
    """Benchmark search response times"""
    
def test_memory_usage_benchmark():
    """Benchmark memory consumption"""
```

---

## Phase 7: Documentation & Examples

### Acceptance Criteria

#### 7.1 User Documentation
**AC-7.1.1**: README provides clear installation and quick start guide
**AC-7.1.2**: Usage examples cover all major functionality
**AC-7.1.3**: Configuration options are documented with examples
**AC-7.1.4**: Troubleshooting guide covers common issues
**AC-7.1.5**: Quick start guide gets users running in <5 minutes

#### 7.2 Developer Documentation
**AC-7.2.1**: Complete API reference with examples
**AC-7.2.2**: Architecture documentation explains design decisions
**AC-7.2.3**: Contributing guidelines with development setup
**AC-7.2.4**: Extension points and plugin system documented
**AC-7.2.5**: Code examples for common use cases provided

### Test Specifications

#### Documentation Tests
```python
# tests/test_documentation.py
def test_readme_examples():
    """Test all README code examples"""
    
def test_api_documentation_examples():
    """Test API documentation examples"""
    
def test_configuration_examples():
    """Test configuration examples"""
```

---

## Testing Infrastructure

### Test Environment Setup

#### Continuous Integration
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          pytest --cov=doc_server --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

#### Local Test Configuration
```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
  --verbose
  --tb=short
  --cov=doc_server
  --cov-report=term-missing
  --cov-report=html
  --cov-fail-under=90
markers =
  unit: Unit tests
  integration: Integration tests
  performance: Performance tests
  security: Security tests
  mcp: MCP protocol tests
```

### Test Data Management

#### Fixtures
```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def sample_repository():
    """Sample repository for testing"""
    
@pytest.fixture
def test_documents():
    """Test documents for search testing"""
    
@pytest.fixture
def temp_storage_dir(tmp_path):
    """Temporary storage directory"""
```

### Quality Gates

#### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
```

#### Performance Thresholds
- **Ingestion**: >100 files/minute
- **Search**: <100ms response time (p95)
- **Memory**: <500MB for typical workloads
- **Startup**: <5 seconds server startup time

This comprehensive acceptance criteria and test specification ensures doc-server meets production quality standards while maintaining clear validation requirements for each development phase.