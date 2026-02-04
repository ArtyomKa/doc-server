# Development Scripts

This directory contains development and maintenance scripts for the doc-server project.

## Script Categories

| Type | Purpose | Relationship to pytest |
|------|---------|------------------------|
| **Utility Scripts** | Development/debugging tools | Independent of pytest |
| **Verification Scripts** | AC criteria validation | Reads pytest results |

## Utility Scripts

### verify_components.py
A manual verification script to test that core doc-server components are working correctly.

**Purpose:**
- Quick validation of component connectivity
- Development-time debugging and troubleshooting
- Manual testing without running the full pytest suite

**Usage:**
```bash
# From project root
python scripts/verify_components.py

# Or from scripts directory
python verify_components.py
```

**What it tests:**
1. Configuration loading from Settings
2. File filtering functionality
3. Document processing and chunking
4. Embedding service connectivity

**Output:**
- Real-time status updates for each component
- Success/failure indicators
- Summary of component health

### ingest_algorithms.py
Direct ingestion script to bypass MCP caching for the algorithms library.

**Purpose:**
- Provides a command-line interface for ingesting documentation from GitHub repositories
- Bypasses the MCP (Model Context Protocol) server caching layer
- Enables direct testing of the ingestion pipeline

**Usage:**
```bash
# From project root
python scripts/ingest_algorithms.py
```

**What it tests:**
- Git repository cloning with shallow clone
- File filtering and .gitignore parsing
- Document processing and chunking
- Embedding generation via EmbeddingService
- ChromaDB vector storage and collection management

**Input:**
- GitHub URL: https://github.com/keon/algorithms
- Library ID: /algorithms

**Output:**
- Success: Ingestion result dictionary with document count
- Failure: Exception with full traceback

## Verification Scripts

### verify_vector_store.py
Verification script for Phase 3.2 (Vector Store Module) acceptance criteria.

**Purpose:**
- Validates ChromaDB vector store implementation against AC-3.2 requirements
- Provides manual verification complementing automated tests
- Documents expected behavior for each acceptance criterion

**Usage:**
```bash
# From project root
python scripts/verify_vector_store.py
```

**Exit Codes:**
- 0: All acceptance criteria met
- 1: One or more acceptance criteria failed

**What it validates:**
1. **AC-3.2.1**: Collection Management by Library ID
   - Verifies collections are created and retrieved by normalized library ID
   - Tests library ID normalization

2. **AC-3.2.2**: Persistent Storage
   - Validates data persists across ChromaVectorStore instances
   - Simulates server restart scenarios

3. **AC-3.2.3**: Collection CRUD Operations
   - CREATE, READ, UPDATE, DELETE operations
   - Document and collection lifecycle

4. **AC-3.2.4**: HNSW Configuration
   - Verifies ChromaDB's HNSW index functionality
   - Vector similarity search validation

5. **AC-3.2.5**: Scaling Limits
   - Tests batch processing with 500 documents
   - Collection count and query performance

6. **Integration Tests**:
   - EmbeddingService integration via ChromaEmbeddingFunction
   - Error handling and custom exception hierarchy

### verify_phase_62.py
Phase 6.2 Integration Tests Acceptance Criteria Verification Summary.

**Purpose:**
- Validates that all AC-6.2 criteria are met by referencing integration test results
- Provides a summary report of test outcomes
- Documents the relationship between integration tests and acceptance criteria

**Usage:**
```bash
# From project root
python scripts/verify_phase_62.py
```

**Exit Codes:**
- 0: All acceptance criteria met
- 1: One or more acceptance criteria failed

**What it validates:**

1. **AC-6.2.1**: End-to-End Ingestion
   - Full pipeline from git clone → file filter → chunk → embed → store
   - Tests: pandas, fastapi, algorithms repositories

2. **AC-6.2.2**: Search Accuracy
   - Vector similarity search functionality
   - ChromaDB collection management

3. **AC-6.2.3**: Performance Benchmarks
   - Document processing: >10 docs/min
   - Large file chunking: <5 seconds for 100KB files
   - Directory filtering: <10 seconds for 100 files

4. **AC-6.2.4**: Error Handling
   - DocumentResult error handling
   - Empty/non-existent library operations
   - Error logging and graceful degradation

5. **AC-6.2.5**: Multiple Libraries
   - Creating, listing, querying, and deleting multiple libraries
   - Library isolation and lifecycle management

## Running Tests

### Automated Testing (Recommended)
For the full automated test suite:
```bash
pytest tests/
```

### Manual Verification
For quick component validation:
```bash
python scripts/verify_components.py
```

### Phase-Specific Verification
For acceptance criteria validation:
```bash
# Phase 3.2 (Vector Store)
python scripts/verify_vector_store.py

# Phase 6.2 (Integration Tests)
python scripts/verify_phase_62.py
```

### Direct Ingestion
For testing ingestion pipeline:
```bash
python scripts/ingest_algorithms.py
```

## Notes

- **Utility scripts** are designed for development-time debugging and quick validation
- **Verification scripts** complement the automated pytest suite by providing targeted AC validation
- All scripts are independent of the pytest framework and can be run directly
- For comprehensive testing, always use `pytest tests/` which includes all unit and integration tests
