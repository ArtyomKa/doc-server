# AGENTS.md

Development guide for agentic coding agents working on the doc-server repository.

## Project Overview

Doc Server is an AI-powered documentation management system with intelligent search capabilities. It provides MCP (Model Context Protocol) server implementation and REST API for ingesting, processing, and searching technical documentation.

**Technology Stack**: Python 3.10+, FastAPI, MCP, ChromaDB/FAISS, OpenAI embeddings, Pydantic

**Current Implementation State (Phase 7.1 Complete)**:
- ✅ Configuration management with Pydantic settings and YAML support
- ✅ Git repository cloning with shallow clone and metadata extraction
- ✅ ZIP archive extraction with security measures
- ✅ File filtering with .gitignore parsing and allowlist enforcement
- ✅ Document processor with chunking and encoding handling
- ✅ MCP server implementation with search and library management tools
- ✅ Search layer (embeddings, vector store, hybrid search)
- ✅ Unit test suite (492 tests, 100% passing) - Phase 6.1 Complete
- ✅ ChromaDB compatibility with full EmbeddingFunction API implementation
- ✅ Search functionality with 634 documents ingested (algorithms library)
- ✅ Hybrid search performance (9.54ms avg vs 500ms target)
- ✅ End-to-end ingestion workflow and document persistence
- ✅ All 15 Phase 6.1.1 acceptance criteria verified by @oracle specialist
- ✅ Test infrastructure (fixtures, sample repos, CI/CD) - Phase 6.3 Complete
- ✅ CLI interface with ingest, search, list, remove, serve, health commands - Phase 8.1 Complete
- ✅ CLI features: progress bars, colored output, multiple output formats (table/json/simple)
- ✅ User documentation (README, quickstart, installation, configuration) - Phase 7.1 Complete
- ✅ CLI reference documentation with examples
- ✅ MCP tools reference documentation with JSON examples
- ✅ Architecture documentation with Mermaid diagrams
- ✅ Troubleshooting guide and usage examples

---

## Quick Reference

**Development Workflow**: See [WORKFLOW.md](WORKFLOW.md) for setup, testing, and build commands

**Testing Guide**: See [TESTING.md](TESTING.md) for running tests, fixtures, and CI/CD information

**Code Style Guidelines**: See [CODING_STANDARDS.md](CODING_STANDARDS.md) for formatting, types, and testing patterns

---

## Repository Structure

```
doc_server/
├── config.py              # ✅ Pydantic settings with YAML support
├── cli.py                 # ✅ CLI interface with Click framework
├── mcp_server.py          # ✅ MCP server entry point with tools
├── utils.py               # ⏳ Utility functions (skeleton)
├── logging_config.py      # ✅ Structured logging with structlog
├── ingestion/
│   ├── git_cloner.py      # ✅ Git repository cloning & metadata
│   ├── document_processor.py  # ✅ Document content extraction & chunking
│   ├── file_filter.py     # ✅ File filtering with .gitignore parsing
│   └── zip_extractor.py   # ✅ ZIP archive extraction with security
└── search/
    ├── embedding_service.py   # ✅ Embedding generation with caching
    ├── vector_store.py    # ✅ ChromaDB vector storage management
    └── hybrid_search.py   # ✅ Hybrid search with BM25 and fusion

tests/
├── test_config.py         # ✅ Full coverage
├── test_cli.py            # ✅ CLI command tests (27 tests)
├── test_git_cloner.py     # ✅ Full coverage
├── test_zip_extractor.py  # ✅ Full coverage
├── test_file_filter.py    # ✅ Full coverage (88% coverage)
├── test_document_processor.py  # ✅ Full coverage
├── test_document_processor_extra.py  # ✅ Additional coverage tests
├── test_embedding_service.py   # ✅ Full coverage (84% coverage)
├── test_vector_store.py    # ✅ Full coverage
├── test_hybrid_search.py   # ✅ Full coverage
├── test_mcp_server.py     # ✅ Full coverage (comprehensive error handling)
├── test_logging_config.py # ✅ Full coverage
├── test_mcp_server_config.py # ✅ Full coverage
└── conftest.py            # ✅ Comprehensive fixtures and test infrastructure

**Test Suite**: 492 tests (100% passing) with comprehensive fixtures and CI/CD pipeline
```

---

## Tools Configuration

- **Black**: 88-char line length, Python 3.10+ target
- **isort**: Black profile, `doc_server` as first-party
- **MyPy**: Strict type checking, no untyped definitions
- **Ruff**: E, W, F, I, B, C4, UP rules (ignores E501, B008)
- **pytest**: Auto asyncio mode, verbose output, short tracebacks

---

## Security Scanning (Future)

**Note**: Security scanning tools (Bandit, Safety) are planned for future implementation. The CI/CD pipeline and Makefile will be updated to include:
- Bandit for static security analysis
- Safety for dependency vulnerability scanning

These tools are currently commented out in `.github/workflows/test.yml` and will be enabled once the security scanning infrastructure is fully configured.