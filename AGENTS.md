# AGENTS.md

Development guide for agentic coding agents working on the doc-server repository.

## Project Overview

Doc Server is an AI-powered documentation management system with intelligent search capabilities. It provides MCP (Model Context Protocol) server implementation and REST API for ingesting, processing, and searching technical documentation.

**Technology Stack**: Python 3.10+, FastAPI, MCP, ChromaDB/FAISS, OpenAI embeddings, Pydantic

**Current Implementation State (Phase 2.4)**:
- ✅ Configuration management with Pydantic settings and YAML support
- ✅ Git repository cloning with shallow clone and metadata extraction
- ✅ ZIP archive extraction with security measures
- ✅ File filtering with .gitignore parsing and allowlist enforcement
- ⏳ Document processor (current phase - in progress)
- ⏳ MCP server implementation (skeleton only)
- ⏳ Search layer (embeddings, vector store, hybrid search - skeleton)

---

## Quick Reference

**Development Workflow**: See [WORKFLOW.md](WORKFLOW.md) for setup, testing, and build commands

**Code Style Guidelines**: See [CODING_STANDARDS.md](CODING_STANDARDS.md) for formatting, types, and testing patterns

---

## Repository Structure

```
doc_server/
├── config.py              # ✅ Pydantic settings with YAML support
├── mcp_server.py          # ⏳ MCP server entry point (skeleton)
├── utils.py               # ⏳ Utility functions (skeleton)
├── ingestion/
│   ├── git_cloner.py      # ✅ Git repository cloning & metadata
│   ├── document_processor.py  # ⏳ Document content extraction (skeleton)
│   ├── file_filter.py     # ✅ File filtering with .gitignore parsing
│   └── zip_extractor.py   # ✅ ZIP archive extraction
└── search/
    ├── embedding_service.py   # ⏳ Embedding generation (skeleton)
    ├── vector_store.py    # ⏳ Vector storage management (skeleton)
    └── hybrid_search.py   # ⏳ Hybrid search implementation (skeleton)

tests/
├── test_config.py         # ✅ Full coverage
├── test_git_cloner.py     # ✅ Full coverage
├── test_zip_extractor.py  # ✅ Full coverage
└── test_file_filter.py    # ✅ Full coverage
```

---

## Tools Configuration

- **Black**: 88-char line length, Python 3.10+ target
- **isort**: Black profile, `doc_server` as first-party
- **MyPy**: Strict type checking, no untyped definitions
- **Ruff**: E, W, F, I, B, C4, UP rules (ignores E501, B008)
- **pytest**: Auto asyncio mode, verbose output, short tracebacks