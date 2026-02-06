# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 9.1: Release infrastructure
  - Manual version bumping process documentation
  - Hatchling build configuration with automatic version sync
  - Pre-release version support (alpha, beta, rc)
  - MANIFEST.in for source distribution exclusions
  - GitHub Actions release workflow
  - Automated changelog generation
  - Installation testing in release pipeline

## [0.1.0] - 2024-02-06

### Added
- Phase 1: Core infrastructure
  - Project structure with pyproject.toml
  - Configuration system with Pydantic settings
  - FastMCP server setup
  
- Phase 2: Ingestion system
  - Git repository cloning with shallow clone
  - ZIP archive extraction with security measures
  - File filtering with .gitignore parsing
  - Document processor with chunking
  
- Phase 3: Search infrastructure
  - Embedding service with sentence-transformers
  - Vector store with ChromaDB integration
  - Hybrid search with BM25 and vector similarity
  
- Phase 4: MCP server implementation
  - search_docs tool for semantic search
  - ingest_library tool for documentation ingestion
  - list_libraries and remove_library tools
  - Health check and logging
  
- Phase 6: Testing & validation
  - 492 unit tests with comprehensive coverage
  - Integration tests for end-to-end workflows
  - Performance regression tests
  - CI/CD pipeline with GitHub Actions
  
- Phase 7: Documentation
  - User documentation (README, quickstart, troubleshooting)
  - Developer documentation
  
- Phase 8: CLI interface
  - Click-based CLI with 6 commands
  - Progress bars and colored output
  - Multiple output formats (table/json/simple)

[Unreleased]: https://github.com/docserver/doc-server/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/docserver/doc-server/releases/tag/v0.1.0
