# doc-server Task Tracker

## Scope Decisions
- **CLI Interface (Phase 5)**: ❌ OUT OF SCOPE per product spec - only MCP server with STDIO transport required
- **Entry Points**: Single entry point `doc-server = "doc_server.mcp_server:main"` (no CLI entry point)

## Project Status: Planning Complete ✅
- [x] Product specification review
- [x] Implementation plan creation
- [x] Task breakdown and prioritization
- [x] Scope alignment verification

---

## Phase 1: Core Infrastructure Setup (Priority: High)

### 1.1 Project Structure Setup
- [x] Create main project directory structure
- [x] Initialize `pyproject.toml` with project metadata
- [x] Set up `requirements.txt` with core dependencies
- [x] Create basic `__init__.py` files for all modules
- [x] Set up development environment configuration

### 1.2 Dependencies & Environment
- [x] Install FastMCP for MCP server framework
- [x] Add sentence-transformers for embeddings
- [x] Add ChromaDB for vector storage
- [x] Add GitPython for git operations
- [x] Add pathspec for .gitignore parsing
- [x] Add pydantic for data models
- [x] Set up development dependencies (pytest, black, flake8)

### 1.3 Configuration System
- [x] Create `config.py` with settings management
- [x] Implement environment-based configuration loading
- [x] Set up default storage paths (`~/.doc-server/`)
- [x] Create configuration validation
- [x] Add configuration documentation

---

## Phase 2: Ingestion System (Priority: High)

### 2.1 Git Cloner Module
- [x] Implement `git_cloner.py` with shallow clone functionality
- [x] Add repository metadata extraction
- [x] Implement error handling for invalid URLs
- [x] Add cleanup utilities for temporary clones
- [x] Write unit tests for git operations

### 2.2 ZIP Extractor Module
- [x] Implement `zip_extractor.py` with secure extraction
- [x] Add path traversal protection
- [x] Extract archive metadata
- [x] Handle password-protected ZIPs (optional)
- [x] Write unit tests for ZIP operations

### 2.3 File Filter Module
- [x] Implement `file_filter.py` with .gitignore parsing
- [x] Add allowlist enforcement for file types
- [x] Implement file size guard (>1MB)
- [x] Add binary file detection (null bytes)
- [x] Write comprehensive tests for filtering logic

### 2.4 Document Processor Module
- [x] Implement `document_processor.py` with metadata headers
- [x] Add content chunking for large files (>2KB threshold)
- [x] Preserve file paths and line numbers
- [x] Optimize for embedding generation
- [x] Preserve code formatting and special characters
- [x] Handle different file encodings (UTF-8, Latin-1)
- [x] Write tests for document formatting

---

## Phase 3: Search Infrastructure (Priority: High)

### 3.1 Embedding Service
- [ ] Implement `embedding_service.py` with all-MiniLM-L6-v2
- [ ] Add batch embedding processing
- [ ] Implement embedding caching system
- [ ] Add model warmup on startup
- [ ] Write performance tests for embedding generation

### 3.2 Vector Store Module
- [ ] Implement `vector_store.py` with ChromaDB integration
- [ ] Add collection management by library ID
- [ ] Configure persistent storage
- [ ] Implement index management utilities
- [ ] Write tests for storage operations

### 3.3 Hybrid Search Module
- [ ] Implement `hybrid_search.py` with vector similarity
- [ ] Add keyword boost for exact terms
- [ ] Implement result ranking algorithm
- [ ] Add metadata enrichment (scores, line numbers)
- [ ] Write tests for search accuracy

---

## Phase 4: MCP Server Implementation (Priority: High)

### 4.1 Core MCP Server
- [ ] Create `mcp_server.py` with FastMCP setup
- [ ] Implement `search_docs(query, library_id)` tool
- [ ] Add STDIO transport configuration
- [ ] Implement proper error handling and logging
- [ ] Add graceful shutdown handling

### 4.2 Library Management Tools
- [ ] Implement `ingest_library(source, library_id)` tool
- [ ] Implement `list_libraries()` tool
- [ ] Implement `remove_library(library_id)` tool
- [ ] Add input validation and sanitization
- [ ] Write MCP protocol compliance tests

### 4.3 Server Configuration
- [ ] Configure logging for debugging
- [ ] Add startup validation checks
- [ ] Implement health check endpoint
- [ ] Add performance monitoring hooks
- [ ] Test with different MCP clients

---

## Phase 5: CLI Interface (Priority: Medium) ❌ OUT OF SCOPE

### 5.1 Command Line Interface
- [ ] Implement CLI entry point with Click
- [ ] Add `ingest` command with options
- [ ] Add `search` command with query parameters
- [ ] Add `list` command for library enumeration
- [ ] Add `remove` command for library deletion
- [ ] Add `serve` command for starting MCP server

### 5.2 User Experience
- [ ] Add progress bars for long operations
- [ ] Format search results for terminal display
- [ ] Add colored output and formatting
- [ ] Implement helpful error messages
- [ ] Add command validation and help text

**Note**: CLI interface excluded per product spec - only MCP server required.

---

## Phase 6: Testing & Validation (Priority: Medium)

### 6.1 Unit Test Suite
- [ ] Write tests for ingestion components
- [ ] Write tests for search functionality
- [ ] Write tests for MCP tool validation
- [ ] Write tests for file filtering logic
- [ ] Achieve >90% code coverage

### 6.2 Integration Tests
- [ ] Test end-to-end ingestion flow
- [ ] Validate search accuracy on sample docs
- [ ] Test with popular libraries (pandas, fastapi)
- [ ] Performance benchmarking
- [ ] Test error scenarios and edge cases

### 6.3 Test Infrastructure
- [ ] Set up pytest configuration
- [ ] Create test data fixtures
- [ ] Add sample repositories for testing
- [ ] Set up continuous integration
- [ ] Document test running procedures

---

## Phase 7: Documentation & Examples (Priority: Low)

### 7.1 User Documentation
- [ ] Write README with installation guide
- [ ] Create usage examples and tutorials
- [ ] Document configuration options
- [ ] Add troubleshooting guide
- [ ] Create quick start guide

### 7.2 Developer Documentation
- [ ] Write API reference documentation
- [ ] Document architecture and design decisions
- [ ] Create contributing guidelines
- [ ] Document extension points and plugins
- [ ] Add code examples for common use cases

---

## MVP Definition (Critical Path)

**Core MVP includes**: Phases 1-4, minimal tests (6.1)

**Stretch MVP includes**: All phases except advanced documentation and CLI (Phase 5 - OUT OF SCOPE)

**Release-ready**: All phases complete (excluding Phase 5)

---

## Progress Tracking

### Current Status
- **Phase 1**: 12/17 tasks complete (1.1 ✅ complete, 1.2 ✅ complete, 1.3 incomplete)
- **Phase 2**: 15/16 tasks complete (2.1 ✅ complete, 2.2 ✅ complete, 2.3 ✅ complete, 2.4 incomplete)
- **Phase 3**: 0/15 tasks complete
- **Phase 4**: 0/9 tasks complete
- **Phase 5**: OUT OF SCOPE (per product spec)
- **Phase 6**: 0/12 tasks complete
- **Phase 7**: 0/10 tasks complete

**Overall Progress**: 27/69 tasks complete (39%)

---

## Dependencies & Blockers

### External Dependencies
- [ ] Verify FastMCP compatibility with Python 3.10+
- [ ] Confirm sentence-transformers model availability
- [ ] Test ChromaDB performance with expected data sizes

### Technical Blockers
- [ ] None identified yet

---

## Next Immediate Tasks (This Week)

1. **Setup Phase 1** - Create project structure and install dependencies
2. **Start Phase 2** - Implement file filtering and git cloner
3. **Test Ingestion** - Validate file processing pipeline
4. **Begin Phase 3** - Implement embedding service and vector store

---

## Notes & Decisions

- Priority is given to working MVP over comprehensive testing
- CLI interface is secondary to MCP functionality
- Performance optimization should be deferred until after MVP
- Documentation should be written alongside implementation, not after