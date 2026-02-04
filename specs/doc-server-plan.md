# doc-server Implementation Plan

## Project Overview
Local MCP server for serving library/framework documentation with local-first approach, single search tool, and hybrid vector+keyword search.

## Architecture Summary
- **Language**: Python 3.10+
- **MCP Framework**: FastMCP
- **Embeddings**: all-MiniLM-L6-v2 (sentence-transformers)
- **Vector DB**: ChromaDB
- **Transport**: STDIO
- **API**: Single tool `search_docs(query, library_id)`

## Implementation Phases

### Phase 1: Core Infrastructure Setup
**Priority**: High
**Timeline**: 2-3 days

#### 1.1 Project Structure
```
doc-server/
├── doc_server/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── git_cloner.py
│   │   ├── zip_extractor.py
│   │   ├── file_filter.py
│   │   └── document_processor.py
│   ├── search/
│   │   ├── __init__.py
│   │   ├── embedding_service.py
│   │   ├── vector_store.py
│   │   └── hybrid_search.py
│   ├── mcp_server.py
│   ├── config.py
│   └── utils.py
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

#### 1.2 Dependencies
```python
# Core MCP
fastmcp>=0.4.0

# ML/Embeddings
sentence-transformers>=2.2.2
chromadb>=0.4.15

# Git/File handling
GitPython>=3.1.37
pathspec>=0.11.1

# Utilities
click>=8.0.0
pydantic>=2.0.0
```

#### 1.3 Configuration System
- Settings for model paths, storage locations, file filters
- Environment-based configuration
- Default storage in `~/.doc-server/`

### Phase 2: Ingestion System
**Priority**: High
**Timeline**: 3-4 days

#### 2.1 Git Cloner (`ingestion/git_cloner.py`)
- Shallow clone (`git clone --depth 1`)
- Repository metadata extraction
- Error handling for invalid URLs
- Cleanup utilities

#### 2.2 ZIP Extractor (`ingestion/zip_extractor.py`)
- Secure ZIP extraction
- Path traversal protection
- Archive metadata extraction

#### 2.3 File Filter (`ingestion/file_filter.py`)
- `.gitignore` parsing using `pathspec`
- Allowlist enforcement: `.md`, `.rst`, `.txt`, `.py`, `.pyi`, `.cpp`, `.h`, `.hpp`, `.c`
- File size guard (>1MB skip)
- Binary detection (null byte check)

#### 2.4 Document Processor (`ingestion/document_processor.py`)
- Metadata header formatting
- Content chunking for large files
- File path preservation
- Line number tracking

### Phase 3: Search Infrastructure
**Priority**: High
**Timeline**: 3-4 days

#### 3.1 Embedding Service (`search/embedding_service.py`)
- all-MiniLM-L6-v2 model loading
- Batch embedding processing
- Caching for repeated embeddings
- Model warmup on startup

#### 3.2 Vector Store (`search/vector_store.py`)
- ChromaDB integration
- Collection management (named by library ID)
- Persistent storage configuration
- Index management utilities

#### 3.3 Hybrid Search (`search/hybrid_search.py`)
- Vector similarity search
- Keyword boost for exact terms
- Result ranking algorithm
- Metadata enrichment (relevance scores, line numbers)

### Phase 4: MCP Server Implementation
**Priority**: High
**Timeline**: 2-3 days

#### 4.1 Core Server (`mcp_server.py`)
```python
@mcp.tool()
def search_docs(query: str, library_id: str) -> List[DocumentResult]:
    """Search across ingested documentation"""
    # Implementation
```

#### 4.2 Library Management Tools
```python
@mcp.tool()
def ingest_library(source: str, library_id: str) -> bool:
    """Ingest documentation from git/zip/local folder"""

@mcp.tool() 
def list_libraries() -> List[LibraryInfo]:
    """List all available libraries"""

@mcp.tool()
def remove_library(library_id: str) -> bool:
    """Remove library from index"""
```

#### 4.3 Server Configuration
- STDIO transport setup
- Error handling and logging
- Graceful shutdown handling

### Phase 8: CLI Interface
**Priority**: Low (after Phase 7 Documentation)
**Timeline**: 1-2 days

#### 8.1 Command Line Interface
```bash
doc-server ingest --source <url|path|zip> --library-id <id>
doc-server search --query <text> --library-id <id>
doc-server list
doc-server remove --library-id <id>
doc-server serve
```

#### 8.2 Progress Indicators
- Ingestion progress bars
- Search result formatting
- Error messages and validation

### Phase 6: Testing & Validation
**Priority**: Medium
**Timeline**: 2-3 days

#### 6.1 Unit Tests
- Ingestion components
- Search functionality
- MCP tool validation
- File filtering logic

#### 6.2 Integration Tests
- End-to-end ingestion flow
- Search accuracy validation
- Performance benchmarks
- Error scenario testing

#### 6.3 Test Documentation
- Sample repositories for testing
- Performance benchmarks
- Search quality metrics

### Phase 6.1.1: ChromaDB Compatibility & Search Implementation
**Priority**: High
**Timeline**: 1-2 days

#### 6.2.1 ChromaDB API Compatibility
- Fix ChromaEmbeddingFunction class to implement required ChromaDB interface
- Resolve type errors in vector_store.py for metadata handling
- Ensure ChromaDB collection operations work with current API version
- Add proper error handling for ChromaDB operations

#### 6.2.2 Search Implementation & Testing
- Complete algorithms library ingestion (currently 0 documents due to ChromaDB issues)
- Verify hybrid search functionality works with ingested content
- Test search queries: "binary search", "sorting algorithms", "data structures"
- Validate search result formatting and relevance scores

#### 6.2.3 Integration Validation
- Test end-to-end ingestion workflow with algorithms library
- Verify document persistence across server restarts
- Validate search performance with ingested algorithms documentation
- Test MCP server search tool functionality

### Phase 7: Documentation & Examples
**Priority**: Low
**Timeline**: 1-2 days

#### 7.1 User Documentation
- Installation guide
- Usage examples
- Configuration options
- Troubleshooting guide

#### 7.2 Developer Documentation
- API reference
- Architecture overview
- Contributing guidelines
- Extension points

## Technical Specifications

### Library ID Format
- Path-based: `/pandas`, `/pandas/v2.2`
- Version support in path segments
- Case-insensitive handling

### Search Response Format
```python
class DocumentResult(BaseModel):
    content: str
    file_path: str
    library_id: str
    relevance_score: float
    line_numbers: Optional[Tuple[int, int]]
```

### Storage Structure
```
~/.doc-server/
├── chroma.db/          # ChromaDB database
├── models/             # Downloaded embedding models
├── libraries/          # Ingested source data
└── config.yaml         # User configuration
```

### Performance Targets
- **Ingestion**: 1000 files/minute
- **Search**: <500ms response time
- **Storage**: ~1GB per 1000 files (including embeddings)
- **Memory**: <2GB runtime usage

## Risk Mitigation

### Technical Risks
1. **Embedding Model Size**: Implement lazy loading and caching
2. **Large Repository Handling**: Strict size limits and progress indicators
3. **Memory Usage**: Batch processing and streaming for large files

### User Experience Risks
1. **Complex Setup**: Provide one-command installation script
2. **Slow Search**: Implement result caching and query optimization
3. **Version Conflicts**: Clear library ID versioning scheme

## Success Metrics
- Successful ingestion of popular libraries (pandas, numpy, fastapi)
- Search relevance >80% for documentation queries
- Sub-second search response times
- Zero dependency on external APIs during operation

## Next Steps
1. Set up project structure and dependencies
2. Implement ingestion pipeline with file filtering
3. Integrate embedding service and vector storage
4. Build MCP server with search functionality
5. Add CLI interface and testing
6. Documentation and examples

This plan prioritizes a working MVP with core functionality, then expands with CLI tools, testing, and documentation based on the product specification requirements.