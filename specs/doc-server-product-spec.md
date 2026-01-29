# Draft: doc-server Product Specification

## Requirements (confirmed)
- **Product Name**: doc-server
- **Purpose**: Local MCP server that serves library/framework documentation (like Context7, but local-first)
- **API Pattern**: Single tool (`search_docs`) - simpler than Context7's two-tool approach
- **Primary Use Case**: General purpose - any documentation (frameworks, SDKs, internal docs)
- **Document Type**: Technical specification with architecture details

## Technical Decisions
- **Embedding Strategy**: Local-only (all-MiniLM-L6-v2 via sentence-transformers) - no API keys needed
- **Library ID Format**: Path-based (e.g., `/pandas`, `/pandas/v2.2`) - familiar Context7-style
- **Search Strategy**: Hybrid (vector + keyword boost for exact terms)
- **MCP Transport**: STDIO (standard input/output) - works with all MCP clients
- **Result Metadata**: Standard (content, file path, relevance score, line numbers)
- **Large File Handling**: Skip files >1MB entirely

## Research Findings
- **Context7 architecture**: Cloud-hosted, two tools (resolve-library-id, query-docs)
- **Context7 limitations**: Rate limits, outdated docs, large libraries can't auto-refresh
- **Best practice**: Hybrid search with keyword boost for exact code terms
- **ChromaDB**: Good for prototyping, LanceDB also viable for local-first
- **FastMCP**: Pythonic way to build MCP servers with `@mcp.tool` decorators

## Scope Boundaries
- **INCLUDE**: 
  - Git clone ingestion (shallow clone)
  - ZIP upload
  - Local folder mapping
  - .gitignore parsing
  - Allowlist filtering (md, rst, txt, py, pyi, cpp, h, hpp, c)
  - ChromaDB vector storage
  - MCP tool exposure
- **EXCLUDE**:
  - Web crawling
  - Cloud hosting
  - User authentication
  - Multi-tenant support
  - API key management
