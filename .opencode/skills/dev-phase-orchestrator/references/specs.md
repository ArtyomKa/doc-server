# Doc-Server Specification Quick Reference

## Product Spec Key Points
- **Purpose**: Local MCP server for documentation
- **Transport**: STDIO only (no CLI)
- **Embeddings**: all-MiniLM-L6-v2 (local)
- **File Size Limit**: >1MB files skipped
- **Supported Extensions**: .md, .rst, .txt, .py, .pyi, .cpp, .h, .hpp, .c

## Task Tracker Structure
- Phase 1: Infrastructure ✅
- Phase 2: Ingestion (2.1✅, 2.2✅, 2.3⏳, 2.4⏳)
- Phase 3: Search (⏳)
- Phase 4: MCP Server (⏳)
- Phase 5: Documentation (⏳)

## Acceptance Criteria Pattern
- AC-X.Y.Z format
- Each has test specifications
- Target >90% coverage
- Security requirements for ZIP, file handling

## Configuration Settings
- `max_file_size: 1048576` (1MB)
- `allowed_extensions: [...]` (see above)
- `storage_path: ~/.doc-server`

## Common Issues Found
1. **Missing pathspec dependency** - Should be in requirements
2. **Wrong entry points** - Only MCP server, no CLI
3. **Security gaps** - Path traversal, binary detection
4. **Testing gaps** - Missing edge cases, error handling