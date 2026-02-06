# CLI Reference

Doc Server provides a comprehensive command-line interface for all operations.

## Global Options

| Option | Description |
|--------|-------------|
| `--version` | Show version and exit |
| `--verbose, -v` | Enable verbose output |
| `--help` | Show help message |

## Commands

### ingest

Ingest documentation from a git repository, ZIP archive, or local directory.

```bash
doc-server ingest -s <source> -l <library-id>
```

**Options:**

| Option | Description |
|--------|-------------|
| `-s, --source` (required) | Source to ingest from (git URL, ZIP path, or local directory) |
| `-l, --library-id` (required) | Library identifier (e.g., `/pandas`, `/fastapi`) |
| `-b, --batch-size` | Batch size for document processing |

**Examples:**

```bash
# Ingest from GitHub
doc-server ingest -s https://github.com/pandas-dev/pandas -l /pandas

# Ingest from ZIP archive
doc-server ingest -s ./docs.zip -l /my-docs

# Ingest from local directory
doc-server ingest -s ./local-docs -l /project-docs

# Ingest with custom batch size
doc-server ingest -s https://github.com/fastapi/fastapi -l /fastapi -b 64
```

### search

Search through ingested documentation.

```bash
doc-server search -q <query> -l <library-id>
```

**Options:**

| Option | Description |
|--------|-------------|
| `-q, --query` (required) | Search query string |
| `-l, --library-id` (required) | Library identifier to search within |
| `-n, --num-results` | Number of results (default: 10, max: 100) |
| `-f, --format` | Output format (`table`, `json`, `simple`) |

**Examples:**

```bash
# Basic search
doc-server search -q "pandas read_csv" -l /pandas

# Search with 5 results in JSON format
doc-server search -q "fastapi routing" -l /fastapi -n 5 -f json

# Search with simple output
doc-server search -q "data structures" -l /algorithms -f simple
```

### list

List all available libraries.

```bash
doc-server list
```

**Options:**

| Option | Description |
|--------|-------------|
| `-f, --format` | Output format (`table`, `json`, `simple`) |

**Examples:**

```bash
# List all libraries in table format
doc-server list

# List in simple format
doc-server list -f simple

# List in JSON format
doc-server list -f json
```

### remove

Remove a library from the index.

```bash
doc-server remove -l <library-id>
```

**Options:**

| Option | Description |
|--------|-------------|
| `-l, --library-id` (required) | Library identifier to remove |
| `-f, --force` | Skip confirmation prompt |

**Examples:**

```bash
# Remove with confirmation
doc-server remove -l /pandas

# Remove without confirmation
doc-server remove -l /pandas --force
```

### serve

Start the MCP server.

```bash
doc-server serve
```

**Options:**

| Option | Description |
|--------|-------------|
| `-t, --transport` | Transport type (`stdio` or `sse`, default: `stdio`) |
| `-h, --host` | Host for SSE transport (default: `127.0.0.1`) |
| `-p, --port` | Port for SSE transport (default: `8080`) |

**Examples:**

```bash
# Start with stdio transport (default)
doc-server serve

# Start with SSE transport
doc-server serve -t sse

# Start with SSE on specific host and port
doc-server serve -t sse -h 0.0.0.0 -p 8080
```

### health

Check the health status of the server.

```bash
doc-server health
```

**Examples:**

```bash
doc-server health
```

## Next Steps

- Read the [Quick Start Guide](quickstart.md) for a getting started tutorial
- See [Examples](examples.md) for advanced usage patterns
- Learn about [MCP Tools](mcp-tools.md) for AI assistant integration
