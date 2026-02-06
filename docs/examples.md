# Usage Examples

Practical examples for common Doc Server workflows.

## Ingesting Documentation

### Ingest from GitHub

Clone and index documentation from a GitHub repository:

```bash
# Ingest pandas documentation
doc-server ingest -s https://github.com/pandas-dev/pandas -l /pandas

# Ingest FastAPI documentation
doc-server ingest -s https://github.com/fastapi/fastapi -l /fastapi

# Ingest Python documentation
doc-server ingest -s https://github.com/python/cpython -l /python
```

### Ingest from Local ZIP Archive

```bash
# Create ZIP of documentation
zip -r docs.zip ./my-docs/

# Ingest the ZIP file
doc-server ingest -s ./docs.zip -l /my-docs
```

### Ingest from Local Directory

```bash
# Ingest local documentation
doc-server ingest -s ./project-docs -l /my-project

# Ingest with custom batch size for large directories
doc-server ingest -s ./large-docs -l /large-project -b 64
```

## Searching Documentation

### Basic Search

```bash
# Search pandas for read_csv
doc-server search -q "pandas read_csv" -l /pandas

# Search FastAPI for routing
doc-server search -q "fastapi routing" -l /fastapi
```

### Advanced Search with Filters

```bash
# Get 20 results in JSON format
doc-server search -q "python async" -l /python -n 20 -f json

# Simple output for scripting
doc-server search -q "configuration" -l /fastapi -f simple
```

### Combine Searches

Search across multiple libraries by running multiple queries:

```bash
# Search in multiple libraries
doc-server search -q "authentication" -l /fastapi -f json > fastapi-auth.json
doc-server search -q "authentication" -l /flask -f json > flask-auth.json
```

## Managing Libraries

### List All Ingested Libraries

```bash
# Table format (default)
doc-server list

# JSON format for scripting
doc-server list -f json
```

### Check Library Statistics

```bash
doc-server list -f json | jq '.[] | {library_id, document_count}'
```

### Remove Unused Libraries

```bash
# Remove with confirmation
doc-server remove -l /unused-docs

# Remove without confirmation
doc-server remove -l /unused-docs --force
```

## MCP Server Integration

### Start MCP Server

```bash
# stdio transport (default)
doc-server serve

# SSE transport on specific port
doc-server serve -t sse -h 0.0.0.0 -p 8080
```

### MCP Client Connection

Connect your MCP-compatible AI assistant to the running server:

```bash
# Start server in background
doc-server serve -t sse -p 8080 &
```

Then configure your client to connect to `http://localhost:8080`.

## Health Checks

### Check Server Health

```bash
doc-server health
```

Sample output:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "vector_store": {
      "status": "healthy",
      "collection_count": 3
    }
  }
}
```

## Batch Ingestion Script

Create a script to ingest multiple libraries:

```bash
#!/bin/bash
# ingest-libraries.sh

LIBRARIES=(
  "https://github.com/pandas-dev/pandas,/pandas"
  "https://github.com/fastapi/fastapi,/fastapi"
  "https://github.com/python/cpython,/python"
)

for lib in "${LIBRARIES[@]}"; do
  IFS=',' read -r url id <<< "$lib"
  echo "Ingesting $id..."
  doc-server ingest -s "$url" -l "$id"
  echo ""
done

echo "All libraries ingested."
doc-server list
```

Run the script:

```bash
chmod +x ingest-libraries.sh
./ingest-libraries.sh
```

## Search Script for Documentation Lookup

```bash
#!/bin/bash
# search-docs.sh

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <query> <library>"
  exit 1
fi

doc-server search -q "$1" -l "$2" -f json | \
  jq -r '.[] | "\(.file_path):\(.line_numbers // [0,0] | .[0])-\(.line_numbers // [0,0] | .[1])\n\(.content[:200])..."'
```

Usage:

```bash
./search-docs.sh "read_csv" /pandas
```

## Configuration Examples

### Development Configuration

```yaml
# ~/.doc-server/config.yaml
log_level: "DEBUG"
mcp_debug: true
keyword_boost: 3.0
embedding_batch_size: 16
```

### Production Configuration

```yaml
# ~/.doc-server/config.yaml
log_level: "WARNING"
storage_path: "/var/lib/doc-server"
embedding_batch_size: 64
default_top_k: 20
```

### GPU Configuration

```yaml
# ~/.doc-server/config.yaml
embedding_device: "cuda"
embedding_batch_size: 32
```

## Next Steps

- Read the [CLI Reference](cli.md) for complete command documentation
- See [MCP Tools](mcp-tools.md) for AI assistant integration
- Configure Doc Server using the [Configuration](configuration.md) guide
- Understand the system in [Architecture](architecture.md)
