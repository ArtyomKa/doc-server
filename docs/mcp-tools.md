# MCP Tools Reference

Doc Server provides a Model Context Protocol (MCP) server with tools for documentation search and management.

## Available Tools

### search_docs

Search through ingested documentation using hybrid search (semantic + keyword).

```json
{
  "name": "search_docs",
  "arguments": {
    "query": "pandas read_csv",
    "library_id": "/pandas",
    "limit": 10
  }
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query string |
| `library_id` | string | Yes | - | Library identifier to search within |
| `limit` | integer | No | 10 | Maximum number of results (1-100) |

**Response:**

```json
[
  {
    "content": "pandas.read_csv(filepath_or_buffer, ...)",
    "file_path": "pandas/io/parsers.py",
    "library_id": "/pandas",
    "relevance_score": 0.85,
    "line_numbers": [234, 245]
  }
]
```

**Examples:**

```json
{
  "name": "search_docs",
  "arguments": {
    "query": "fastapi routing",
    "library_id": "/fastapi",
    "limit": 5
  }
}
```

---

### list_libraries

List all available libraries that have been ingested.

```json
{
  "name": "list_libraries",
  "arguments": {}
}
```

**Response:**

```json
[
  {
    "library_id": "/pandas",
    "collection_name": "lib_pandas",
    "document_count": 150,
    "embedding_model": "all-MiniLM-L6-v2",
    "created_at": 1704067200.0
  }
]
```

---

### remove_library

Remove a library from the index and delete all its documents.

```json
{
  "name": "remove_library",
  "arguments": {
    "library_id": "/pandas"
  }
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `library_id` | string | Yes | Library identifier to remove |

**Response:**

```json
true
```

---

### ingest_library

Ingest documentation from a git repository, ZIP archive, or local folder.

```json
{
  "name": "ingest_library",
  "arguments": {
    "source": "https://github.com/pandas-dev/pandas",
    "library_id": "/pandas"
  }
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source` | string | Yes | Source to ingest from (git URL, ZIP path, or local directory) |
| `library_id` | string | Yes | Library identifier for this ingestion |

**Response:**

```json
{
  "success": true,
  "documents_ingested": 150,
  "library_id": "/pandas"
}
```

---

### health_check

Check the health status of the doc-server.

```json
{
  "name": "health_check",
  "arguments": {}
}
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": 1704067200.0,
  "components": {
    "vector_store": {
      "status": "healthy",
      "collection_count": 3
    }
  }
}
```

---

### validate_server

Validate server dependencies and configuration.

```json
{
  "name": "validate_server",
  "arguments": {}
}
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": 1704067200.0,
  "components": {
    "embedding_model": {
      "status": "healthy",
      "model": "all-MiniLM-L6-v2"
    },
    "vector_store": {
      "status": "healthy",
      "type": "chroma"
    }
  }
}
```

## Running the MCP Server

Start the MCP server using the CLI:

```bash
doc-server serve
```

For SSE transport:

```bash
doc-server serve -t sse -h 0.0.0.0 -p 8080
```

## Connecting Clients

Configure your MCP-compatible AI assistant to connect to the running server. The server accepts connections via stdio or HTTP SSE transport.

## Next Steps

- See the [CLI Reference](cli.md) for command-line usage
- Read the [Architecture](architecture.md) to understand the system design
- Explore [Examples](examples.md) for usage patterns
