# Configuration Guide

Doc Server can be configured through YAML configuration files, environment variables, or command-line options.

## Configuration Sources

Configuration is loaded in the following priority order (highest to lowest):

1. Command-line options
2. `~/.doc-server/config.yaml` (YAML configuration file)
3. Environment variables (`DOC_SERVER_*` prefix)
4. `.env` file in the current directory
5. Default values

## Configuration File

Create or edit `~/.doc-server/config.yaml`:

```yaml
# Storage paths
storage_path: "~/.doc-server"

# Embedding configuration
embedding_model: "all-MiniLM-L6-v2"
embedding_device: "cpu"
embedding_batch_size: 32

# Search behavior
default_top_k: 10
search_min_score: 0.5
keyword_boost: 2.0

# File processing limits
max_file_size: 1048576

# Allowed file extensions
allowed_extensions:
  - ".py"
  - ".pyi"
  - ".c"
  - ".cpp"
  - ".h"
  - ".hpp"
  - ".md"
  - ".rst"
  - ".txt"
  - ".json"
  - ".yaml"
  - ".yml"
  - ".toml"
  - ".cfg"
  - ".ini"
  - ".conf"

# Logging configuration
log_level: "INFO"
log_format: "json"

# MCP server settings
mcp_transport: "stdio"
mcp_debug: false

# Remote Backend settings
mode: "local"  # or "remote"
backend_url: "http://localhost:8000"
backend_api_key: ""
backend_timeout: 30
backend_verify_ssl: true
```

## Environment Variables

Set configuration via environment variables:

```bash
export DOC_SERVER_STORAGE_PATH="~/.doc-server"
export DOC_SERVER_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export DOC_SERVER_EMBEDDING_DEVICE="cpu"
export DOC_SERVER_DEFAULT_TOP_K=10
export DOC_SERVER_LOG_LEVEL="INFO"
export DOC_SERVER_MCP_TRANSPORT="stdio"
export DOC_SERVER_MODE="remote"
export DOC_SERVER_BACKEND_URL="http://localhost:8000"
export DOC_SERVER_BACKEND_API_KEY="your-secret-key"
```

## Configuration Options

### Storage Paths

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `storage_path` | `DOC_SERVER_STORAGE_PATH` | `~/.doc-server` | Base directory for all data |
| `chroma_db_path` | `DOC_SERVER_CHROMA_DB_PATH` | `~/.doc-server/chroma.db` | ChromaDB database location |
| `models_path` | `DOC_SERVER_MODELS_PATH` | `~/.doc-server/models` | Downloaded embedding models |
| `libraries_path` | `DOC_SERVER_LIBRARIES_PATH` | `~/.doc-server/libraries` | Ingested source data |

### Embedding Settings

| Option | Environment Variable | Default | Description |
|--------|----------------------|---------|-------------|
| `embedding_model` | `DOC_SERVER_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model name |
| `embedding_device` | `DOC_SERVER_EMBEDDING_DEVICE` | `cpu` | Device for embeddings (`cpu` or `cuda`) |
| `embedding_batch_size` | `DOC_SERVER_EMBEDDING_BATCH_SIZE` | `32` | Batch size for embedding generation |

### Search Settings

| Option | Environment Variable | Default | Description |
|--------|----------------------|---------|-------------|
| `default_top_k` | `DOC_SERVER_DEFAULT_TOP_K` | `10` | Default number of search results |
| `search_min_score` | `DOC_SERVER_SEARCH_MIN_SCORE` | `0.5` | Minimum relevance score threshold |
| `keyword_boost` | `DOC_SERVER_KEYWORD_BOOST` | `2.0` | Weight multiplier for keyword matches |

### File Processing

| Option | Environment Variable | Default | Description |
|--------|----------------------|---------|-------------|
| `max_file_size` | `DOC_SERVER_MAX_FILE_SIZE` | `1048576` | Maximum file size in bytes |
| `allowed_extensions` | `DOC_SERVER_ALLOWED_EXTENSIONS` | See list above | List of allowed file extensions |

### MCP Settings

| Option | Environment Variable | Default | Description |
|--------|----------------------|---------|-------------|
| `mcp_transport` | `DOC_SERVER_MCP_TRANSPORT` | `stdio` | Transport type (`stdio` or `sse`) |
| `mcp_debug` | `DOC_SERVER_MCP_DEBUG` | `false` | Enable debug mode for MCP |

### Remote Backend Settings

| Option | Environment Variable | Default | Description |
|--------|----------------------|---------|-------------|
| `mode` | `DOC_SERVER_MODE` | `local` | Operation mode (`local` or `remote`) |
| `backend_url` | `DOC_SERVER_BACKEND_URL` | `http://localhost:8000` | URL of the remote backend server |
| `backend_api_key` | `DOC_SERVER_BACKEND_API_KEY` | `""` | API Key for backend authentication |
| `backend_timeout` | `DOC_SERVER_BACKEND_TIMEOUT` | `30` | Request timeout in seconds |
| `backend_verify_ssl` | `DOC_SERVER_BACKEND_VERIFY_SSL` | `true` | Verify SSL certificates |

### Logging Settings

| Option | Environment Variable | Default | Description |
|--------|----------------------|---------|-------------|
| `log_level` | `DOC_SERVER_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `log_format` | `DOC_SERVER_LOG_FORMAT` | `json` | Log format (`json` or `console`) |

## Development Configuration

For development, use debug logging and smaller batches:

```yaml
log_level: "DEBUG"
mcp_debug: true
keyword_boost: 3.0
embedding_batch_size: 16
```

## Production Configuration

For production deployments:

```yaml
log_level: "WARNING"
storage_path: "/var/lib/doc-server"
embedding_batch_size: 64
default_top_k: 20
```

## Next Steps

- See the [CLI Reference](cli.md) for command-line usage
- Learn about [MCP Tools](mcp-tools.md) for AI integration
- Read the [Examples](examples.md) for usage patterns
