# Troubleshooting Guide

Common issues and solutions for Doc Server.

## Installation Issues

### pip Installation Fails

Ensure you have the latest pip:

```bash
pip install --upgrade pip
```

Then retry:

```bash
pip install doc-server
```

### Python Version Error

Doc Server requires Python 3.10 or higher:

```bash
python --version
```

If you have an older version, install Python 3.10+ from python.org or your package manager.

### torch Installation Issues

If torch-related errors occur during installation:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install doc-server
```

Or use CPU-only PyTorch:

```bash
pip install torch torchvision torchaudio
pip install doc-server
```

## Git Clone Failures

### Authentication Errors

For private repositories, use SSH URLs:

```bash
doc-server ingest -s git@github.com:username/private-repo.git -l /my-repo
```

Ensure your SSH key is added to the SSH agent:

```bash
ssh-add ~/.ssh/id_rsa
```

### Network Timeouts

For slow connections, increase timeout or use shallow clone:

```bash
GIT_TERMINAL_PROMPT=0 doc-server ingest -s <url> -l /repo
```

### Large Repository Clones

For large repositories, the initial clone may take time. Monitor progress in the terminal output.

## Search Issues

### No Results Found

1. Verify the library was ingested:

   ```bash
   doc-server list
   ```

2. Check spelling of library_id:

   ```bash
   doc-server search -q "query" -l /pandas
   ```

3. Try broader search terms:

   ```bash
   doc-server search -q "data" -l /pandas
   ```

### Poor Search Relevance

Try adjusting the keyword boost in configuration:

```yaml
keyword_boost: 3.0  # Increase for more keyword matching
```

Or increase the number of results and manually filter:

```bash
doc-server search -q "query" -l /library -n 20
```

### Empty Results After Ingestion

1. Check if files were filtered out:

   ```bash
   # Ingestion output shows filtered count
   doc-server ingest -s <source> -l /library
   ```

2. Verify allowed extensions in configuration include your file types.

3. Check for .gitignore filtering:

   ```bash
   # Large .gitignore files may filter many files
   ls -la .gitignore 2>/dev/null || echo "No .gitignore"
   ```

## MCP Server Issues

### Connection Refused

Ensure the MCP server is running:

```bash
doc-server serve -t sse -h 0.0.0.0 -p 8080
```

Check the server is listening:

```bash
curl http://localhost:8080/health
```

### stdio Transport Not Working

Some clients may have issues with stdio. Try SSE transport:

```bash
doc-server serve -t sse
```

### MCP Tools Not Available

Verify tools are registered:

```bash
doc-server health
```

Check MCP server logs for errors.

## Performance Issues

### Slow Ingestion

1. Reduce batch size:

   ```bash
   doc-server ingest -s <url> -l /id -b 16
   ```

2. Use CPU device if GPU memory is limited:

   ```yaml
   embedding_device: "cpu"
   ```

3. For large repositories, consider ingesting in batches.

### High Memory Usage

1. Reduce embedding batch size:

   ```yaml
   embedding_batch_size: 16
   ```

2. Limit file size:

   ```yaml
   max_file_size: 524288  # 512KB
   ```

3. Remove unused libraries:

   ```bash
   doc-server remove -l /unused-library
   ```

## Configuration Issues

### Settings Not Applied

1. Verify configuration file path:

   ```bash
   ls -la ~/.doc-server/config.yaml
   ```

2. Check configuration syntax (YAML):

   ```bash
   python -c "import yaml; yaml.safe_load(open('~/.doc-server/config.yaml'))"
   ```

3. Environment variables take precedence:

   ```bash
   echo $DOC_SERVER_LOG_LEVEL
   ```

### Configuration File Not Found

Create the default configuration:

```bash
mkdir -p ~/.doc-server
cp config.yaml.example ~/.doc-server/config.yaml
```

## Logging and Debugging

### Enable Debug Logging

```bash
DOC_SERVER_LOG_LEVEL=DEBUG doc-server ingest -s <url> -l /id
```

Or in configuration:

```yaml
log_level: "DEBUG"
```

### Verbose CLI Output

Use the verbose flag:

```bash
doc-server ingest -s <url> -l /id -v
```

## Getting Help

If your issue is not listed here:

1. Check the logs for detailed error messages
2. Run with debug logging enabled
3. Report issues at: https://github.com/ArtyomKa/doc-server/issues

## Next Steps

- Read the [Configuration](configuration.md) guide for tuning
- See [Examples](examples.md) for usage patterns
- Learn about [MCP Tools](mcp-tools.md)
