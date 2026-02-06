# Quick Start Guide

Get up and running with Doc Server in under 5 minutes.

## Installation (1 minute)

```bash
pip install doc-server
```

For development installation with all dependencies:

```bash
git clone https://github.com/ArtyomKa/doc-server.git
cd doc-server
pip install -e ".[dev]"
```

## Initial Setup (1 minute)

Create the configuration directory:

```bash
mkdir -p ~/.doc-server
cp config.yaml.example ~/.doc-server/config.yaml
```

The default configuration works out of the box. See [Configuration](configuration.md) for customization options.

## Ingest Documentation (2 minutes)

Clone and index documentation from a Git repository:

```bash
doc-server ingest -s https://github.com/pandas-dev/pandas -l /pandas
```

Or from a ZIP archive:

```bash
doc-server ingest -s ./docs.zip -l /my-docs
```

Or from a local directory:

```bash
doc-server ingest -s ./local-docs -l /project-docs
```

## Search Documentation (1 minute)

Search across your ingested libraries:

```bash
doc-server search -q "pandas read_csv" -l /pandas
doc-server search -q "fastapi routing" -l /fastapi -n 5 -f json
```

List available libraries:

```bash
doc-server list
```

## Start MCP Server

For integration with MCP-compatible AI assistants:

```bash
doc-server serve
```

With SSE transport on a specific port:

```bash
doc-server serve -t sse -h 0.0.0.0 -p 8080
```

## Next Steps

- Read the [CLI Reference](cli.md) for all available commands
- See [Examples](examples.md) for advanced usage patterns
- Configure Doc Server using the [Configuration](configuration.md) guide
- Learn about [MCP Tools](mcp-tools.md) for AI assistant integration
