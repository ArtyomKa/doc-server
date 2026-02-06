# Doc Server

[![CI](https://github.com/ArtyomKa/doc-server/actions/workflows/test.yml/badge.svg)](https://github.com/ArtyomKa/doc-server/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-85%25%20minimum-green)](https://github.com/ArtyomKa/doc-server/actions/workflows/test.yml)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Types: MyPy](https://img.shields.io/badge/types-MyPy-blue.svg)](https://mypy.readthedocs.io)
[![Linting: Ruff](https://img.shields.io/badge/linting-Ruff-blue.svg)](https://docs.astral.sh/ruff)

AI-powered documentation management system with intelligent search capabilities.

## Quick Start

```bash
pip install doc-server
doc-server ingest -s https://github.com/pandas-dev/pandas -l /pandas
doc-server search -q "pandas read_csv" -l /pandas
doc-server serve
```

## Features

- **Hybrid Search**: Combines semantic similarity with keyword matching
- **Multi-Source Ingestion**: Clone Git repositories, extract ZIP archives, or use local directories
- **MCP Integration**: Full Model Context Protocol server implementation
- **CLI Interface**: Comprehensive command-line tool for all operations

## Documentation

See the [docs/](docs/) directory for complete documentation:

- [Quick Start Guide](docs/quickstart.md) - Get running in 5 minutes
- [Installation](docs/installation.md) - Detailed installation instructions
- [CLI Reference](docs/cli.md) - All CLI commands and options
- [MCP Tools](docs/mcp-tools.md) - Available MCP tools and usage
- [Configuration](docs/configuration.md) - Configuration options
- [Architecture](docs/architecture.md) - System design and data flows
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions
- [Examples](docs/examples.md) - Usage examples and tutorials

## Requirements

- Python 3.10+
- See `pyproject.toml` for full dependency list

## License

MIT License - see LICENSE file for details.
