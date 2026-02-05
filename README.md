# Doc Server

[![CI](https://github.com/ArtyomKa/doc-server/actions/workflows/test.yml/badge.svg)](https://github.com/ArtyomKa/doc-server/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/ArtyomKa/doc-server/branch/master/graph/badge.svg)](https://codecov.io/gh/ArtyomKa/doc-server)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Types: MyPy](https://img.shields.io/badge/types-MyPy-blue.svg)](https://mypy.readthedocs.io)
[![Linting: Ruff](https://img.shields.io/badge/linting-Ruff-blue.svg)](https://docs.astral.sh/ruff)
[![Tests](https://img.shields.io/badge/tests-457-green.svg)]()

AI-powered documentation management system with intelligent search capabilities.

## Overview

Doc Server is a comprehensive solution for ingesting, processing, and searching technical documentation. It provides intelligent search capabilities that combine keyword matching with semantic understanding to deliver accurate, context-aware results.

## Features

- **Multi-source Ingestion**: Clone Git repositories or extract ZIP archives
- **Intelligent File Processing**: Support for Markdown, code files, PDFs, and Word documents
- **Hybrid Search**: Combines keyword search with semantic understanding
- **Vector Storage**: Efficient storage and retrieval of document embeddings
- **MCP Integration**: Full Model Context Protocol (MCP) server implementation
- **REST API**: FastAPI-based HTTP interface for integration with external tools

## Quick Start

### Installation

```bash
uv pip install -e .
```

### Development Setup

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black .
isort .

# Type checking
mypy doc_server
```

## Usage

### MCP Server

```bash
python -m doc_server.mcp_server
```

### HTTP API

```bash
uvicorn doc_server.mcp_server:app --reload
```

## Architecture

- **Ingestion Layer**: Handles Git cloning, ZIP extraction, and document processing
- **Search Layer**: Manages embeddings, vector storage, and hybrid search
- **API Layer**: Provides MCP and HTTP interfaces
- **Configuration**: Centralized settings management

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list

## License

MIT License - see LICENSE file for details.