# Architecture Overview

This document describes the high-level architecture and data flows of Doc Server.

## System Architecture

```mermaid
graph TB
    subgraph User Layer
        CLI[CLI Interface]
        MCP[MCP Client]
    end

    subgraph API Layer
        MCP_Server[MCP Server<br/>FastMCP]
    end

    subgraph Ingestion Layer
        Git[Git Cloner]
        ZIP[ZIP Extractor]
        Processor[Document Processor]
        Filter[File Filter]
    end

    subgraph Search Layer
        Hybrid[Hybrid Search]
        Embeddings[Embedding Service]
        BM25[BM25 Keyword Search]
        RRF[RRF Fusion]
    end

    subgraph Storage Layer
        VectorStore[(Vector Store<br/>ChromaDB)]
        FileSystem[File System<br/>Libraries]
    end

    CLI --> MCP_Server
    MCP --> MCP_Server

    MCP_Server --> Ingestion
    MCP_Server --> Search

    Ingestion --> Git
    Ingestion --> ZIP
    Ingestion --> Processor
    Processor --> Filter
    Filter --> VectorStore

    Search --> Hybrid
    Hybrid --> Embeddings
    Hybrid --> BM25
    Hybrid --> RRF
    Embeddings --> VectorStore
```

## Components

### Ingestion Layer

Responsible for acquiring and processing documentation:

- **Git Cloner**: Shallow clones Git repositories with metadata extraction
- **ZIP Extractor**: Extracts ZIP archives with security validation
- **File Filter**: Filters files using `.gitignore` patterns and allowlists
- **Document Processor**: Extracts content, chunks documents, handles encodings

### Search Layer

Provides hybrid search capabilities:

- **Embedding Service**: Generates semantic embeddings using sentence-transformers
- **BM25 Scorer**: Performs keyword-based scoring using BM25 algorithm
- **RRF Fusion**: Combines vector and keyword results using Reciprocal Rank Fusion

### Storage Layer

Persists data for retrieval:

- **Vector Store**: ChromaDB for storing embeddings and metadata
- **File System**: Stores source documentation for reference

### API Layer

Provides external interfaces:

- **CLI**: Command-line interface using Click
- **MCP Server**: Model Context Protocol server using FastMCP

## Data Flow: Ingestion

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Ingestion
    participant Filter
    participant Processor
    participant VectorStore

    User->>CLI: doc-server ingest -s <url> -l <id>
    CLI->>Ingestion: ingest_library(source, library_id)

    Note over Ingestion: Determine source type
    Ingestion->>Ingestion: Clone Git repo OR<br/>Extract ZIP OR<br/>Use local dir

    Ingestion->>Filter: filter_files(all_files)
    Filter-->>Ingestion: filtered_files

    loop For each file
        Ingestion->>Processor: process_file(file_path, library_id)
        Processor-->>Ingestion: document_chunks
    end

    Ingestion->>VectorStore: create_collection(library_id)
    Ingestion->>VectorStore: add_documents(chunks, metadata)

    VectorStore-->>Ingestion: confirmation
    Ingestion-->>CLI: result
    CLI-->>User: "150 documents ingested"
```

## Data Flow: Search

```mermaid
sequenceDiagram
    participant User
    participant MCP
    participant Hybrid
    participant Embeddings
    participant BM25
    participant VectorStore

    User->>MCP: search_docs(query, library_id)
    MCP->>Hybrid: search(query, library_id, limit)

    par Vector Search
        Hybrid->>Embeddings: embed(query)
        Embeddings-->>Hybrid: query_embedding
        Hybrid->>VectorStore: similarity_search(query_embedding)
        VectorStore-->>Hybrid: vector_results
    end

    par Keyword Search
        Hybrid->>BM25: score(query, library_id)
        BM25-->>Hybrid: keyword_results
    end

    Hybrid->>Hybrid: RRF Fusion<br/>Combine and rerank
    Hybrid-->>MCP: ranked_results
    MCP-->>User: search results
```

## Data Flow: MCP Communication

```mermaid
sequenceDiagram
    participant User
    participant MCP_Client
    participant MCP_Server
    participant Tools
    participant VectorStore

    User->>MCP_Client: Send request<br/>search_docs(query, id)

    MCP_Client->>MCP_Server: JSON-RPC Request
    MCP_Server->>Tools: Dispatch to search_docs

    Tools->>VectorStore: search()
    VectorStore-->>Tools: results
    Tools-->>MCP_Server: formatted_results

    MCP_Server->>MCP_Client: JSON-RPC Response
    MCP_Client-->>User: Display results
```

## Library ID Format

Libraries are identified using path-based identifiers:

- `/pandas` - Main pandas library
- `/pandas/v2.2` - Version-specific identifier
- Case-insensitive matching

## Storage Structure

```
~/.doc-server/
├── chroma.db/          # ChromaDB database
├── models/             # Downloaded embedding models
├── libraries/          # Ingested source data
└── config.yaml         # User configuration
```

## Next Steps

- See the [CLI Reference](cli.md) for usage
- Learn about [MCP Tools](mcp-tools.md)
- Read the [Configuration](configuration.md) guide
- Explore [Examples](examples.md)
