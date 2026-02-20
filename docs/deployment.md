# Deployment Guide

Doc Server supports two deployment modes: **Local** (default) and **Remote Backend**.

## Remote Backend Architecture

In the Remote Backend architecture, a centralized server hosts the vector database and embedding model, while lightweight clients (CLI or MCP) communicate with it via a REST API. This is ideal for teams sharing documentation or offloading heavy processing.

### 1. Running the Backend Server

You can run the backend server using the CLI or Docker.

#### Option A: Using CLI

```bash
# Start server on port 8000
doc-server backend --port 8000 --host 0.0.0.0

# With API Key authentication (Recommended)
export DOC_SERVER_BACKEND_API_KEY="your-secret-key"
doc-server backend --port 8000
```

#### Option B: Using Docker

A `Dockerfile` is included in the repository.

```bash
# Build the image
docker build -t doc-server .

# Run the container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  -e DOC_SERVER_BACKEND_API_KEY="your-secret-key" \
  doc-server
```

The container exposes port 8000 and stores data in `/data`.

### 2. Configuring the Client

Configure your local `doc-server` (CLI or MCP) to use the remote backend.

#### Environment Variables

```bash
export DOC_SERVER_MODE=remote
export DOC_SERVER_BACKEND_URL=http://localhost:8000
export DOC_SERVER_BACKEND_API_KEY=your-secret-key
```

#### Configuration File (`config.yaml`)

```yaml
mode: remote
backend_url: "http://localhost:8000"
backend_api_key: "your-secret-key"
```

### 3. Verifying Connection

Once configured, verify the connection using the health check:

```bash
# Should return health status from the REMOTE server
doc-server health
```

### 4. Production Deployment

For production deployments, we recommend:
1.  **HTTPS**: Run behind a reverse proxy (Nginx, Traefik) to terminate SSL.
2.  **Authentication**: Always set `DOC_SERVER_BACKEND_API_KEY`.
3.  **Persistence**: Mount a volume for `/data` to persist the vector database.
4.  **Workers**: Increase workers for higher concurrency: `doc-server backend --workers 4`.

#### Docker Compose Example

```yaml
version: '3.8'

services:
  doc-server:
    image: doc-server:latest
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./doc_data:/data
    environment:
      - DOC_SERVER_BACKEND_API_KEY=${API_KEY}
      - DOC_SERVER_STORAGE_PATH=/data
    restart: unless-stopped
```
