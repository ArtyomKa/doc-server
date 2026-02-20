# Remote Backend Requirements

## 1. Executive Summary

This document outlines the requirements for extending Doc Server to support a **Remote Backend** architecture. This architecture enables a centralized documentation server that can be shared across a team of developers, providing shared storage, GPU resources, and consistent search results.

## 2. Problem Statement

Currently, Doc Server runs entirely locally. This presents several challenges for team environments:
- **Resource Duplication**: Every developer must ingest and embed the same documentation libraries locally.
- **Hardware Requirements**: Every developer needs sufficient RAM and CPU/GPU power to run embedding models.
- **Inconsistency**: Different developers might have different versions of documentation indexed.
- **Lack of Centralization**: No central control over what documentation is available to the team.

## 3. Goals & Objectives

- **Centralization**: Enable a single server to host the vector database and embedding service.
- **Resource Efficiency**: Offload heavy embedding tasks to a shared server (potentially with GPU).
- **Consistency**: Ensure all developers search against the same documentation corpus.
- **Flexibility**: Support both the existing "Local Mode" and the new "Remote Mode" seamlessly.
- **Security**: Provide basic authentication for the remote server.

## 4. Use Cases

### UC-001: Team Development
A development team sets up a shared Doc Server instance on a GPU-enabled server. The team lead ingests relevant documentation (internal docs, libraries). Developers configure their local MCP clients to point to this server, enabling instant access to shared documentation without local processing.

### UC-003: Low-Power Client
A developer working on a lightweight laptop (e.g., MacBook Air) connects to a powerful remote server, getting fast search results without draining local battery or CPU.

## 5. Functional Requirements

### System Modes
- **REQ-001**: The system MUST support a `local` mode (default) where all components run locally.
- **REQ-002**: The system MUST support a `remote` mode where the MCP client communicates with a backend server via REST API.

### Backend Server
- **REQ-003**: The backend MUST provide a REST API for all core functionality (Search, Ingest, List, Remove, Health).
- **REQ-004**: The backend MUST support API Key authentication via HTTP headers.
- **REQ-005**: The backend MUST handle full ingestion pipelines (Clone -> Process -> Embed -> Store) triggered by a single API call, accepting an optional user-supplied **version identifier**.
- **REQ-006**: The backend MUST support configurable storage paths for the vector database and models.
- **REQ-007**: The backend MUST expose a health check endpoint for monitoring.

### MCP Client (Remote Mode)
- **REQ-008**: The MCP client MUST be configurable via environment variables or config file to use a remote backend.
- **REQ-009**: The MCP client MUST forward search queries to the remote backend.
- **REQ-010**: The MCP client MUST forward ingestion requests (source URL + library ID + optional version) to the remote backend.
- **REQ-011**: The MCP client MUST handle network timeouts and errors gracefully.

### CLI
- **REQ-012**: The CLI MUST support a command to start the backend server (`doc-server backend`).
- **REQ-013**: The CLI `ingest` and `search` commands MUST respect the configured mode (local vs remote).

## 6. Non-Functional Requirements

- **NFR-001 Performance**: Search requests to the remote backend should complete within 200ms (excluding network latency).
- **NFR-002 Scalability**: The backend should handle concurrent search requests from multiple developers.
- **NFR-003 Security**: API Keys must be validated for all privileged operations.
- **NFR-004 Compatibility**: The API should be versioned (`/api/v1/`) to support future changes.
- **NFR-005 Deployment**: The backend must be deployable via Docker and as a standard Python process.

## 7. Constraints & Assumptions

- **Network**: Clients are assumed to have network access to the backend server.
- **Ingestion**: In remote mode, ingestion is performed entirely on the backend. Local files cannot be ingested to a remote server in this phase (future extension).
- **Single Tenant**: The system is designed for a single trusted team (shared API key), not multi-tenancy.

## 8. Testing Requirements

### Unit Tests
- **Coverage**: Full coverage (100% target) of all new components:
  - API Client (`api_client.py`)
  - API Server endpoints (`api_server.py`)
  - Configuration changes (`config.py`)

### Integration Tests
- **Mocked Backend**: Client communication with mocked FastAPI server to verify protocol.
- **Local Backend**: Client communication with actual local backend server instance.

### End-to-End (E2E) Tests
- **Full CRUD Operations**:
  - Ingest a library via remote backend (with version).
  - Search documents via remote backend.
  - List libraries via remote backend (verify version presence).
  - Remove library via remote backend.

## 9. Acceptance Criteria

The feature is considered complete when:
1. All unit tests pass.
2. All integration tests pass.
3. All E2E tests pass.
4. Code passes all linting and type checks (ruff, black, mypy).
5. Manual verification of the end-to-end flow (CLI -> Remote Backend) is successful.
