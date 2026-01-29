# Product Requirement: Tailored Technical Context Service ("The Flattener")

**Version:** 1.0 (Initial Prompt)
**Target Domain:** Local-First RAG for Software Development (Python/C++ Focus)

## 1. Executive Summary
**Objective:** Build a lightweight, specialized Retrieval-Augmented Generation (RAG) service that bypasses web crawling in favor of direct "source-of-truth" ingestion. 
**Problem Solved:** Web crawlers introduce noise (HTML/CSS), latency, and version mismatches. Existing RAG tools often hallucinate APIs.
**Solution:** A "Direct Ingestion" engine that accepts GitHub repositories or ZIP archives, flattens them into pure code/markdown context, and serves them via the Model Context Protocol (MCP).

## 2. Core Philosophy
* **No Crawling:** We do not traverse links. We ingest file trees.
* **Zero Noise:** We filter out all non-essential files (lockfiles, assets, binaries) before the LLM ever sees them.
* **Version Precision:** The user explicitly provides the source (e.g., "Pandas v2.2 Zip"), preventing version hallucination.

## 3. Architecture Specification

### Component A: The Ingestion Layer (Input)
The system must support three specific entry vectors:
1.  **Git Clone (Optimized):**
    * Must use `git clone --depth 1 <url>` to fetch only the latest snapshot (head), ignoring commit history to save bandwidth and storage.
2.  **Zip Upload:**
    * Standard extraction of user-uploaded documentation archives.
3.  **Local Folder Mapping:**
    * Ability to point to a local directory (e.g., `~/docs/pydantic`) for instant indexing.

### Component B: The Gatekeeper (Filtering)
*CRITICAL:* This is the primary quality control mechanism.
* **Denylist Logic:** Must implement a `.gitignore` parser (using `pathspec`) to automatically respect existing repository exclusion rules.
* **Allowlist Logic:** Strictly process only high-value text extensions.
    * *Primary:* `.md`, `.rst`, `.txt` (Documentation)
    * *Code:* `.py`, `.pyi` (Python), `.cpp`, `.h`, `.hpp`, `.c` (C/C++)
* **Binary Guard:** Heuristic check to skip files >1MB or files containing null bytes, to prevent choking the embedding model.

### Component C: The Flattener (Context Packing)
The system must not merely paste text. It must wrap content in metadata headers to preserve file structure awareness for the LLM.

**Format Standard:**
--- START FILE: src/core/engine.cpp ---
[Content of the file]
--- END FILE ---

### Component D: The Serving Layer (MCP)
* **Protocol:** Model Context Protocol (MCP).
* **Tool Exposure:** The service exposes a tool `search_docs(query, library_name)`.
* **Backend:**
    * **Embeddings:** `text-embedding-3-small` (or local equivalent like `all-MiniLM-L6-v2`).
    * **Vector DB:** `ChromaDB` (Local persistent storage).

## 4. Technical Stack Recommendations

* **Language:** Python 3.10+ (Recommended for AI library compatibility).
* **Server Framework:** `FastAPI` / `FastMCP` (For rapid MCP compliance and async performance).
* **Ingestion Tools:**
    * `GitPython`: For managing clone operations.
    * `pathspec`: For robust `.gitignore` parsing.
* **Database:** `ChromaDB` (Running locally, no cloud dependency).

## 5. Implementation Roadmap (MVP)

**Phase 1: The Ingest Script**
Develop a Python script `ingest.py` that accepts a URL/Path, walks the directory tree, filters files against the Gatekeeper rules, and outputs a list of "Document" objects.

**Phase 2: The Vector Index**
Integrate `ChromaDB`. Iterate through the "Document" objects, generate embeddings, and store them in a collection named after the repository (e.g., `collection="langchain_v0_3"`).

**Phase 3: The Agent Interface**
Wrap the query logic in a FastMCP server function. Connect this server to the user's IDE (Cursor/Windsurf).

## 6. Future Scope (Post-MVP)
* **C++ Optimization:** Use `tree-sitter` to parse C++ files and extract only header definitions (interfaces) while ignoring implementation details to save tokens.
* **Auto-Update:** A background cron job that does `git pull` on ingested repos to keep documentation fresh.