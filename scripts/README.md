# Development Scripts

This directory contains development and maintenance scripts for the doc-server project.

## Available Scripts

### verify_components.py
A manual verification script to test that core doc-server components are working correctly.

**Purpose:**
- Quick validation of component connectivity
- Development-time debugging and troubleshooting
- Manual testing without running the full pytest suite

**Usage:**
```bash
# From project root
python scripts/verify_components.py

# Or from scripts directory
python verify_components.py
```

**What it tests:**
1. Configuration loading from Settings
2. File filtering functionality
3. Document processing and chunking
4. Embedding service connectivity

**Output:**
- Real-time status updates for each component
- Success/failure indicators
- Summary of component health

**Note:** This is a development tool, not a replacement for the comprehensive test suite in `tests/`. For full testing, use `pytest tests/`.