# Implementation Plan: Makefile and CI Standardization

## Overview
Create a comprehensive Makefile to standardize all development commands and update CI workflow to use it, ensuring consistency between local development and CI/CD processes.

## Files to Create/Modify

### 1. Create: `/home/artyom/devel/doc-server/Makefile`
**Purpose**: Centralized command definitions for all development tasks

**Targets**:
- `install` - Install dependencies with `uv sync`
- `dev` - Install with dev dependencies
- `format` - Auto-format with Black + isort
- `lint` - Run all linting checks (Black check, isort check, Ruff)
- `lint-fix` - Fix auto-fixable issues
- `typecheck` - Run MyPy type checking
- `test` - Run all tests
- `test-cov` - Run tests with coverage (85% threshold)
- `test-fast` - Run quick tests only (exclude slow/performance/integration)
- `serve` - Run MCP server
- `ci` - Run full CI pipeline
- `clean` - Clean cache files and artifacts
- `help` - Show help message

### 2. Modify: `.github/workflows/test.yml`
**Purpose**: Replace direct tool invocations with Makefile targets

**Changes**:
- Replace individual Black/isort/Ruff commands with `make lint`
- Replace test execution with `make test-cov`
- Add Makefile availability check

### 3. Modify: `WORKFLOW.md`
**Purpose**: Update documentation to promote Makefile usage

**Changes**:
- Add "Quick Start with Make" section at the top
- Replace manual uv commands with Make equivalents as primary examples
- Keep manual commands as "Advanced Usage" reference
- Update pre-commit validation section to use `make ci`

### 4. Modify: `TESTING.md`
**Purpose**: Update testing documentation for Makefile

**Changes**:
- Add "Quick Start with Make" section
- Update coverage threshold from 90% to 85%
- Replace manual pytest commands with Make equivalents as primary examples
- Keep manual commands for specific/advanced scenarios

### 5. Modify: `AGENTS.md`
**Purpose**: Add note about future security scanning

**Changes**:
- Add note: "Security scanning (bandit, safety) is planned for future implementation"

## Benefits
1. **Single Source of Truth**: Makefile centralizes all tool configurations
2. **Consistency**: Same commands locally and in CI
3. **Simplicity**: Developers just run `make ci` before committing
4. **Maintainability**: Easy to add new tools or change configurations
5. **Documentation**: Clear, consistent examples across all docs

## Acceptance Criteria
- [ ] Makefile created with all specified targets
- [ ] CI workflow updated to use Makefile targets
- [ ] All documentation files updated
- [ ] Coverage threshold consistently set to 85% across all files
- [ ] Security scanning note added to AGENTS.md
