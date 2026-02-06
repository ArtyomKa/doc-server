# Release Guide for doc-server

This document describes the manual release process for doc-server. Version management is currently manual (automated versioning tools like `bump2version` or `semantic-release` may be adopted in future phases).

## Versioning Strategy

We follow [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.y.z): Incompatible API changes
- **MINOR** (x.Y.z): New functionality, backwards compatible
- **PATCH** (x.y.Z): Bug fixes, backwards compatible

### Pre-release Versions

Pre-release versions use the format: `{major}.{minor}.{patch}-{prerelease_type}.{number}`

Supported prerelease types:
- **alpha**: Early testing, unstable (`0.1.0-alpha.1`)
- **beta**: Feature complete, testing phase (`0.1.0-beta.1`)
- **rc** (release candidate): Final testing before stable (`0.1.0-rc.1`)

### Example Version Progression

```
0.1.0-alpha.1 → 0.1.0-alpha.2 → 0.1.0-beta.1 → 0.1.0-rc.1 → 0.1.0
```

## Pre-Release Checklist

Before creating a release:

- [ ] All tests pass (`pytest`)
- [ ] Code quality checks pass (`make lint` or `black`, `isort`, `ruff`, `mypy`)
- [ ] Documentation is up to date
- [ ] CHANGELOG.md is updated with changes since last release
- [ ] Version numbers are updated in:
  - [ ] `doc_server/__init__.py` (primary source of truth)
  - [ ] `pyproject.toml` (hatchling reads from `__init__.py` automatically)

## Step-by-Step Release Process

### 1. Update Version

Edit `doc_server/__init__.py` and update the `__version__` variable:

```python
__version__ = "0.1.0"  # or pre-release: "0.1.0-alpha.1"
```

The hatchling build system automatically reads the version from `doc_server/__init__.py`, so you don't need to update `pyproject.toml`.

### 2. Update Changelog

Update `CHANGELOG.md` with changes since the last release. Follow the [Keep a Changelog](https://keepachangelog.com/) format.

### 3. Commit Changes

```bash
git add doc_server/__init__.py CHANGELOG.md
git commit -m "chore: bump version to X.Y.Z"
```

### 4. Create and Push Tag

Tags must follow the format `v{major}.{minor}.{patch}[-{prerelease}]`:

```bash
# For stable releases
git tag -a v0.1.0 -m "Release version 0.1.0"

# For pre-releases
git tag -a v0.1.0-alpha.1 -m "Release version 0.1.0-alpha.1"
git tag -a v0.1.0-beta.1 -m "Release version 0.1.0-beta.1"
git tag -a v0.1.0-rc.1 -m "Release version 0.1.0-rc.1"

# Push tag to trigger release workflow
git push origin v0.1.0
```

### 5. Verify Release

The GitHub Actions release workflow will automatically:
1. Build wheel and source distribution
2. Create a GitHub Release
3. Attach artifacts
4. Mark as pre-release if tag contains alpha/beta/rc

### 6. Post-Release Verification

After the release workflow completes:

```bash
# Download wheel from GitHub Release
pip install doc-server-X.Y.Z-py3-none-any.whl

# Verify version
doc-server --version

# Test basic functionality
doc-server health
```

## Rollback Procedures

If a release needs to be withdrawn:

1. **Delete the GitHub Release** (from repository Releases page)
2. **Delete the git tag**:
   ```bash
   git push --delete origin v0.1.0
   git tag -d v0.1.0
   ```
3. **Fix issues and re-release** with incremented version or pre-release number

## When to Bump Version Components

### MAJOR (breaking changes)
- Incompatible API changes to MCP tools
- Breaking changes to CLI commands
- Removal of functionality

### MINOR (new features)
- New MCP tools added
- New CLI commands or options
- New ingestion sources supported
- Performance improvements

### PATCH (bug fixes)
- Bug fixes
- Security patches
- Documentation improvements
- Dependency updates

## Future Automation

Planned improvements for future phases:

- [ ] **bump2version**: Automated version bumping across files
- [ ] **semantic-release**: Automated versioning based on conventional commits
- [ ] **automatic changelog**: Generated from commit messages
- [ ] **signed releases**: GPG-signed tags and artifacts

## Questions?

For questions about the release process, please:
1. Check existing releases for examples
2. Review the GitHub Actions workflow in `.github/workflows/release.yml`
3. Open an issue in the repository
