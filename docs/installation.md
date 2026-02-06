# Installation Guide

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager
- 2GB RAM minimum (4GB recommended for embedding models)

## Stable Release

Install from PyPI:

```bash
pip install doc-server
```

## Development Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ArtyomKa/doc-server.git
cd doc-server
pip install -e ".[dev]"
```

This installs all development dependencies including testing and linting tools.

## Verification

Verify the installation:

```bash
doc-server --version
doc-server --help
```

## Platform-Specific Notes

### Linux

No special requirements. Ensure you have Python 3.10+ and pip installed:

```bash
python3 --version
pip3 --version
```

### macOS

Install Python from python.org or via Homebrew:

```bash
brew install python@3.10
pip3 install doc-server
```

### Windows

Install Python 3.10+ from python.org or via Windows Store, then:

```powershell
pip install doc-server
```

## GPU Support (Optional)

For faster embedding generation with GPU acceleration:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install doc-server
```

Ensure your CUDA driver supports the installed PyTorch version.

## Troubleshooting

### Installation Fails

If the installation fails, ensure you have the latest pip:

```bash
pip install --upgrade pip
```

Then retry the installation.

### PyTorch/Torchvision Compatibility

If you see torch-related errors, install torch separately before doc-server:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install doc-server
```

### Permission Errors

On Linux/macOS, use user installation if you encounter permission errors:

```bash
pip install --user doc-server
```

Or use virtual environments:

```bash
python -m venv venv
source venv/bin/activate
pip install doc-server
```

## Next Steps

- Read the [Quick Start Guide](quickstart.md) to get running
- See [Configuration](configuration.md) for customization options
- Explore [Examples](examples.md) for usage patterns
