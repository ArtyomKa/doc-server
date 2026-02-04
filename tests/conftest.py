"""
Pytest configuration and fixtures for doc-server tests.
"""

import sys
from typing import Any

import pytest

from doc_server.config import Settings
from doc_server.logging_config import configure_structlog


@pytest.fixture(scope="session", autouse=True)
def configure_structlog_for_tests() -> None:
    """
    Configure structlog for test sessions.

    Uses console output to avoid JSON serialization issues in tests.
    """
    # Force development mode for tests to use ConsoleRenderer
    original_stderr = sys.stderr

    # Mock isatty to return False for development mode
    def mock_isatty():
        return False

    # Configure with environment override
    import os

    original_log_format = os.environ.get("DOC_SERVER_LOG_FORMAT")
    os.environ["DOC_SERVER_LOG_FORMAT"] = "console"

    try:
        configure_structlog()
    finally:
        if original_log_format:
            os.environ["DOC_SERVER_LOG_FORMAT"] = original_log_format
        else:
            os.environ.pop("DOC_SERVER_LOG_FORMAT", None)


@pytest.fixture
def config() -> Settings:
    """Create a Settings instance for testing."""
    return Settings()
