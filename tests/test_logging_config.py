"""
Tests for logging configuration module.
"""

import sys
from unittest.mock import patch

import structlog

from doc_server.logging_config import (
    LogContext,
    bind_context,
    clear_context,
    configure_structlog,
    get_logger,
    unbind_context,
)


class TestLoggingConfig:
    """Test cases for logging configuration."""

    def test_configure_structlog_does_not_raise(self):
        """Test that configure_structlog runs without errors."""
        # Should not raise any exceptions
        configure_structlog()

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a structlog logger."""
        logger = get_logger("test")
        assert logger is not None

    def test_bind_context_function(self):
        """Test that bind_context can be called."""
        # Should not raise any exceptions
        bind_context(test_key="test_value")

        # Clean up
        unbind_context("test_key")

    def test_unbind_context_function(self):
        """Test that unbind_context can be called."""
        # Should not raise any exceptions
        unbind_context("nonexistent_key")

    def test_clear_context_function(self):
        """Test that clear_context can be called."""
        # Should not raise any exceptions
        clear_context()

    def test_log_context_manager(self):
        """Test LogContext context manager."""
        with LogContext(test_key="test_value"):
            # Inside context, should be able to log
            logger = get_logger("test")
            logger.info("Test log message")

        # After context, should clean up
        unbind_context("test_key")

    def test_log_context_nested(self):
        """Test nested LogContext managers."""
        with LogContext(key1="value1"):
            with LogContext(key2="value2"):
                logger = get_logger("test")
                logger.info("Nested context test")


class TestDevelopmentModeLogging:
    """Test logging in development mode with console output."""

    def test_development_mode_logging_with_printlogger(self):
        """
        Test that logging works in development mode with PrintLogger.

        This reproduces the bug where 'PrintLogger' object has no attribute 'name'
        because add_logger_name processor was applied to PrintLogger which doesn't
        have a .name attribute.
        """
        # Clear any existing configuration
        structlog.reset_defaults()

        # Mock settings to use console format and simulate TTY
        with patch("doc_server.logging_config.settings") as mock_settings:
            mock_settings.log_format = "console"
            mock_settings.log_level = "INFO"

            # Mock sys.stderr.isatty() to return True (development mode)
            with patch.object(sys.stderr, "isatty", return_value=True):
                # Configure structlog in development mode
                configure_structlog()

                # Get a logger and try to log - this should NOT raise
                logger = get_logger("test_dev_mode")

                # This is the actual test - logging should work without error
                try:
                    logger.info("Test message in development mode")
                except AttributeError as e:
                    if "'PrintLogger' object has no attribute 'name'" in str(e):
                        raise AssertionError(
                            "PrintLogger.name bug reproduced: "
                            "add_logger_name processor incompatible with PrintLogger"
                        ) from e
                    raise
