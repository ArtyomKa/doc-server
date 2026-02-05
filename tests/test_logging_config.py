"""
Tests for logging configuration module.
"""


from structlog import get_logger

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
