"""
Structured logging configuration for doc-server using structlog.

Provides:
- JSON output for production environments
- Human-readable console output for development
- Context variables for library_id, operation, timing, etc.
- Performance timing hooks
"""

import logging
import sys
from typing import Any

import structlog
from structlog.processors import CallsiteParameterAdder

from .config import settings


def _json_serializer(obj: Any, **kwargs: Any) -> str:
    """
    Fast JSON serializer for production logs.

    Uses orjson if available, falls back to json.
    Accepts additional kwargs for structlog compatibility.
    """
    try:
        import orjson

        # orjson doesn't support sort_keys or other json.dumps kwargs
        # Filter out incompatible options
        orjson_opts = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NAIVE_UTC
        return orjson.dumps(obj, option=orjson_opts).decode("utf-8")
    except ImportError:
        import json

        return json.dumps(obj, default=str, ensure_ascii=False)


def configure_structlog() -> None:
    """
    Configure structlog for the doc-server application.

    Supports both JSON output for production and console output for development.
    Includes context variables for library_id, operation, timing, etc.
    """
    # Shared processors used in both production and development
    shared_processors = [
        # Merge context variables (thread/async-safe)
        structlog.contextvars.merge_contextvars,
        # Add log level
        structlog.stdlib.add_log_level,
        # Format positional arguments
        structlog.stdlib.PositionalArgumentsFormatter(),
        # Add timestamp in ISO 8601 format
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Add callsite information (file, function, line)
        CallsiteParameterAdder(
            parameters={
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            },
        ),
        # Format exception info
        structlog.processors.format_exc_info,
    ]

    # Determine output format based on environment
    is_development = settings.log_format.lower() == "console" or sys.stderr.isatty()

    if is_development:
        # Development: Pretty console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.rich_traceback,
            )
        ]
        logger_factory: Any = structlog.PrintLoggerFactory()
    else:
        # Production: Structured JSON output
        processors = shared_processors + [
            # Add logger name (only for stdlib loggers)
            structlog.stdlib.add_logger_name,
            # Filter by log level (only for stdlib loggers)
            structlog.stdlib.filter_by_level,
            # Convert bytes to unicode
            structlog.processors.UnicodeDecoder(),
            # Render as JSON
            structlog.processors.JSONRenderer(
                serializer=_json_serializer,
            ),
        ]
        logger_factory = structlog.stdlib.LoggerFactory()

    # Configure structlog
    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=logger_factory,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a configured structlog logger."""
    return structlog.get_logger(name)


# Context management functions
def bind_context(**kwargs: Any) -> None:
    """Bind context variables to all subsequent log entries."""
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Unbind specific context variables."""
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()


class LogContext:
    """Context manager for temporary log context."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self._entered = False

    def __enter__(self) -> "LogContext":
        bind_context(**self.kwargs)
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._entered:
            # Unbind the same keys we bound
            unbind_context(*self.kwargs.keys())
