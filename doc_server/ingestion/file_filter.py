"""
File filtering and selection functionality.

Provides filtering of files based on .gitignore patterns, allowlist,
file size limits, and binary content detection.
"""

from pathlib import Path

import structlog
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPattern
from pydantic import BaseModel

from doc_server.config import Settings

logger = structlog.get_logger()


class FilterResult(BaseModel):
    """Result of filtering a file."""

    file_path: str
    included: bool
    reason: str
    extension: str
    size_bytes: int
    is_binary: bool | None = None
    gitignore_matched: bool = False


class FileFilterError(Exception):
    """Base exception for file filtering errors."""

    pass


class InvalidGitignoreError(FileFilterError):
    """Raised when .gitignore file cannot be parsed."""

    def __init__(self, message: str, gitignore_path: str) -> None:
        """Initialize error with context."""
        self.gitignore_path = gitignore_path
        super().__init__(message)


class FileSizeExceededError(FileFilterError):
    """Raised when file size exceeds limit."""

    def __init__(
        self, message: str, file_path: str, file_size: int, max_size: int
    ) -> None:
        """Initialize error with context."""
        self.file_path = file_path
        self.file_size = file_size
        self.max_size = max_size
        super().__init__(message)


class BinaryFileError(FileFilterError):
    """Raised when a file contains binary content."""

    def __init__(self, message: str, file_path: str) -> None:
        """Initialize error with context."""
        self.file_path = file_path
        super().__init__(message)


class FileFilter:
    """
    File filter with .gitignore support, allowlist, and binary detection.

    Provides comprehensive filtering for documentation ingestion including:
    - .gitignore pattern matching using pathspec
    - Allowlist enforcement for file extensions
    - File size limits (default 1MB)
    - Binary content detection (null byte check)
    """

    def __init__(self, config: Settings) -> None:
        """
        Initialize the FileFilter.

        Args:
            config: Application settings instance
        """
        self._config = config
        self._allowed_extensions = {
            ext.lower() for ext in self._config.allowed_extensions
        }
        logger.info(
            "FileFilter initialized",
            max_file_size=self._config.max_file_size,
            allowed_extensions=list(self._allowed_extensions),
        )

    def load_gitignore(
        self,
        gitignore_path: str | None,
    ) -> PathSpec | None:
        """
        Load .gitignore patterns from a file.

        Args:
            gitignore_path: Path to .gitignore file. If None, returns None.

        Returns:
            PathSpec: Compiled pattern matcher, or None if no gitignore

        Raises:
            InvalidGitignoreError: If .gitignore file cannot be read or parsed
        """
        if gitignore_path is None:
            logger.debug("No .gitignore path provided")
            return None

        from pathlib import Path

        path = Path(gitignore_path)
        if not path.exists():
            logger.debug(".gitignore file not found", path=str(path))
            return None

        try:
            with open(path, encoding="utf-8") as f:
                patterns = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]

            if not patterns:
                logger.debug(".gitignore file is empty", path=str(path))
                return None

            spec = PathSpec.from_lines(GitWildMatchPattern, patterns)
            logger.info(
                "Loaded .gitignore patterns",
                path=str(path),
                pattern_count=len(patterns),
            )
            return spec

        except UnicodeDecodeError as exc:
            raise InvalidGitignoreError(
                f"Failed to decode .gitignore file (not UTF-8): {exc}",
                str(path),
            ) from exc
        except Exception as exc:
            raise InvalidGitignoreError(
                f"Failed to load .gitignore file: {exc}",
                str(path),
            ) from exc

    def should_include_file(
        self,
        file_path: str,
        gitignore: PathSpec | None = None,
        base_path: str | None = None,
    ) -> bool:
        """
        Determine if a file should be included based on filtering rules.

        Args:
            file_path: Path to the file to check
            gitignore: Optional PathSpec from .gitignore patterns
            base_path: Base directory path for gitignore matching.
                      If None, uses file_path's parent directory.

        Returns:
            bool: True if file should be included, False otherwise

        Raises:
            FileFilterError: If file cannot be accessed
        """
        from pathlib import Path

        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            logger.warning("File not found during filtering", path=str(path))
            return False

        # Check if it's a file (not directory)
        if not path.is_file():
            logger.debug("Path is not a file", path=str(path))
            return False

        # Check gitignore patterns
        if gitignore is not None:
            rel_path = (
                path.relative_to(base_path)
                if base_path and path.is_relative_to(base_path)
                else path
            )
            gitignore_path = str(rel_path).replace("\\", "/")

            if gitignore.match_file(gitignore_path):
                logger.debug("File excluded by .gitignore", path=str(path))
                return False

        # Check extension against allowlist
        if not self._check_extension(path):
            logger.debug("File excluded by allowlist", path=str(path))
            return False

        # Check file size
        if not self._check_file_size(path):
            logger.debug("File excluded by size limit", path=str(path))
            return False

        # Check for binary content
        if self._check_binary_content(path):
            logger.debug("File excluded as binary", path=str(path))
            return False

        # All checks passed
        return True

    def filter_file(
        self,
        file_path: str,
        gitignore: PathSpec | None = None,
        base_path: str | None = None,
    ) -> FilterResult:
        """
        Filter a file and return detailed result.

        Args:
            file_path: Path to the file to filter
            gitignore: Optional PathSpec from .gitignore patterns
            base_path: Base directory path for gitignore matching

        Returns:
            FilterResult: Detailed filtering result

        Raises:
            FileFilterError: If file cannot be accessed
        """
        from pathlib import Path

        path = Path(file_path)

        # Gather file info
        try:
            size_bytes = path.stat().st_size if path.exists() else 0
            extension = path.suffix.lower()
        except OSError as exc:
            raise FileFilterError(f"Cannot access file: {exc}") from exc

        # Check gitignore first
        if gitignore is not None:
            rel_path = (
                path.relative_to(base_path)
                if base_path and path.is_relative_to(base_path)
                else path
            )
            gitignore_path = str(rel_path).replace("\\", "/")

            if gitignore.match_file(gitignore_path):
                return FilterResult(
                    file_path=str(path),
                    included=False,
                    reason="Excluded by .gitignore",
                    extension=extension,
                    size_bytes=size_bytes,
                    is_binary=None,
                    gitignore_matched=True,
                )

        # Check extension
        if extension not in self._allowed_extensions:
            return FilterResult(
                file_path=str(path),
                included=False,
                reason=f"Extension '{extension}' not in allowlist",
                extension=extension,
                size_bytes=size_bytes,
                is_binary=None,
                gitignore_matched=False,
            )

        # Check file size
        if size_bytes > self._config.max_file_size:
            return FilterResult(
                file_path=str(path),
                included=False,
                reason=f"File size {size_bytes} exceeds limit {self._config.max_file_size}",
                extension=extension,
                size_bytes=size_bytes,
                is_binary=None,
                gitignore_matched=False,
            )

        # Check binary content
        is_binary = self._check_binary_content(path)
        if is_binary:
            return FilterResult(
                file_path=str(path),
                included=False,
                reason="File contains binary content (null bytes)",
                extension=extension,
                size_bytes=size_bytes,
                is_binary=True,
                gitignore_matched=False,
            )

        # All checks passed
        return FilterResult(
            file_path=str(path),
            included=True,
            reason="File passes all filters",
            extension=extension,
            size_bytes=size_bytes,
            is_binary=False,
            gitignore_matched=False,
        )

    def filter_directory(
        self,
        directory_path: str,
        gitignore: PathSpec | None = None,
    ) -> list[FilterResult]:
        """
        Filter all files in a directory recursively.

        Args:
            directory_path: Path to the directory to filter
            gitignore: Optional PathSpec from .gitignore patterns

        Returns:
            list[FilterResult]: List of filtering results for all files

        Raises:
            FileFilterError: If directory cannot be accessed
        """
        from pathlib import Path

        directory = Path(directory_path)

        if not directory.exists():
            raise FileFilterError(f"Directory not found: {directory_path}")

        if not directory.is_dir():
            raise FileFilterError(f"Path is not a directory: {directory_path}")

        results: list[FilterResult] = []
        logger.info("Filtering directory", directory=str(directory))

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    result = self.filter_file(str(file_path), gitignore, str(directory))
                    results.append(result)
                except FileFilterError as exc:
                    logger.warning(
                        "Failed to filter file",
                        path=str(file_path),
                        error=str(exc),
                    )
                    # Create a result with failure info
                    results.append(
                        FilterResult(
                            file_path=str(file_path),
                            included=False,
                            reason=f"Filtering error: {exc}",
                            extension=file_path.suffix.lower(),
                            size_bytes=(
                                file_path.stat().st_size if file_path.exists() else 0
                            ),
                            is_binary=None,
                            gitignore_matched=False,
                        )
                    )

        included_count = sum(1 for r in results if r.included)
        logger.info(
            "Directory filtering complete",
            directory=str(directory),
            total_files=len(results),
            included=included_count,
            excluded=len(results) - included_count,
        )

        return results

    def _check_extension(self, file_path: Path) -> bool:
        """
        Check if file extension is in the allowlist.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if extension is allowed
        """
        extension = file_path.suffix.lower()
        return extension in self._allowed_extensions

    def _check_file_size(self, file_path: Path) -> bool:
        """
        Check if file size is within limits.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file size is within limit

        Raises:
            FileSizeExceededError: If file size exceeds limit
        """
        try:
            size = file_path.stat().st_size
            if size > self._config.max_file_size:
                logger.debug(
                    "File size exceeded",
                    path=str(file_path),
                    size=size,
                    limit=self._config.max_file_size,
                )
                return False
            return True
        except OSError as exc:
            logger.error("Failed to get file size", path=str(file_path), error=str(exc))
            return False

    def _check_binary_content(self, file_path: Path) -> bool:
        """
        Check if file contains binary content by detecting null bytes.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file contains binary content
        """
        try:
            # Read first 8192 bytes to check for null bytes
            with open(file_path, "rb") as f:
                chunk = f.read(8192)
                # Check for null bytes (characteristic of binary files)
                if b"\x00" in chunk:
                    logger.debug("Binary content detected", path=str(file_path))
                    return True
            return False
        except OSError as exc:
            logger.error(
                "Failed to check binary content", path=str(file_path), error=str(exc)
            )
            # Assume binary if we can't read
            return True
