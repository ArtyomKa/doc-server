"""
ZIP archive extraction functionality with security features.
"""

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from doc_server.config import Settings

logger = structlog.get_logger()


@dataclass
class ArchiveMetadata:
    """Metadata extracted from a ZIP archive."""

    archive_path: str
    extract_path: Path
    total_files: int
    total_directories: int
    total_size: int
    compressed_size: int
    archive_size: int
    earliest_date: str
    latest_date: str


class ZIPExtractionError(Exception):
    """Raised when ZIP extraction fails."""

    def __init__(self, message: str, archive_path: str) -> None:
        """Initialize error with context."""
        self.archive_path = archive_path
        super().__init__(message)


class InvalidArchiveError(ZIPExtractionError):
    """Raised when a ZIP archive is invalid or corrupted."""

    pass


class PathTraversalError(ZIPExtractionError):
    """Raised when path traversal attempt is detected."""

    pass


class ZIPExtractor:
    """
    ZIP archive extractor with security features and metadata extraction.

    Provides secure extraction with path traversal protection
    and metadata extraction for documentation ingestion.
    """

    def __init__(self, config: Settings) -> None:
        """
        Initialize the ZIPExtractor.

        Args:
            config: Application settings instance
        """
        self._config = config
        self._temp_extractions: list[Path] = []

    def extract_archive(
        self,
        archive_path: str,
        destination: Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Extract a ZIP archive securely with path traversal protection.

        Args:
            archive_path: Path to the ZIP archive file
            destination: Directory where files should be extracted. If None,
                        creates a temporary directory that needs to be cleaned up.
            overwrite: Whether to overwrite existing files during extraction

        Returns:
            Path: Path to the extraction directory

        Raises:
            FileNotFoundError: If archive_path doesn't exist
            InvalidArchiveError: If the file is not a valid ZIP archive
            PathTraversalError: If archive contains path traversal attempts
            ZIPExtractionError: For other extraction failures
        """
        # Validate archive path
        archive_file = Path(archive_path)
        if not archive_file.exists():
            raise FileNotFoundError(f"Archive file not found: {archive_path}")
        if not archive_file.is_file():
            raise InvalidArchiveError(
                f"Archive path is not a file: {archive_path}",
                archive_path,
            )

        # Create extraction directory
        if destination is None:
            destination = Path(tempfile.mkdtemp(prefix="doc-server-extract-"))
            self._temp_extractions.append(destination)
            logger.info(
                "Created temporary extraction directory",
                archive_path=archive_path,
                path=str(destination),
            )
        else:
            destination.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(archive_file, mode="r") as zip_ref:
                logger.info(
                    "Extracting archive",
                    archive_path=archive_path,
                    destination=str(destination),
                    file_count=len(zip_ref.infolist()),
                )

                # Extract each file with security checks
                for member in zip_ref.infolist():
                    self._validate_member_path(member, destination)

                    if not overwrite:
                        output_path = destination / member.filename
                        if output_path.exists():
                            logger.warning(
                                "Skipping existing file",
                                file=member.filename,
                                path=str(output_path),
                            )
                            continue

                    # Extract file
                    zip_ref.extract(member, destination)

                logger.info(
                    "Successfully extracted archive",
                    archive_path=archive_path,
                    destination=str(destination),
                )

                return destination

        except PathTraversalError:
            # Re-raise path traversal errors as-is
            raise
        except zipfile.BadZipFile as exc:
            raise InvalidArchiveError(
                f"Invalid or corrupted ZIP archive: {exc}",
                archive_path,
            ) from exc
        except RuntimeError as exc:
            raise ZIPExtractionError(
                f"Failed to extract archive: {exc}",
                archive_path,
            ) from exc
        except Exception as exc:
            raise ZIPExtractionError(
                f"Unexpected error extracting archive: {exc}",
                archive_path,
            ) from exc

    def extract_metadata(
        self,
        archive_path: str,
        extract_path: Path,
    ) -> ArchiveMetadata:
        """
        Extract metadata from a ZIP archive.

        Args:
            archive_path: Path to the ZIP archive file
            extract_path: Path where archive was extracted

        Returns:
            ArchiveMetadata: Metadata about the archive

        Raises:
            InvalidArchiveError: If the file is not a valid ZIP archive
            ZIPExtractionError: If metadata extraction fails
        """
        try:
            archive_file = Path(archive_path)
            if not archive_file.exists():
                raise FileNotFoundError(f"Archive not found: {archive_path}")

            with zipfile.ZipFile(archive_file, mode="r") as zip_ref:
                members = zip_ref.infolist()

                # Count files and directories
                file_count = 0
                dir_count = 0
                total_size = 0
                compressed_size = 0
                earliest_date = None
                latest_date = None

                for member in members:
                    if member.is_dir():
                        dir_count += 1
                    else:
                        file_count += 1
                        total_size += member.file_size
                        compressed_size += member.compress_size

                    # Track dates
                    member_date = datetime(*member.date_time[:6])
                    if earliest_date is None or member_date < earliest_date:
                        earliest_date = member_date
                    if latest_date is None or member_date > latest_date:
                        latest_date = member_date

                # Get archive file size
                archive_size = archive_file.stat().st_size

                metadata = ArchiveMetadata(
                    archive_path=archive_path,
                    extract_path=extract_path,
                    total_files=file_count,
                    total_directories=dir_count,
                    total_size=total_size,
                    compressed_size=compressed_size,
                    archive_size=archive_size,
                    earliest_date=earliest_date.isoformat() if earliest_date else "",
                    latest_date=latest_date.isoformat() if latest_date else "",
                )

                logger.info(
                    "Extracted archive metadata",
                    archive_path=archive_path,
                    file_count=file_count,
                    dir_count=dir_count,
                    total_size=total_size,
                )

                return metadata

        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise
        except zipfile.BadZipFile as exc:
            raise InvalidArchiveError(
                f"Invalid or corrupted ZIP archive: {exc}",
                archive_path,
            ) from exc
        except Exception as exc:
            raise ZIPExtractionError(
                f"Failed to extract metadata: {exc}",
                archive_path,
            ) from exc

    def cleanup_extraction(self, extract_path: Path) -> bool:
        """
        Remove an extracted archive directory.

        Args:
            extract_path: Path to the extracted archive directory

        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            if extract_path.exists():
                shutil.rmtree(extract_path)
                if extract_path in self._temp_extractions:
                    self._temp_extractions.remove(extract_path)
                logger.info("Cleaned up extraction directory", path=str(extract_path))
                return True
            return False
        except OSError as exc:
            logger.error(
                "Failed to cleanup extraction directory",
                path=str(extract_path),
                error=str(exc),
            )
            return False

    def cleanup_all_temp_extractions(self) -> int:
        """
        Clean up all temporary extraction directories created by this instance.

        Returns:
            int: Number of extractions cleaned up
        """
        cleaned_count = 0
        for extract_path in list(self._temp_extractions):
            if self.cleanup_extraction(extract_path):
                cleaned_count += 1
        return cleaned_count

    def _validate_member_path(
        self,
        member: zipfile.ZipInfo,
        extract_path: Path,
    ) -> None:
        """
        Validate that a ZIP member doesn't contain path traversal attempts.

        Args:
            member: ZipInfo object for the member
            extract_path: Destination extraction path

        Raises:
            PathTraversalError: If path traversal is detected
        """
        # Normalize the member filename to resolve any '..' or '.'
        member_path = Path(member.filename)
        resolved_path = (extract_path / member_path).resolve()

        # Check if the resolved path is outside the extraction directory
        try:
            # Check if the resolved path starts with extraction directory
            extract_resolved = extract_path.resolve()
            resolved_path.relative_to(extract_resolved)
        except ValueError as exc:
            raise PathTraversalError(
                f"Path traversal attempt detected: {member.filename}",
                member.filename,
            ) from exc

        # Additional check: verify member doesn't contain '..' components
        if ".." in member.filename.split("/") or ".." in member.filename.split("\\"):
            raise PathTraversalError(
                f"Path traversal attempt detected: {member.filename}",
                member.filename,
            )

    def __enter__(self) -> "ZIPExtractor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup temp extractions."""
        self.cleanup_all_temp_extractions()
