"""
Unit tests for ZIPExtractor module.
"""

import tempfile
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from doc_server.config import Settings
from doc_server.ingestion.zip_extractor import (
    ArchiveMetadata,
    InvalidArchiveError,
    PathTraversalError,
    ZIPExtractor,
)


@pytest.fixture
def config() -> Settings:
    """Create a Settings instance for testing."""
    return Settings()


@pytest.fixture
def zip_extractor(config: Settings) -> ZIPExtractor:
    """Create a ZIPExtractor instance for testing."""
    return ZIPExtractor(config)


@pytest.fixture
def sample_zip_file(tmp_path: Path) -> Path:
    """Create a sample ZIP archive for testing."""
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        # Add a file
        zf.writestr("file1.txt", "Content of file 1")
        # Add another file in a subdirectory
        zf.writestr("subdir/file2.txt", "Content of file 2")
        # Add a directory
        zf.writestr("dir1/", "")
    return zip_path


@pytest.fixture
def path_traversal_zip_file(tmp_path: Path) -> Path:
    """Create a ZIP archive with path traversal attempt."""
    zip_path = tmp_path / "traversal.zip"
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        # This would extract to parent directory - path traversal
        zf.writestr("../evil.txt", "Evil content")
    return zip_path


@pytest.fixture
def nested_path_traversal_zip_file(tmp_path: Path) -> Path:
    """Create a ZIP archive with nested path traversal attempt."""
    zip_path = tmp_path / "nested_traversal.zip"
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        # Nested traversal attempt
        zf.writestr("safe/../../evil.txt", "Evil content")
    return zip_path


@pytest.fixture
def large_zip_file(tmp_path: Path) -> Path:
    """Create a ZIP archive with multiple files for metadata testing."""
    zip_path = tmp_path / "large.zip"
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        # Add multiple files
        for i in range(5):
            zf.writestr(f"file{i}.txt", f"Content {i}")
        # Add directories
        zf.writestr("dir1/", "")
        zf.writestr("dir2/subdir/", "")
    return zip_path


class TestArchiveMetadata:
    """Tests for ArchiveMetadata dataclass."""

    def test_archive_metadata_creation(self) -> None:
        """Test creating ArchiveMetadata instance."""
        metadata = ArchiveMetadata(
            archive_path="/path/to/archive.zip",
            extract_path=Path("/tmp/extract"),
            total_files=10,
            total_directories=3,
            total_size=1024,
            compressed_size=512,
            archive_size=2048,
            earliest_date="2024-01-01T00:00:00",
            latest_date="2024-01-02T00:00:00",
        )

        assert metadata.archive_path == "/path/to/archive.zip"
        assert metadata.total_files == 10
        assert metadata.total_directories == 3


class TestZIPExtractorExtraction:
    """Tests for ZIPExtractor extraction functionality."""

    def test_extract_to_temporary_directory(
        self,
        zip_extractor: ZIPExtractor,
        sample_zip_file: Path,
    ) -> None:
        """Test extracting to a temporary directory."""
        extract_path = zip_extractor.extract_archive(
            archive_path=str(sample_zip_file),
        )

        assert extract_path.exists()
        assert (extract_path / "file1.txt").exists()
        assert (extract_path / "subdir" / "file2.txt").exists()
        assert (extract_path / "file1.txt").read_text() == "Content of file 1"

        # Verify temp extraction is tracked
        assert extract_path in zip_extractor._temp_extractions

    def test_extract_to_specified_destination(
        self,
        zip_extractor: ZIPExtractor,
        sample_zip_file: Path,
    ) -> None:
        """Test extracting to a specific destination directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "extract"
            extract_path = zip_extractor.extract_archive(
                archive_path=str(sample_zip_file),
                destination=dest,
            )

            assert extract_path == dest
            assert (extract_path / "file1.txt").exists()

    def test_extract_with_overwrite(
        self,
        zip_extractor: ZIPExtractor,
        sample_zip_file: Path,
    ) -> None:
        """Test extracting with overwrite flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "extract"
            dest.mkdir(parents=True, exist_ok=True)

            # First extraction
            zip_extractor.extract_archive(
                archive_path=str(sample_zip_file),
                destination=dest,
            )

            # Modify a file
            (dest / "file1.txt").write_text("Modified content")

            # Second extraction without overwrite should skip existing file
            zip_extractor.extract_archive(
                archive_path=str(sample_zip_file),
                destination=dest,
                overwrite=False,
            )
            assert (dest / "file1.txt").read_text() == "Modified content"

            # Second extraction with overwrite should replace file
            zip_extractor.extract_archive(
                archive_path=str(sample_zip_file),
                destination=dest,
                overwrite=True,
            )
            assert (dest / "file1.txt").read_text() == "Content of file 1"

    def test_extract_nonexistent_file(
        self,
        zip_extractor: ZIPExtractor,
    ) -> None:
        """Test extraction of non-existent file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            zip_extractor.extract_archive(archive_path="/nonexistent.zip")

    def test_extract_invalid_zip(
        self,
        zip_extractor: ZIPExtractor,
        tmp_path: Path,
    ) -> None:
        """Test extraction of invalid ZIP file."""
        # Create a file that's not a valid ZIP
        invalid_zip = tmp_path / "invalid.zip"
        invalid_zip.write_text("Not a valid ZIP file")

        with pytest.raises(InvalidArchiveError, match="Invalid or corrupted"):
            zip_extractor.extract_archive(archive_path=str(invalid_zip))

    def test_extract_directory_instead_of_file(
        self,
        zip_extractor: ZIPExtractor,
        tmp_path: Path,
    ) -> None:
        """Test extraction when path is a directory."""
        with pytest.raises(InvalidArchiveError, match="not a file"):
            zip_extractor.extract_archive(archive_path=str(tmp_path))

    def test_extract_path_traversal_simple(
        self,
        zip_extractor: ZIPExtractor,
        path_traversal_zip_file: Path,
    ) -> None:
        """Test extraction rejects simple path traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "extract"
            with pytest.raises(PathTraversalError, match="Path traversal"):
                zip_extractor.extract_archive(
                    archive_path=str(path_traversal_zip_file),
                    destination=dest,
                )

    def test_extract_path_traversal_nested(
        self,
        zip_extractor: ZIPExtractor,
        nested_path_traversal_zip_file: Path,
    ) -> None:
        """Test extraction rejects nested path traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "extract"
            with pytest.raises(PathTraversalError, match="Path traversal"):
                zip_extractor.extract_archive(
                    archive_path=str(nested_path_traversal_zip_file),
                    destination=dest,
                )


class TestZIPExtractorMetadata:
    """Tests for ZIPExtractor metadata extraction."""

    def test_extract_metadata_success(
        self,
        zip_extractor: ZIPExtractor,
        large_zip_file: Path,
    ) -> None:
        """Test successful metadata extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_path = Path(tmpdir) / "extract"
            extract_path.mkdir()

            metadata = zip_extractor.extract_metadata(
                archive_path=str(large_zip_file),
                extract_path=extract_path,
            )

            assert isinstance(metadata, ArchiveMetadata)
            assert metadata.archive_path == str(large_zip_file)
            assert metadata.extract_path == extract_path
            assert metadata.total_files == 5
            assert metadata.total_directories == 2

    def test_extract_metadata_nonexistent_file(
        self,
        zip_extractor: ZIPExtractor,
    ) -> None:
        """Test metadata extraction of non-existent file."""
        with pytest.raises(FileNotFoundError):
            zip_extractor.extract_metadata(
                archive_path="/nonexistent.zip",
                extract_path=Path("/tmp"),
            )

    def test_extract_metadata_invalid_zip(
        self,
        zip_extractor: ZIPExtractor,
        tmp_path: Path,
    ) -> None:
        """Test metadata extraction of invalid ZIP file."""
        invalid_zip = tmp_path / "invalid.zip"
        invalid_zip.write_text("Not a valid ZIP file")

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_path = Path(tmpdir) / "extract"
            extract_path.mkdir()

            with pytest.raises(InvalidArchiveError, match="Invalid or corrupted"):
                zip_extractor.extract_metadata(
                    archive_path=str(invalid_zip),
                    extract_path=extract_path,
                )

    def test_extract_metadata_archive_size(
        self,
        zip_extractor: ZIPExtractor,
        sample_zip_file: Path,
    ) -> None:
        """Test that archive size is correctly extracted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_path = Path(tmpdir) / "extract"
            extract_path.mkdir()

            metadata = zip_extractor.extract_metadata(
                archive_path=str(sample_zip_file),
                extract_path=extract_path,
            )

            assert metadata.archive_size == sample_zip_file.stat().st_size


class TestZIPExtractorCleanup:
    """Tests for ZIPExtractor cleanup functionality."""

    @patch("shutil.rmtree")
    @patch.object(Path, "exists", return_value=True)
    def test_cleanup_extraction_success(
        self,
        mock_exists: Any,
        mock_rmtree: Any,
        zip_extractor: ZIPExtractor,
    ) -> None:
        """Test successful cleanup of extraction directory."""
        extract_path = Path("/tmp/test-extract")
        result = zip_extractor.cleanup_extraction(extract_path)

        assert result is True
        mock_rmtree.assert_called_once_with(extract_path)

    @patch.object(Path, "exists", return_value=False)
    def test_cleanup_extraction_nonexistent(
        self,
        mock_exists: Any,
        zip_extractor: ZIPExtractor,
    ) -> None:
        """Test cleanup of non-existent path."""
        extract_path = Path("/tmp/test-extract")
        result = zip_extractor.cleanup_extraction(extract_path)

        assert result is False

    @patch("shutil.rmtree")
    @patch.object(Path, "exists", return_value=True)
    def test_cleanup_extraction_with_temp_list(
        self,
        mock_exists: Any,
        mock_rmtree: Any,
        zip_extractor: ZIPExtractor,
    ) -> None:
        """Test cleanup removes path from temp_extractions list."""
        extract_path = Path("/tmp/test-extract")
        zip_extractor._temp_extractions.append(extract_path)

        zip_extractor.cleanup_extraction(extract_path)

        assert extract_path not in zip_extractor._temp_extractions

    @patch("shutil.rmtree")
    @patch.object(Path, "exists", return_value=True)
    def test_cleanup_all_temp_extractions(
        self,
        mock_exists: Any,
        mock_rmtree: Any,
        zip_extractor: ZIPExtractor,
    ) -> None:
        """Test cleanup of all temporary extractions."""
        zip_extractor._temp_extractions = [
            Path("/tmp/extract1"),
            Path("/tmp/extract2"),
            Path("/tmp/extract3"),
        ]

        cleaned_count = zip_extractor.cleanup_all_temp_extractions()

        assert cleaned_count == 3
        assert len(zip_extractor._temp_extractions) == 0
        assert mock_rmtree.call_count == 3

    @patch("shutil.rmtree")
    @patch.object(Path, "exists", return_value=True)
    def test_context_manager_cleanup(
        self,
        mock_exists: Any,
        mock_rmtree: Any,
        zip_extractor: ZIPExtractor,
        sample_zip_file: Path,
    ) -> None:
        """Test context manager cleans up on exit."""
        extract_path = Path("/tmp/test-extract")
        zip_extractor._temp_extractions.append(extract_path)

        with zip_extractor:
            pass

        assert len(zip_extractor._temp_extractions) == 0
        mock_rmtree.assert_called_once()


class TestZIPExtractorSecurity:
    """Tests for ZIPExtractor security features."""

    def test_path_traversal_via_absolute_path(
        self,
        zip_extractor: ZIPExtractor,
        tmp_path: Path,
    ) -> None:
        """Test that absolute paths are rejected as path traversal."""
        zip_path = tmp_path / "absolute.zip"
        with zipfile.ZipFile(zip_path, mode="w") as zf:
            # Try to use absolute path
            zf.writestr("/tmp/evil.txt", "Evil content")

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "extract"
            # Absolute paths should be rejected as path traversal
            with pytest.raises(PathTraversalError, match="Path traversal"):
                zip_extractor.extract_archive(
                    archive_path=str(zip_path),
                    destination=dest,
                )

    def test_path_traversal_via_backslash(
        self,
        zip_extractor: ZIPExtractor,
        tmp_path: Path,
    ) -> None:
        """Test that backslash traversal is rejected."""
        zip_path = tmp_path / "backslash.zip"
        with zipfile.ZipFile(zip_path, mode="w") as zf:
            zf.writestr("..\\evil.txt", "Evil content")

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "extract"
            with pytest.raises(PathTraversalError, match="Path traversal"):
                zip_extractor.extract_archive(
                    archive_path=str(zip_path),
                    destination=dest,
                )
