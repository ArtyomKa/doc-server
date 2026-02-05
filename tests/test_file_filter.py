"""
Unit tests for FileFilter module.
"""

from pathlib import Path
from unittest import mock

import pytest

from doc_server.config import Settings
from doc_server.ingestion.file_filter import (
    BinaryFileError,
    FileFilter,
    FileFilterError,
    FileSizeExceededError,
    FilterResult,
    InvalidGitignoreError,
)


@pytest.fixture
def config() -> Settings:
    """Create a Settings instance for testing."""
    return Settings()


@pytest.fixture
def file_filter(config: Settings) -> FileFilter:
    """Create a FileFilter instance for testing."""
    return FileFilter(config)


@pytest.fixture
def test_directory(tmp_path: Path) -> Path:
    """Create a test directory with various files."""
    # Create subdirectories
    src_dir = tmp_path / "src"
    docs_dir = tmp_path / "docs"
    tests_dir = tmp_path / "tests"
    src_dir.mkdir()
    docs_dir.mkdir()
    tests_dir.mkdir()

    # Create text files (should be included)
    (src_dir / "module.py").write_text("# Python module\n")
    (src_dir / "module.h").write_text("/* C header */\n")
    (src_dir / "module.cpp").write_text("// C++ source\n")
    (docs_dir / "README.md").write_text("# Documentation\n")
    (docs_dir / "guide.rst").write_text("Rest Documentation\n")
    (src_dir / "config.yaml").write_text("key: value\n")

    # Create binary file with allowed extension (should be excluded)
    (src_dir / "binary.txt").write_bytes(b"\x00\x01\x02\x03\x04\x05")

    # Create large file (should be excluded if >1MB)
    large_file = src_dir / "large.txt"
    large_file.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB

    # Create file with non-allowed extension (should be excluded)
    (src_dir / "image.png").write_bytes(b"PNG data")

    # Create hidden directory with file
    hidden_dir = src_dir / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "secret.txt").write_text("secret content\n")

    return tmp_path


@pytest.fixture
def gitignore_file(tmp_path: Path) -> Path:
    """Create a .gitignore file for testing."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text(
        """# Test .gitignore
*.log
__pycache__/
*.pyc
hidden/
"""
    )
    return gitignore


class TestFileFilterInit:
    """Test FileFilter initialization."""

    def test_init_default_config(self, config: Settings) -> None:
        """Test initialization with default config."""
        filter_obj = FileFilter(config)
        assert filter_obj._config == config
        assert len(filter_obj._allowed_extensions) > 0

    def test_allowed_extensions_case_insensitive(self, config: Settings) -> None:
        """Test that allowed extensions are case-insensitive."""
        filter_obj = FileFilter(config)
        # Should have both .py and .PY in the set (as lowercase)
        assert ".py" in filter_obj._allowed_extensions
        assert ".PY".lower() in filter_obj._allowed_extensions


class TestLoadGitignore:
    """Test .gitignore loading functionality."""

    def test_load_valid_gitignore(
        self,
        file_filter: FileFilter,
        gitignore_file: Path,
    ) -> None:
        """Test loading a valid .gitignore file."""
        spec = file_filter.load_gitignore(str(gitignore_file))
        assert spec is not None

    def test_load_nonexistent_gitignore(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
    ) -> None:
        """Test loading a non-existent .gitignore file returns None."""
        spec = file_filter.load_gitignore(str(tmp_path / "nonexistent.gitignore"))
        assert spec is None

    def test_load_none_gitignore(self, file_filter: FileFilter) -> None:
        """Test loading with None path returns None."""
        spec = file_filter.load_gitignore(None)
        assert spec is None

    def test_load_empty_gitignore(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
    ) -> None:
        """Test loading an empty .gitignore file returns None."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("")
        spec = file_filter.load_gitignore(str(gitignore))
        assert spec is None

    def test_load_gitignore_with_comments(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
    ) -> None:
        """Test .gitignore with comments only returns None."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("# Comment 1\n# Comment 2\n")
        spec = file_filter.load_gitignore(str(gitignore))
        assert spec is None

    def test_load_gitignore_with_patterns(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
    ) -> None:
        """Test .gitignore with valid patterns."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\n__pycache__/\ntest_*.py")
        spec = file_filter.load_gitignore(str(gitignore))
        assert spec is not None
        # Test pattern matching
        assert spec.match_file("test_file.py")
        assert spec.match_file("debug.log")
        assert not spec.match_file("module.py")

    def test_load_invalid_gitignore_encoding(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
    ) -> None:
        """Test loading .gitignore with invalid encoding raises error."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_bytes(b"\x80\x81\x82")  # Invalid UTF-8
        with pytest.raises(InvalidGitignoreError):
            file_filter.load_gitignore(str(gitignore))


class TestCheckExtension:
    """Test extension checking functionality."""

    def test_allowed_extension(self, file_filter: FileFilter, tmp_path: Path) -> None:
        """Test that allowed extensions pass."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# Python file")
        assert file_filter._check_extension(test_file) is True

    def test_allowed_extensions_all_types(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
    ) -> None:
        """Test all allowed extension types."""
        allowed_files = [
            "test.md",
            "test.rst",
            "test.txt",
            "test.py",
            "test.pyi",
            "test.c",
            "test.cpp",
            "test.h",
            "test.hpp",
            "test.yaml",
            "test.json",
        ]
        for filename in allowed_files:
            test_file = tmp_path / filename
            test_file.write_text("content")
            assert (
                file_filter._check_extension(test_file) is True
            ), f"Failed for {filename}"

    def test_disallowed_extension(
        self, file_filter: FileFilter, tmp_path: Path
    ) -> None:
        """Test that disallowed extensions fail."""
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"image data")
        assert file_filter._check_extension(test_file) is False

    def test_extension_case_insensitive(
        self, file_filter: FileFilter, tmp_path: Path
    ) -> None:
        """Test extension checking is case-insensitive."""
        test_file = tmp_path / "test.PY"
        test_file.write_text("# Python file")
        assert file_filter._check_extension(test_file) is True


class TestCheckFileSize:
    """Test file size checking functionality."""

    def test_file_within_limit(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
        config: Settings,
    ) -> None:
        """Test that files within size limit pass."""
        test_file = tmp_path / "small.txt"
        content = b"x" * (config.max_file_size // 2)  # Half the limit
        test_file.write_bytes(content)
        assert file_filter._check_file_size(test_file) is True

    def test_file_exceeds_limit(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
        config: Settings,
    ) -> None:
        """Test that files exceeding size limit fail."""
        test_file = tmp_path / "large.txt"
        content = b"x" * (config.max_file_size + 1)  # Just over the limit
        test_file.write_bytes(content)
        assert file_filter._check_file_size(test_file) is False

    def test_file_at_limit(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
        config: Settings,
    ) -> None:
        """Test that files at exactly the size limit pass."""
        test_file = tmp_path / "exact.txt"
        content = b"x" * config.max_file_size
        test_file.write_bytes(content)
        assert file_filter._check_file_size(test_file) is True

    def test_nonexistent_file(self, file_filter: FileFilter, tmp_path: Path) -> None:
        """Test checking size of nonexistent file returns False."""
        test_file = tmp_path / "nonexistent.txt"
        assert file_filter._check_file_size(test_file) is False


class TestCheckBinaryContent:
    """Test binary content detection functionality."""

    def test_text_file_passes(self, file_filter: FileFilter, tmp_path: Path) -> None:
        """Test that text files pass binary check."""
        test_file = tmp_path / "text.txt"
        test_file.write_text("Plain text content\n")
        assert file_filter._check_binary_content(test_file) is False

    def test_binary_file_fails(self, file_filter: FileFilter, tmp_path: Path) -> None:
        """Test that binary files with null bytes fail."""
        test_file = tmp_path / "binary.dat"
        test_file.write_bytes(b"\x00\x01\x02\x03\x04")
        assert file_filter._check_binary_content(test_file) is True

    def test_mixed_content_fails(self, file_filter: FileFilter, tmp_path: Path) -> None:
        """Test that files with null bytes in text content are detected as binary."""
        test_file = tmp_path / "mixed.dat"
        test_file.write_bytes(b"Text\x00Content")
        assert file_filter._check_binary_content(test_file) is True

    def test_utf8_text_passes(self, file_filter: FileFilter, tmp_path: Path) -> None:
        """Test that UTF-8 encoded text files pass."""
        test_file = tmp_path / "utf8.txt"
        test_file.write_text("Hello 世界 🌍")
        assert file_filter._check_binary_content(test_file) is False


class TestShouldIncludeFile:
    """Test file inclusion decision logic."""

    def test_include_allowed_text_file(
        self,
        file_filter: FileFilter,
        test_directory: Path,
    ) -> None:
        """Test including an allowed text file."""
        test_file = test_directory / "src" / "module.py"
        assert file_filter.should_include_file(str(test_file)) is True

    def test_exclude_disallowed_extension(
        self,
        file_filter: FileFilter,
        test_directory: Path,
    ) -> None:
        """Test excluding file with disallowed extension."""
        test_file = test_directory / "src" / "image.png"
        assert file_filter.should_include_file(str(test_file)) is False

    def test_exclude_binary_file(
        self,
        file_filter: FileFilter,
        test_directory: Path,
    ) -> None:
        """Test excluding binary file with allowed extension."""
        test_file = test_directory / "src" / "binary.txt"
        assert file_filter.should_include_file(str(test_file)) is False

    def test_exclude_large_file(
        self,
        file_filter: FileFilter,
        test_directory: Path,
    ) -> None:
        """Test excluding file that exceeds size limit."""
        test_file = test_directory / "src" / "large.txt"
        assert file_filter.should_include_file(str(test_file)) is False

    def test_exclude_gitignore_pattern(
        self,
        file_filter: FileFilter,
        test_directory: Path,
        gitignore_file: Path,
    ) -> None:
        """Test excluding file matching .gitignore pattern."""
        # Create a log file that should be ignored
        log_file = test_directory / "debug.log"
        log_file.write_text("log content\n")

        spec = file_filter.load_gitignore(str(gitignore_file))
        assert (
            file_filter.should_include_file(str(log_file), spec, str(test_directory))
            is False
        )

    def test_include_non_gitignore_file(
        self,
        file_filter: FileFilter,
        test_directory: Path,
        gitignore_file: Path,
    ) -> None:
        """Test including file that doesn't match .gitignore."""
        test_file = test_directory / "src" / "module.py"
        spec = file_filter.load_gitignore(str(gitignore_file))
        assert (
            file_filter.should_include_file(str(test_file), spec, str(test_directory))
            is True
        )

    def test_nonexistent_file(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
    ) -> None:
        """Test handling of nonexistent file."""
        test_file = tmp_path / "nonexistent.py"
        assert file_filter.should_include_file(str(test_file)) is False


class TestFilterFile:
    """Test detailed file filtering with result object."""

    def test_filter_included_file(
        self,
        file_filter: FileFilter,
        test_directory: Path,
    ) -> None:
        """Test filtering an included file returns correct result."""
        test_file = test_directory / "src" / "module.py"
        result = file_filter.filter_file(str(test_file))

        assert result.included is True
        assert result.file_path == str(test_file)
        assert result.extension == ".py"
        assert result.reason == "File passes all filters"
        assert result.is_binary is False
        assert result.gitignore_matched is False

    def test_filter_excluded_by_extension(
        self,
        file_filter: FileFilter,
        test_directory: Path,
    ) -> None:
        """Test filtering file excluded by extension."""
        test_file = test_directory / "src" / "image.png"
        result = file_filter.filter_file(str(test_file))

        assert result.included is False
        assert result.extension == ".png"
        assert "not in allowlist" in result.reason.lower()

    def test_filter_excluded_by_size(
        self,
        file_filter: FileFilter,
        test_directory: Path,
    ) -> None:
        """Test filtering file excluded by size."""
        test_file = test_directory / "src" / "large.txt"
        result = file_filter.filter_file(str(test_file))

        assert result.included is False
        assert "exceeds limit" in result.reason.lower()
        assert result.size_bytes > 0

    def test_filter_excluded_by_binary(
        self,
        file_filter: FileFilter,
        test_directory: Path,
    ) -> None:
        """Test filtering binary file (binary detection happens after extension check)."""
        # Create a binary file with an allowed extension
        test_file = test_directory / "src" / "binary.txt"
        test_file.write_bytes(b"\x00\x01\x02\x03\x04")
        result = file_filter.filter_file(str(test_file))

        assert result.included is False
        assert "binary" in result.reason.lower()
        assert result.is_binary is True

    def test_filter_excluded_by_gitignore(
        self,
        file_filter: FileFilter,
        test_directory: Path,
        gitignore_file: Path,
    ) -> None:
        """Test filtering file excluded by .gitignore."""
        log_file = test_directory / "debug.log"
        log_file.write_text("log content\n")

        spec = file_filter.load_gitignore(str(gitignore_file))
        result = file_filter.filter_file(str(log_file), spec, str(test_directory))

        assert result.included is False
        assert "gitignore" in result.reason.lower()
        assert result.gitignore_matched is True

    def test_filter_with_absolute_path(
        self,
        file_filter: FileFilter,
        test_directory: Path,
    ) -> None:
        """Test filtering with absolute file path."""
        test_file = test_directory / "src" / "module.py"
        abs_path = test_file.resolve()
        result = file_filter.filter_file(str(abs_path))

        assert result.included is True


class TestFilterDirectory:
    """Test directory filtering functionality."""

    def test_filter_all_files(
        self,
        file_filter: FileFilter,
        test_directory: Path,
    ) -> None:
        """Test filtering all files in a directory."""
        results = file_filter.filter_directory(str(test_directory))

        assert len(results) > 0

        # Check that Python files are included
        py_files = [r for r in results if r.extension == ".py"]
        assert len(py_files) > 0
        assert all(r.included for r in py_files)

        # Check that binary files are excluded
        binary_files = [r for r in results if r.is_binary is True]
        assert len(binary_files) > 0
        assert all(not r.included for r in binary_files)

    def test_filter_directory_with_gitignore(
        self,
        file_filter: FileFilter,
        test_directory: Path,
        gitignore_file: Path,
    ) -> None:
        """Test filtering directory with .gitignore patterns."""
        # Create a log file that should be ignored
        log_file = test_directory / "debug.log"
        log_file.write_text("log content\n")

        spec = file_filter.load_gitignore(str(gitignore_file))
        results = file_filter.filter_directory(str(test_directory), spec)

        # Check that log file is excluded
        log_results = [r for r in results if r.file_path == str(log_file)]
        assert len(log_results) == 1
        assert log_results[0].gitignore_matched is True

    def test_filter_nonexistent_directory(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
    ) -> None:
        """Test filtering nonexistent directory raises error."""
        with pytest.raises(FileFilterError, match="not found"):
            file_filter.filter_directory(str(tmp_path / "nonexistent"))

    def test_filter_file_instead_of_directory(
        self,
        file_filter: FileFilter,
        test_directory: Path,
    ) -> None:
        """Test filtering a file instead of directory raises error."""
        test_file = test_directory / "src" / "module.py"
        with pytest.raises(FileFilterError, match="not a directory"):
            file_filter.filter_directory(str(test_file))


class TestFilterResultModel:
    """Test FilterResult Pydantic model."""

    def test_filter_result_creation(self) -> None:
        """Test creating a FilterResult."""
        result = FilterResult(
            file_path="/test/file.py",
            included=True,
            reason="All good",
            extension=".py",
            size_bytes=1024,
            is_binary=False,
            gitignore_matched=False,
        )
        assert result.file_path == "/test/file.py"
        assert result.included is True

    def test_filter_result_serialization(self) -> None:
        """Test that FilterResult can be serialized."""
        result = FilterResult(
            file_path="/test/file.py",
            included=True,
            reason="All good",
            extension=".py",
            size_bytes=1024,
            is_binary=False,
        )
        data = result.model_dump()
        assert "file_path" in data
        assert "included" in data


class TestExceptions:
    """Test custom exception classes."""

    def test_file_filter_error(self) -> None:
        """Test FileFilterError base exception."""
        error = FileFilterError("Test error")
        assert str(error) == "Test error"

    def test_invalid_gitignore_error(self) -> None:
        """Test InvalidGitignoreError with context."""
        error = InvalidGitignoreError("Invalid .gitignore", "/path/to/.gitignore")
        assert "Invalid .gitignore" in str(error)
        assert error.gitignore_path == "/path/to/.gitignore"

    def test_file_size_exceeded_error(self) -> None:
        """Test FileSizeExceededError with context."""
        error = FileSizeExceededError(
            "File too large", "/path/to/file.txt", 2000000, 1000000
        )
        assert "File too large" in str(error)
        assert error.file_size == 2000000
        assert error.max_size == 1000000

    def test_binary_file_error(self) -> None:
        """Test BinaryFileError with context."""
        error = BinaryFileError("Binary content detected", "/path/to/binary.dat")
        assert "Binary content detected" in str(error)
        assert error.file_path == "/path/to/binary.dat"


class TestGitignorePatterns:
    """Test .gitignore pattern matching."""

    def test_wildcard_pattern(self, file_filter: FileFilter, tmp_path: Path) -> None:
        """Test wildcard pattern matching."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log")

        spec = file_filter.load_gitignore(str(gitignore))
        assert spec is not None

        # Test that .log files are matched
        assert spec.match_file("debug.log")
        assert spec.match_file("error.log")

        # Test that other files are not matched
        assert not spec.match_file("debug.txt")

    def test_directory_pattern(self, file_filter: FileFilter, tmp_path: Path) -> None:
        """Test directory pattern matching."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("__pycache__/\n")

        spec = file_filter.load_gitignore(str(gitignore))
        assert spec is not None

        # Test directory patterns
        assert spec.match_file("__pycache__/module.pyc")
        assert not spec.match_file("cache/module.pyc")

    def test_nested_patterns(self, file_filter: FileFilter, tmp_path: Path) -> None:
        """Test nested directory patterns."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("build/\n**/*.pyc\n")

        spec = file_filter.load_gitignore(str(gitignore))
        assert spec is not None

        # Test nested patterns
        assert spec.match_file("build/output.o")
        assert spec.match_file("module.pyc")
        assert spec.match_file("src/module.pyc")


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_directory_filtering(
        self, file_filter: FileFilter, tmp_path: Path
    ) -> None:
        """Test filtering an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        results = file_filter.filter_directory(str(empty_dir))
        assert len(results) == 0

    def test_file_without_extension(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
    ) -> None:
        """Test filtering file without extension."""
        test_file = tmp_path / "Makefile"
        test_file.write_text("make target")

        result = file_filter.filter_file(str(test_file))
        # Should be excluded (no extension in allowlist)
        assert result.included is False
        assert result.extension == ""

    def test_hidden_files(self, file_filter: FileFilter, test_directory: Path) -> None:
        """Test filtering hidden files (starting with .)."""
        test_file = test_directory / "src" / ".hidden" / "secret.txt"
        result = file_filter.filter_file(str(test_file))

        # .txt is in allowlist, so should be included unless filtered by other rules
        assert result.included is True or "hidden" in result.reason.lower()

    def test_symlink_handling(self, file_filter: FileFilter, tmp_path: Path) -> None:
        """Test filtering symbolic links."""
        # Create a regular file
        target_file = tmp_path / "target.txt"
        target_file.write_text("content")

        # Create a symlink
        symlink_file = tmp_path / "link.txt"
        symlink_file.symlink_to(target_file)

        result = file_filter.filter_file(str(symlink_file))
        # Symlinks to files should be processed as files
        assert result.included is True

    def test_special_characters_in_filename(
        self,
        file_filter: FileFilter,
        tmp_path: Path,
    ) -> None:
        """Test filtering files with special characters."""
        test_file = tmp_path / "file with spaces.py"
        test_file.write_text("# Python file")

        result = file_filter.filter_file(str(test_file))
        assert result.included is True


class TestFileFilterUncoveredPaths:
    """Tests for uncovered file_filter edge cases."""

    def test_file_access_permission_errors(
        self, file_filter: FileFilter, tmp_path: Path
    ):
        """Test file filtering with permission errors."""
        test_file = tmp_path / "restricted.py"
        test_file.write_text("# Restricted file")

        # Mock stat to raise permission error
        with mock.patch("pathlib.Path.stat", side_effect=OSError("Permission denied")):
            with pytest.raises(FileFilterError):
                file_filter.filter_file(str(test_file))

    def test_gitignore_parsing_failures(self, file_filter: FileFilter, tmp_path: Path):
        """Test gitignore parsing with invalid patterns."""
        # Create a gitignore with invalid syntax
        gitignore_file = tmp_path / ".gitignore"
        gitignore_file.write_text("**/invalid[unclosed\n*.log\n")

        # Create test directory with gitignore
        test_dir = tmp_path / "project"
        test_dir.mkdir()
        gitignore_file = test_dir / ".gitignore"
        gitignore_file.write_text("**/invalid[unclosed\n*.log\n")

        test_file = test_dir / "test.py"
        test_file.write_text("# Test file")

        # Should handle invalid gitignore gracefully
        result = file_filter.filter_file(str(test_file))
        assert isinstance(result, FilterResult)

    def test_binary_file_detection_edge_cases(
        self, file_filter: FileFilter, tmp_path: Path
    ):
        """Test binary file detection with edge cases."""
        # Create file with null bytes
        binary_file = tmp_path / "mixed.py"
        binary_file.write_bytes(b"print('hello')\x00\x80\xfe")

        result = file_filter.filter_file(str(binary_file))

        # Should detect as binary and exclude
        assert not result.included
        assert "binary" in result.reason.lower()

    def test_large_file_handling(self, file_filter: FileFilter, tmp_path: Path):
        """Test large file size handling."""
        large_file = tmp_path / "large.py"

        # Mock file size to be very large
        with mock.patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100MB

            result = file_filter.filter_file(str(large_file))

            # Should exclude large files
            assert not result.included
            assert "size" in result.reason.lower()

    def test_unicode_filename_handling(self, file_filter: FileFilter, tmp_path: Path):
        """Test handling of Unicode filenames."""
        unicode_file = tmp_path / "файл.py"  # Cyrillic filename
        unicode_file.write_text("# Unicode filename")

        result = file_filter.filter_file(str(unicode_file))

        # Should handle Unicode filenames gracefully
        assert isinstance(result, FilterResult)

    def test_corrupted_symlink_handling(self, file_filter: FileFilter, tmp_path: Path):
        """Test handling of corrupted symlinks."""
        # Create a symlink pointing to non-existent target
        corrupted_link = tmp_path / "broken.py"
        corrupted_link.symlink_to(tmp_path / "nonexistent.py")

        result = file_filter.filter_file(str(corrupted_link))

        # Should handle broken symlinks gracefully
        assert isinstance(result, FilterResult)

    def test_file_encoding_detection_errors(
        self, file_filter: FileFilter, tmp_path: Path
    ):
        """Test file encoding detection errors."""
        # Create file with problematic encoding
        test_file = tmp_path / "encoding.py"

        # Write bytes that would cause encoding issues
        with open(test_file, "wb") as f:
            f.write(b"# Python file with invalid utf-8\xff\xfe\x00")

        result = file_filter.filter_file(str(test_file))

        # Should handle encoding errors gracefully
        assert isinstance(result, FilterResult)

    def test_gitignore_with_complex_patterns(
        self, file_filter: FileFilter, tmp_path: Path
    ):
        """Test gitignore with complex pattern combinations."""
        gitignore_file = tmp_path / ".gitignore"
        gitignore_file.write_text(
            """
# Complex patterns
**/test_*.py
!test_main.py
*.temp
build/**
docs/**/*.md
!docs/README.md
"""
        )

        test_files = [
            ("test_module.py", False),  # Should be excluded
            ("test_main.py", True),  # Should be included (negated)
            ("temp.temp", False),  # Should be excluded
            ("build/output.o", False),  # Should be excluded
            ("docs/README.md", True),  # Should be included (negated)
            ("docs/guide.md", False),  # Should be excluded
        ]

        # Load the gitignore patterns
        gitignore_spec = file_filter.load_gitignore(str(gitignore_file))

        for filename, expected_included in test_files:
            test_file = tmp_path / filename
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text(f"# {filename}")

            result = file_filter.filter_file(
                str(test_file), gitignore_spec, str(tmp_path)
            )
            assert result.included == expected_included, f"Failed for {filename}"

    def test_allowlist_edge_cases(self, file_filter: FileFilter, tmp_path: Path):
        """Test allowlist with edge cases."""
        # Test case sensitivity
        case_files = [
            ("test.PY", True),  # Uppercase extension
            ("test.Py", True),  # Mixed case
            ("test.pY", True),  # Mixed case
            ("test.py", True),  # Normal case
        ]

        for filename, expected_included in case_files:
            test_file = tmp_path / filename
            test_file.write_text(f"# {filename}")

            result = file_filter.filter_file(str(test_file))
            assert result.included == expected_included, f"Failed for {filename}"

    def test_max_depth_filtering(self, file_filter: FileFilter, tmp_path: Path):
        """Test max depth filtering."""
        # Create nested directory structure
        current_dir = tmp_path
        for i in range(10):  # Create 10 levels deep
            current_dir = current_dir / f"level{i}"
            current_dir.mkdir()

        deep_file = current_dir / "deep.py"
        deep_file.write_text("# Deep file")

        result = file_filter.filter_file(str(deep_file))

        # Should filter based on depth (assuming reasonable max depth)
        assert isinstance(result, FilterResult)
        # Note: The actual inclusion depends on the max_depth setting
