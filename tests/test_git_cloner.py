"""
Unit tests for GitCloner module.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import pytest

from doc_server.config import Settings
from doc_server.ingestion.git_cloner import (
    GitCloneError,
    GitCloner,
    InvalidURLError,
    RepositoryMetadata,
)


@pytest.fixture
def config() -> Settings:
    """Create a Settings instance for testing."""
    return Settings()


@pytest.fixture
def git_cloner(config: Settings) -> GitCloner:
    """Create a GitCloner instance for testing."""
    return GitCloner(config)


@pytest.fixture
def mock_repo() -> Mock:
    """Create a mock GitPython Repo object."""
    repo = Mock()
    repo.working_dir = "/tmp/test-repo"
    repo.git_dir = "/tmp/test-repo/.git"

    # Mock commit
    commit = Mock()
    commit.hexsha = "a1b2c3d4e5f6g7h8i9j0"
    commit.message = "Initial commit\n\nThis is a test commit."
    commit.author.name = "Test Author"
    commit.author.email = "test@example.com"
    commit.committed_datetime.isoformat.return_value = "2024-01-01T12:00:00"
    repo.head.commit = commit

    # Mock branch
    repo.active_branch.name = "main"

    # Mock remotes
    remote = Mock()
    remote.urls = ["https://github.com/test/repo.git"]
    repo.remotes = [remote]

    with patch.object(Path, "exists", return_value=False):
        return repo


class TestRepositoryMetadata:
    """Tests for RepositoryMetadata dataclass."""

    def test_repository_metadata_creation(self) -> None:
        """Test creating RepositoryMetadata instance."""
        metadata = RepositoryMetadata(
            clone_url="https://github.com/test/repo.git",
            clone_path=Path("/tmp/repo"),
            commit_hash="a1b2c3d4",
            commit_message="Test commit",
            branch="main",
            remote_url="https://github.com/test/repo.git",
            author="Test Author <test@example.com>",
            timestamp="2024-01-01T12:00:00",
            is_shallow=False,
        )

        assert metadata.clone_url == "https://github.com/test/repo.git"
        assert metadata.commit_hash == "a1b2c3d4"
        assert metadata.branch == "main"
        assert metadata.is_shallow is False


class TestGitClonerValidation:
    """Tests for GitCloner URL validation."""

    def test_validate_url_valid_https(self, git_cloner: GitCloner) -> None:
        """Test validation of valid HTTPS URL."""
        # Should not raise
        git_cloner._validate_url("https://github.com/test/repo.git")

    def test_validate_url_valid_http(self, git_cloner: GitCloner) -> None:
        """Test validation of valid HTTP URL."""
        git_cloner._validate_url("http://github.com/test/repo.git")

    def test_validate_url_valid_git_protocol(self, git_cloner: GitCloner) -> None:
        """Test validation of valid git:// URL."""
        git_cloner._validate_url("git://github.com/test/repo.git")

    def test_validate_url_valid_ssh(self, git_cloner: GitCloner) -> None:
        """Test validation of valid SSH URL."""
        git_cloner._validate_url("ssh://git@github.com/test/repo.git")

    def test_validate_url_valid_git_at(self, git_cloner: GitCloner) -> None:
        """Test validation of valid git@ URL."""
        git_cloner._validate_url("git@github.com:test/repo.git")

    def test_validate_url_empty_string(self, git_cloner: GitCloner) -> None:
        """Test validation rejects empty string."""
        with pytest.raises(InvalidURLError, match="non-empty"):
            git_cloner._validate_url("")

    def test_validate_url_none(self, git_cloner: GitCloner) -> None:
        """Test validation rejects None."""
        with pytest.raises(InvalidURLError, match="must be a non-empty string"):
            git_cloner._validate_url(None)  # type: ignore

    def test_validate_url_whitespace_only(self, git_cloner: GitCloner) -> None:
        """Test validation rejects whitespace-only string."""
        with pytest.raises(InvalidURLError, match="cannot be empty"):
            git_cloner._validate_url("   ")

    def test_validate_url_invalid_prefix(self, git_cloner: GitCloner) -> None:
        """Test validation rejects URL without valid prefix."""
        with pytest.raises(InvalidURLError, match="Invalid git URL format"):
            git_cloner._validate_url("ftp://github.com/test/repo.git")

    def test_validate_url_missing_git_extension(self, git_cloner: GitCloner) -> None:
        """Test validation warns but accepts URL without .git extension."""
        # Should not raise, but would log a warning
        git_cloner._validate_url("https://github.com/test/repo")


class TestGitClonerClone:
    """Tests for GitCloner clone functionality."""

    @patch("doc_server.ingestion.git_cloner.Repo.clone_from")
    def test_clone_repository_with_destination(
        self,
        mock_clone_from: Mock,
        git_cloner: GitCloner,
        mock_repo: Mock,
    ) -> None:
        """Test cloning to a specific destination."""
        mock_clone_from.return_value = mock_repo

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "repo"
            result = git_cloner.clone_repository(
                clone_url="https://github.com/test/repo.git",
                destination=dest,
            )

            mock_clone_from.assert_called_once()
            assert result == mock_repo
            call_kwargs = mock_clone_from.call_args[1]
            assert call_kwargs["url"] == "https://github.com/test/repo.git"
            assert call_kwargs["to_path"] == dest
            assert call_kwargs["depth"] == 1

    @patch("doc_server.ingestion.git_cloner.Repo.clone_from")
    @patch("doc_server.ingestion.git_cloner.tempfile.mkdtemp")
    def test_clone_repository_without_destination(
        self,
        mock_mkdtemp: Mock,
        mock_clone_from: Mock,
        git_cloner: GitCloner,
        mock_repo: Mock,
    ) -> None:
        """Test cloning to temporary directory."""
        mock_mkdtemp.return_value = "/tmp/doc-server-clone-xyz"
        mock_clone_from.return_value = mock_repo

        result = git_cloner.clone_repository(
            clone_url="https://github.com/test/repo.git",
        )

        assert result == mock_repo
        mock_mkdtemp.assert_called_once_with(prefix="doc-server-clone-")
        mock_clone_from.assert_called_once()
        assert len(git_cloner._temp_clones) == 1

    @patch("doc_server.ingestion.git_cloner.Repo.clone_from")
    def test_clone_repository_with_branch(
        self,
        mock_clone_from: Mock,
        git_cloner: GitCloner,
        mock_repo: Mock,
    ) -> None:
        """Test cloning specific branch."""
        mock_clone_from.return_value = mock_repo

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "repo"
            git_cloner.clone_repository(
                clone_url="https://github.com/test/repo.git",
                destination=dest,
                branch="develop",
            )

            call_kwargs = mock_clone_from.call_args[1]
            assert call_kwargs["branch"] == "develop"

    @patch("doc_server.ingestion.git_cloner.Repo.clone_from")
    def test_clone_repository_full_depth(
        self,
        mock_clone_from: Mock,
        git_cloner: GitCloner,
        mock_repo: Mock,
    ) -> None:
        """Test cloning with full history (not shallow)."""
        mock_clone_from.return_value = mock_repo

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "repo"
            git_cloner.clone_repository(
                clone_url="https://github.com/test/repo.git",
                destination=dest,
                shallow=False,
            )

            call_kwargs = mock_clone_from.call_args[1]
            assert "depth" not in call_kwargs

    @patch("doc_server.ingestion.git_cloner.Repo.clone_from")
    def test_clone_repository_invalid_url(
        self,
        mock_clone_from: Mock,
        git_cloner: GitCloner,
    ) -> None:
        """Test clone with invalid URL."""
        with pytest.raises(InvalidURLError):
            git_cloner.clone_repository(
                clone_url="ftp://invalid.com",
                destination=Path("/tmp/repo"),
            )

    @patch("doc_server.ingestion.git_cloner.Repo.clone_from")
    def test_clone_repository_git_error(
        self,
        mock_clone_from: Mock,
        git_cloner: GitCloner,
    ) -> None:
        """Test clone when git command fails."""
        from git.exc import GitCommandError

        mock_clone_from.side_effect = GitCommandError(
            "clone", ["https://github.com/test/repo.git"], stderr="repository not found"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "repo"
            with pytest.raises(GitCloneError, match="Failed to clone"):
                git_cloner.clone_repository(
                    clone_url="https://github.com/test/repo.git",
                    destination=dest,
                )


class TestGitClonerMetadata:
    """Tests for GitCloner metadata extraction."""

    def test_extract_metadata_success(
        self,
        git_cloner: GitCloner,
        mock_repo: Mock,
    ) -> None:
        """Test successful metadata extraction."""
        metadata = git_cloner.extract_metadata(
            repo=mock_repo,
            clone_url="https://github.com/test/repo.git",
        )

        assert isinstance(metadata, RepositoryMetadata)
        assert metadata.clone_url == "https://github.com/test/repo.git"
        assert metadata.commit_hash == "a1b2c3d4"
        assert metadata.commit_message == "Initial commit"
        assert metadata.branch == "main"
        assert metadata.author == "Test Author <test@example.com>"
        assert metadata.timestamp == "2024-01-01T12:00:00"
        assert metadata.is_shallow is False

    def test_extract_metadata_detached_head(
        self,
        git_cloner: GitCloner,
    ) -> None:
        """Test metadata extraction with detached HEAD."""
        repo = Mock()
        repo.working_dir = "/tmp/test-repo"
        repo.git_dir = "/tmp/test-repo/.git"
        repo.head.commit = Mock(
            hexsha="a1b2c3d4e5f6g7h8i9j0",
            message="Test commit",
            author=Mock(name="Test Author", email="test@example.com"),
        )
        # Simulate detached HEAD by making active_branch.name raise TypeError
        type(repo.active_branch).name = PropertyMock(side_effect=TypeError("detached"))
        repo.remotes = [Mock(urls=["https://github.com/test/repo.git"])]

        with patch.object(Path, "exists", return_value=False):
            metadata = git_cloner.extract_metadata(
                repo=repo,
                clone_url="https://github.com/test/repo.git",
            )

        assert metadata.branch == "detached"


class TestGitClonerCleanup:
    """Tests for GitCloner cleanup functionality."""

    @patch("doc_server.ingestion.git_cloner.shutil.rmtree")
    @patch.object(Path, "exists", return_value=True)
    def test_cleanup_clone_success(
        self,
        mock_exists: Mock,
        mock_rmtree: Mock,
        git_cloner: GitCloner,
    ) -> None:
        """Test successful cleanup of clone directory."""
        clone_path = Path("/tmp/test-clone")
        result = git_cloner.cleanup_clone(clone_path)

        assert result is True
        mock_rmtree.assert_called_once_with(clone_path)

    @patch.object(Path, "exists", return_value=False)
    def test_cleanup_clone_nonexistent(
        self,
        mock_exists: Mock,
        git_cloner: GitCloner,
    ) -> None:
        """Test cleanup of non-existent path."""
        clone_path = Path("/tmp/test-clone")
        result = git_cloner.cleanup_clone(clone_path)

        assert result is False

    @patch("doc_server.ingestion.git_cloner.shutil.rmtree")
    @patch.object(Path, "exists", return_value=True)
    def test_cleanup_clone_with_temp_list(
        self,
        mock_exists: Mock,
        mock_rmtree: Mock,
        git_cloner: GitCloner,
    ) -> None:
        """Test cleanup removes path from temp_clones list."""
        clone_path = Path("/tmp/test-clone")
        git_cloner._temp_clones.append(clone_path)

        git_cloner.cleanup_clone(clone_path)

        assert clone_path not in git_cloner._temp_clones

    @patch("doc_server.ingestion.git_cloner.shutil.rmtree")
    @patch.object(Path, "exists", return_value=True)
    def test_cleanup_all_temp_clones(
        self,
        mock_exists: Mock,
        mock_rmtree: Mock,
        git_cloner: GitCloner,
    ) -> None:
        """Test cleanup of all temporary clones."""
        git_cloner._temp_clones = [
            Path("/tmp/clone1"),
            Path("/tmp/clone2"),
            Path("/tmp/clone3"),
        ]

        cleaned_count = git_cloner.cleanup_all_temp_clones()

        assert cleaned_count == 3
        assert len(git_cloner._temp_clones) == 0
        assert mock_rmtree.call_count == 3

    @patch("doc_server.ingestion.git_cloner.shutil.rmtree")
    @patch.object(Path, "exists", return_value=True)
    def test_context_manager_cleanup(
        self,
        mock_exists: Mock,
        mock_rmtree: Mock,
        git_cloner: GitCloner,
    ) -> None:
        """Test context manager cleans up on exit."""
        git_cloner._temp_clones = [Path("/tmp/clone1")]

        with git_cloner:
            pass

        assert len(git_cloner._temp_clones) == 0
        mock_rmtree.assert_called_once()


class TestGitClonerErrorParsing:
    """Tests for GitCloner error message parsing."""

    def test_parse_git_error_repo_not_found(self, git_cloner: GitCloner) -> None:
        """Test parsing 'repository not found' error."""
        from git.exc import GitCommandError

        error = GitCommandError(
            "clone",
            ["https://github.com/invalid/repo.git"],
            stderr="fatal: repository 'https://github.com/invalid/repo.git' not found",
        )

        message = git_cloner._parse_git_error(error)

        assert "Repository not found" in message

    def test_parse_git_error_permission_denied(self, git_cloner: GitCloner) -> None:
        """Test parsing 'permission denied' error."""
        from git.exc import GitCommandError

        error = GitCommandError(
            "clone",
            ["https://github.com/test/repo.git"],
            stderr="fatal: permission denied",
        )

        message = git_cloner._parse_git_error(error)

        assert "Permission denied" in message

    def test_parse_git_error_connection_refused(self, git_cloner: GitCloner) -> None:
        """Test parsing 'connection refused' error."""
        from git.exc import GitCommandError

        error = GitCommandError(
            "clone",
            ["https://github.com/test/repo.git"],
            stderr="fatal: could not connect to host, connection refused",
        )

        message = git_cloner._parse_git_error(error)

        assert "Could not connect" in message

    def test_parse_git_error_ssl_error(self, git_cloner: GitCloner) -> None:
        """Test parsing SSL certificate error."""
        from git.exc import GitCommandError

        error = GitCommandError(
            "clone",
            ["https://github.com/test/repo.git"],
            stderr="SSL certificate problem",
        )

        message = git_cloner._parse_git_error(error)

        assert "SSL/TLS certificate" in message

    def test_parse_git_error_unknown_host(self, git_cloner: GitCloner) -> None:
        """Test parsing unknown host error."""
        from git.exc import GitCommandError

        error = GitCommandError(
            "clone",
            ["https://unknown-host.com/repo.git"],
            stderr="fatal: unknown host: unknown-host.com",
        )

        message = git_cloner._parse_git_error(error)

        assert "Unknown host" in message

    def test_parse_git_error_fallback(self, git_cloner: GitCloner) -> None:
        """Test fallback to raw error message."""
        from git.exc import GitCommandError

        error = GitCommandError(
            "clone",
            ["https://github.com/test/repo.git"],
            stderr="some unknown error",
        )

        message = git_cloner._parse_git_error(error)

        assert "some unknown error" in message
