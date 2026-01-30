"""
Git repository cloning functionality with shallow clone support.
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from git import GitCommandError, InvalidGitRepositoryError, Repo

from doc_server.config import Settings

logger = structlog.get_logger()


@dataclass
class RepositoryMetadata:
    """Metadata extracted from a git repository."""

    clone_url: str
    clone_path: Path
    commit_hash: str
    commit_message: str
    branch: str
    remote_url: str
    author: str
    timestamp: str
    is_shallow: bool


class GitCloneError(Exception):
    """Raised when git cloning fails."""

    def __init__(self, message: str, clone_url: str) -> None:
        """Initialize error with context."""
        self.clone_url = clone_url
        super().__init__(message)


class InvalidURLError(GitCloneError):
    """Raised when a git URL is invalid."""

    pass


class GitCloner:
    """
    Git repository cloner with shallow clone support and metadata extraction.

    Supports both HTTP/HTTPS and SSH git URLs. Provides utilities for cleanup
    and metadata extraction from cloned repositories.
    """

    def __init__(self, config: Settings) -> None:
        """
        Initialize the GitCloner.

        Args:
            config: Application settings instance
        """
        self._config = config
        self._temp_clones: list[Path] = []

    def clone_repository(
        self,
        clone_url: str,
        destination: Path | None = None,
        branch: str | None = None,
        shallow: bool = True,
        depth: int = 1,
    ) -> Repo:
        """
        Clone a git repository with optional shallow clone.

        Args:
            clone_url: URL of the git repository to clone
            destination: Path where repository should be cloned. If None,
                        creates a temporary directory that needs to be cleaned up.
            branch: Specific branch to clone. If None, clones default branch.
            shallow: Whether to perform a shallow clone (no history)
            depth: Depth of shallow clone. 1 means only latest commit.

        Returns:
            Repo: GitPython Repo object for the cloned repository

        Raises:
            InvalidURLError: If the clone URL is invalid
            GitCloneError: If cloning fails for any reason
        """
        self._validate_url(clone_url)

        try:
            if destination is None:
                destination = Path(tempfile.mkdtemp(prefix="doc-server-clone-"))
                self._temp_clones.append(destination)
                logger.info(
                    "Created temporary clone directory",
                    clone_url=clone_url,
                    path=str(destination),
                )

            # Prepare clone arguments
            clone_args: dict[str, Any] = {
                "url": clone_url,
                "to_path": destination,
            }

            if shallow:
                clone_args["depth"] = depth
                logger.debug(
                    "Performing shallow clone",
                    clone_url=clone_url,
                    depth=depth,
                )

            if branch:
                clone_args["branch"] = branch
                logger.debug(
                    "Cloning specific branch",
                    clone_url=clone_url,
                    branch=branch,
                )

            # Perform the clone
            repo = Repo.clone_from(**clone_args)

            logger.info(
                "Successfully cloned repository",
                clone_url=clone_url,
                path=str(destination),
                branch=branch or "default",
                shallow=shallow,
            )

            return repo

        except GitCommandError as exc:
            error_msg = self._parse_git_error(exc)
            raise GitCloneError(
                f"Failed to clone repository: {error_msg}",
                clone_url=clone_url,
            ) from exc
        except Exception as exc:
            raise GitCloneError(
                f"Unexpected error cloning repository: {exc}",
                clone_url=clone_url,
            ) from exc

    def extract_metadata(self, repo: Repo, clone_url: str) -> RepositoryMetadata:
        """
        Extract metadata from a cloned repository.

        Args:
            repo: GitPython Repo object
            clone_url: Original URL used for cloning

        Returns:
            RepositoryMetadata: Metadata about the repository

        Raises:
            GitCloneError: If metadata extraction fails
        """
        try:
            # Get the current commit
            commit = repo.head.commit

            # Check if this is a shallow repository
            is_shallow = self._is_shallow_repo(repo)

            # Get branch name
            try:
                branch = repo.active_branch.name
            except TypeError:
                # Detached HEAD state (common in shallow clones)
                branch = "detached"

            # Get remote URL
            try:
                remote_url = (
                    list(repo.remotes[0].urls)[0] if repo.remotes else clone_url
                )
            except (IndexError, AttributeError, StopIteration):
                remote_url = clone_url

            # Format timestamp
            timestamp = commit.committed_datetime.isoformat()

            # Handle commit message (might be bytes or str)
            message = commit.message
            if isinstance(message, bytes):
                message = message.decode("utf-8", errors="ignore")
            message = str(message).strip().splitlines()[0]

            metadata = RepositoryMetadata(
                clone_url=clone_url,
                clone_path=Path(repo.working_dir),
                commit_hash=commit.hexsha[:8],
                commit_message=message,
                branch=branch,
                remote_url=remote_url,
                author=f"{commit.author.name} <{commit.author.email}>",
                timestamp=timestamp,
                is_shallow=is_shallow,
            )

            logger.info(
                "Extracted repository metadata",
                clone_url=clone_url,
                commit_hash=metadata.commit_hash,
                branch=metadata.branch,
            )

            return metadata

        except (InvalidGitRepositoryError, ValueError, AttributeError) as exc:
            raise GitCloneError(
                f"Failed to extract repository metadata: {exc}",
                clone_url=clone_url,
            ) from exc

    def cleanup_clone(self, clone_path: Path) -> bool:
        """
        Remove a cloned repository directory.

        Args:
            clone_path: Path to the cloned repository

        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            if clone_path.exists():
                shutil.rmtree(clone_path)
                if clone_path in self._temp_clones:
                    self._temp_clones.remove(clone_path)
                logger.info("Cleaned up clone directory", path=str(clone_path))
                return True
            return False
        except OSError as exc:
            logger.error(
                "Failed to cleanup clone directory",
                path=str(clone_path),
                error=str(exc),
            )
            return False

    def cleanup_all_temp_clones(self) -> int:
        """
        Clean up all temporary clone directories created by this instance.

        Returns:
            int: Number of clones cleaned up
        """
        cleaned_count = 0
        for clone_path in list(self._temp_clones):
            if self.cleanup_clone(clone_path):
                cleaned_count += 1
        return cleaned_count

    def _validate_url(self, clone_url: str) -> None:
        """
        Validate a git repository URL.

        Args:
            clone_url: URL to validate

        Raises:
            InvalidURLError: If the URL is invalid
        """
        if not clone_url or not isinstance(clone_url, str):
            raise InvalidURLError("Clone URL must be a non-empty string", clone_url)

        clone_url = clone_url.strip()

        if not clone_url:
            raise InvalidURLError("Clone URL cannot be empty", clone_url)

        # Check for supported URL patterns
        valid_prefixes = [
            "http://",
            "https://",
            "git://",
            "ssh://",
            "git@",
        ]

        if not any(clone_url.startswith(prefix) for prefix in valid_prefixes):
            raise InvalidURLError(
                f"Invalid git URL format. Must start with one of: {', '.join(valid_prefixes)}",
                clone_url,
            )

        # Check for common invalid patterns
        if ".git" not in clone_url and not any(
            clone_url.startswith(prefix) for prefix in ["git@", "git://"]
        ):
            # Warn but don't fail - some git servers support URLs without .git extension
            logger.warning(
                "URL does not end with .git - this might not be a git repository",
                clone_url=clone_url,
            )

    def _is_shallow_repo(self, repo: Repo) -> bool:
        """
        Check if a repository is a shallow clone.

        Args:
            repo: GitPython Repo object

        Returns:
            bool: True if repository is shallow
        """
        shallow_file = Path(repo.git_dir) / "shallow"
        return shallow_file.exists()

    def _parse_git_error(self, error: GitCommandError) -> str:
        """
        Parse a git command error to extract a user-friendly message.

        Args:
            error: GitCommandError instance

        Returns:
            str: User-friendly error message
        """
        stderr = error.stderr.strip() if error.stderr else ""
        stdout = error.stdout.strip() if error.stdout else ""

        # Common error patterns
        if "does not appear to be a git repository" in stderr.lower():
            return "URL is not a valid git repository"
        elif "could not read username" in stderr.lower():
            return "Authentication required for this repository"
        elif "permission denied" in stderr.lower():
            return "Permission denied - check your credentials"
        elif "repository" in stderr.lower() and "not found" in stderr.lower():
            return "Repository not found at the specified URL"
        elif "connection refused" in stderr.lower():
            return "Could not connect to the git server"
        elif "ssl" in stderr.lower() or "certificate" in stderr.lower():
            return "SSL/TLS certificate error"
        elif "unknown host" in stderr.lower():
            return "Unknown host - check your network connection"

        # Fall back to the raw error message
        return stderr or stdout or str(error)

    def __enter__(self) -> "GitCloner":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup temp clones."""
        self.cleanup_all_temp_clones()
