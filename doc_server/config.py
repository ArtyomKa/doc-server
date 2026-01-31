"""
Configuration management for doc-server.
"""

import re
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support and YAML configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DOC_SERVER_",
        extra="ignore",
    )

    # Storage Paths
    storage_path: Path = Field(default=Path.home() / ".doc-server")

    # Embedding Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # Search Settings
    default_top_k: int = 10
    search_min_score: float = 0.5
    keyword_boost: float = 2.0

    # File Processing
    max_file_size: int = 1048576  # 1MB in bytes
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            # Code files
            ".py",
            ".pyi",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            # Documentation
            ".md",
            ".rst",
            ".txt",
            # Configuration files
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".cfg",
            ".ini",
            ".conf",
        ]
    )

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # MCP Settings
    mcp_transport: str = "stdio"
    mcp_debug: bool = False

    # Path Properties for storage directories
    @property
    def chroma_db_path(self) -> Path:
        """Path to ChromaDB storage directory."""
        return self.storage_path / "chroma.db"

    @property
    def models_path(self) -> Path:
        """Path to downloaded embedding models directory."""
        return self.storage_path / "models"

    @property
    def libraries_path(self) -> Path:
        """Path to ingested source data directory."""
        return self.storage_path / "libraries"

    @property
    def config_file(self) -> Path:
        """Path to configuration YAML file."""
        return self.storage_path / "config.yaml"

    @property
    def vector_db_path(self) -> Path:
        """Alias for chroma_db_path for backward compatibility."""
        return self.chroma_db_path

    def normalize_library_id(self, library_id: str) -> str:
        """
        Normalize library ID to ensure it starts with '/' and is valid.

        Args:
            library_id: Library identifier (e.g., 'pandas', '/pandas', 'pandas/v2.2')

        Returns:
            Normalized library ID (e.g., '/pandas', '/pandas/v2.2')

        Raises:
            ValueError: If library_id contains invalid characters
        """
        if not library_id or not library_id.strip():
            raise ValueError("Library ID cannot be empty")

        # Remove leading/trailing whitespace
        library_id = library_id.strip()

        # Ensure it starts with '/'
        if not library_id.startswith("/"):
            library_id = "/" + library_id

        # Validate characters: only alphanumeric, underscore, hyphen, forward slash, dot
        if not re.match(r"^/[a-zA-Z0-9_/.-]+$", library_id):
            raise ValueError(
                f"Invalid library ID '{library_id}'. "
                "Only letters, numbers, underscores, hyphens, forward slashes, and dots are allowed."
            )

        # Remove consecutive slashes
        library_id = re.sub(r"/+", "/", library_id)

        # Handle edge case of just "/"
        if library_id == "/":
            raise ValueError("Library ID cannot be empty")

        return library_id

    def load_yaml_config(self) -> None:
        """
        Load settings from config.yaml file if it exists.
        YAML settings override environment variables.
        """
        if not self.config_file.exists():
            return

        try:
            import yaml

            with open(self.config_file, encoding="utf-8") as f:
                yaml_settings = yaml.safe_load(f) or {}

            # Override env defaults with YAML values
            for key, value in yaml_settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        except ImportError:
            # YAML not available, skip YAML config loading
            pass
        except Exception as exc:
            # Log error but don't fail startup
            print(f"Warning: Failed to load YAML config: {exc}")

    def save_yaml_config(self) -> None:
        """
        Save current settings to config.yaml file.
        Only saves non-default values to keep config file clean.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to save configuration. "
                "Install with: pip install pyyaml"
            ) from exc

        # Get default settings for comparison
        default_settings = Settings()

        # Only save values that differ from defaults
        settings_to_save = {}
        for field_name, _field_info in self.__class__.model_fields.items():
            current_value = getattr(self, field_name)
            default_value = getattr(default_settings, field_name)
            if current_value != default_value:
                # Convert Path objects to strings for YAML
                if isinstance(current_value, Path):
                    settings_to_save[field_name] = str(current_value)
                else:
                    settings_to_save[field_name] = current_value

        # Ensure directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to YAML file
        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(settings_to_save, f, default_flow_style=False, indent=2)

    def __post_init__(self) -> None:
        """Initialize settings after creation."""
        # Load YAML configuration if available
        self.load_yaml_config()


# Global settings instance
settings = Settings()
