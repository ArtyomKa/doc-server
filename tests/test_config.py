"""
Tests for doc_server configuration management.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from doc_server.config import Settings


class TestSettings:
    """Test cases for Settings class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = Settings()

        assert settings.storage_path == Path.home() / ".doc-server"
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.embedding_device == "cpu"
        assert settings.embedding_batch_size == 32
        assert settings.default_top_k == 10
        assert settings.search_min_score == 0.5
        assert settings.keyword_boost == 2.0
        assert settings.max_file_size == 1048576
        assert settings.log_level == "INFO"
        assert settings.mcp_transport == "stdio"
        assert settings.mcp_debug is False

    def test_storage_path_properties(self):
        """Test that storage path properties are computed correctly."""
        settings = Settings(storage_path=Path("/test/path"))

        assert settings.chroma_db_path == Path("/test/path/chroma.db")
        assert settings.models_path == Path("/test/path/models")
        assert settings.libraries_path == Path("/test/path/libraries")
        assert settings.config_file == Path("/test/path/config.yaml")
        assert settings.vector_db_path == Path("/test/path/chroma.db")  # alias

    def test_allowed_extensions_default(self):
        """Test default allowed extensions list includes all expected file types."""
        settings = Settings()
        expected_extensions = {
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
        }

        actual_extensions = set(settings.allowed_extensions)
        assert actual_extensions == expected_extensions
        assert len(settings.allowed_extensions) == 16

    @pytest.mark.parametrize(
        "env_var,value,expected_attr",
        [
            ("DOC_SERVER_STORAGE_PATH", "/custom/path", "storage_path"),
            ("DOC_SERVER_EMBEDDING_MODEL", "custom-model", "embedding_model"),
            ("DOC_SERVER_EMBEDDING_DEVICE", "cuda", "embedding_device"),
            ("DOC_SERVER_EMBEDDING_BATCH_SIZE", "64", "embedding_batch_size"),
            ("DOC_SERVER_DEFAULT_TOP_K", "20", "default_top_k"),
            ("DOC_SERVER_SEARCH_MIN_SCORE", "0.8", "search_min_score"),
            ("DOC_SERVER_KEYWORD_BOOST", "3.0", "keyword_boost"),
            ("DOC_SERVER_MAX_FILE_SIZE", "2097152", "max_file_size"),
            ("DOC_SERVER_LOG_LEVEL", "DEBUG", "log_level"),
            ("DOC_SERVER_MCP_TRANSPORT", "http", "mcp_transport"),
            ("DOC_SERVER_MCP_DEBUG", "true", "mcp_debug"),
        ],
    )
    def test_environment_variable_loading(self, env_var, value, expected_attr):
        """Test that environment variables are loaded correctly."""
        with patch.dict(os.environ, {env_var: str(value)}):
            settings = Settings()
            actual_value = getattr(settings, expected_attr)

            # Handle type conversions
            if expected_attr == "storage_path":
                assert actual_value == Path(value)
            elif expected_attr in [
                "embedding_batch_size",
                "default_top_k",
                "max_file_size",
            ]:
                assert actual_value == int(value)
            elif expected_attr in ["search_min_score", "keyword_boost"]:
                assert actual_value == float(value)
            elif expected_attr == "mcp_debug":
                assert actual_value is True
            else:
                assert actual_value == value

    def test_normalize_library_id_adds_slash(self):
        """Test that normalize_library_id adds leading slash if missing."""
        settings = Settings()

        assert settings.normalize_library_id("pandas") == "/pandas"
        assert settings.normalize_library_id("pandas/v2.2") == "/pandas/v2.2"

    def test_normalize_library_id_preserves_slash(self):
        """Test that normalize_library_id preserves existing leading slash."""
        settings = Settings()

        assert settings.normalize_library_id("/pandas") == "/pandas"
        assert settings.normalize_library_id("/pandas/v2.2") == "/pandas/v2.2"

    def test_normalize_library_id_cleans_multiple_slashes(self):
        """Test that normalize_library_id cleans consecutive slashes."""
        settings = Settings()

        assert settings.normalize_library_id("//pandas") == "/pandas"
        assert settings.normalize_library_id("///pandas//v2.2") == "/pandas/v2.2"

    def test_normalize_library_id_validates_characters(self):
        """Test that normalize_library_id validates allowed characters."""
        settings = Settings()

        # Valid characters
        assert settings.normalize_library_id("pandas-2023") == "/pandas-2023"
        assert settings.normalize_library_id("pandas_2") == "/pandas_2"
        assert settings.normalize_library_id("pandas/v2.2") == "/pandas/v2.2"

        # Invalid characters
        with pytest.raises(ValueError, match="Invalid library ID"):
            settings.normalize_library_id("pandas@2.2")

        with pytest.raises(ValueError, match="Invalid library ID"):
            settings.normalize_library_id("pandas#2.2")

        with pytest.raises(ValueError, match="Invalid library ID"):
            settings.normalize_library_id("pandas 2.2")

    def test_normalize_library_id_rejects_empty(self):
        """Test that normalize_library_id rejects empty string."""
        settings = Settings()

        with pytest.raises(ValueError, match="Library ID cannot be empty"):
            settings.normalize_library_id("")

        with pytest.raises(ValueError, match="Library ID cannot be empty"):
            settings.normalize_library_id("   ")

    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
embedding_model: "custom-model"
embedding_batch_size: 64
default_top_k: 15
log_level: "DEBUG"
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"
            config_file.write_text(yaml_content)

            settings = Settings(storage_path=Path(temp_dir))
            settings.load_yaml_config()

            assert settings.embedding_model == "custom-model"
            assert settings.embedding_batch_size == 64
            assert settings.default_top_k == 15
            assert settings.log_level == "DEBUG"

    def test_yaml_config_loading_no_file(self):
        """Test that missing YAML file doesn't cause errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(storage_path=Path(temp_dir))

            # Should not raise an exception
            settings.load_yaml_config()

            # Should still have default values
            assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_yaml_config_loading_invalid_yaml(self):
        """Test handling of invalid YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"
            config_file.write_text("invalid: yaml: content: [")

            settings = Settings(storage_path=Path(temp_dir))

            # Should not raise an exception, just warn
            settings.load_yaml_config()

            # Should still have default values
            assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_save_yaml_config(self):
        """Test saving configuration to YAML file."""
        # Create settings with custom values
        settings = Settings(
            embedding_model="custom-model",
            embedding_batch_size=64,
            default_top_k=15,
            storage_path=Path(tempfile.mkdtemp()),
        )

        settings.save_yaml_config()

        # Verify file was created and contains expected content
        assert settings.config_file.exists()
        content = settings.config_file.read_text()

        # Should contain only non-default values
        assert "custom-model" in content
        assert "64" in content
        assert "15" in content
        assert "all-MiniLM-L6-v2" not in content  # default value

    def test_save_yaml_config_no_pyyaml(self):
        """Test that missing PyYAML raises informative error."""
        settings = Settings()

        with patch.dict("sys.modules", {"yaml": None}):
            with pytest.raises(ImportError, match="PyYAML is required"):
                settings.save_yaml_config()

    def test_path_object_handling(self):
        """Test that Path objects are handled correctly in config operations."""
        custom_path = Path("/custom/storage")
        settings = Settings(storage_path=custom_path)

        # Properties should return Path objects
        assert isinstance(settings.chroma_db_path, Path)
        assert isinstance(settings.models_path, Path)
        assert isinstance(settings.libraries_path, Path)
        assert isinstance(settings.config_file, Path)
