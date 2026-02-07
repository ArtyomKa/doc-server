"""
Tests for embedding service initialization error handling.
"""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from doc_server.search.embedding_service import EmbeddingService


class TestInitializationErrorHandling:
    """Test error handling during service initialization."""

    def test_initialization_retry_success(self):
        """Test that initialization retries on network errors."""
        with patch(
            "doc_server.search.embedding_service.SentenceTransformer"
        ) as mock_cls:
            # First attempt fails with timeout, second succeeds
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_cls.side_effect = [Exception("Connection timeout"), mock_instance]

            with tempfile.TemporaryDirectory() as temp_dir:
                # Mock time.sleep to speed up test
                with patch("time.sleep"):
                    service = EmbeddingService(cache_dir=temp_dir, enable_cache=False)

                assert service.model is not None
                assert mock_cls.call_count == 2

    def test_initialization_retry_failure(self):
        """Test that initialization fails after retries exhausted."""
        with patch(
            "doc_server.search.embedding_service.SentenceTransformer"
        ) as mock_cls:
            # All attempts fail
            mock_cls.side_effect = Exception("Connection timeout")

            with tempfile.TemporaryDirectory() as temp_dir:
                # Mock time.sleep to speed up test
                with patch("time.sleep"):
                    with pytest.raises(Exception) as exc_info:
                        EmbeddingService(cache_dir=temp_dir, enable_cache=False)

                assert "Connection timeout" in str(exc_info.value)
                assert mock_cls.call_count == 3  # Default max_load_retries

    def test_initialization_non_network_error(self):
        """Test that initialization does not retry on non-network errors."""
        with patch(
            "doc_server.search.embedding_service.SentenceTransformer"
        ) as mock_cls:
            # Fails with ValueError
            mock_cls.side_effect = ValueError("Invalid model")

            with tempfile.TemporaryDirectory() as temp_dir:
                with pytest.raises(ValueError) as exc_info:
                    EmbeddingService(cache_dir=temp_dir, enable_cache=False)

                assert "Invalid model" in str(exc_info.value)
                assert mock_cls.call_count == 1
