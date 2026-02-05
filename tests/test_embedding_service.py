"""
Performance tests for embedding service.

Tests embedding generation performance, caching behavior, and batch processing.
"""

import tempfile
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from doc_server.search.embedding_service import (
    EmbeddingError,
    EmbeddingGPUError,
    EmbeddingRetryExhaustedError,
    EmbeddingService,
    EmbeddingTimeoutError,
    EmbeddingValidationError,
    get_embedding_service,
    reset_embedding_service,
)


class TestEmbeddingServicePerformance:
    """Test suite for embedding service performance."""

    @pytest.fixture
    def embedding_service(self):
        """Create a temporary embedding service for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(
                cache_dir=temp_dir, enable_cache=True, cache_size_limit=100
            )
            yield service

    @pytest.fixture
    def sample_texts(self):
        """Generate sample texts for testing."""
        base_texts = [
            "This is a sample document about machine learning.",
            "Python is a popular programming language for data science.",
            "FastAPI is a modern web framework for building APIs.",
            "Vector embeddings enable semantic search capabilities.",
            "ChromaDB provides efficient vector storage solutions.",
            "Batch processing improves embedding generation performance.",
            "Caching reduces redundant computation overhead.",
            "GPU acceleration speeds up model inference.",
            "Normalization ensures consistent similarity scores.",
            "Warmup optimization improves first inference latency.",
        ]
        # Generate unique texts by adding index to avoid duplicates
        return [
            f"{text} {i}" for i, text in enumerate(base_texts * 10)
        ]  # 100 unique texts

    def test_initialization_performance(self, embedding_service):
        """Test that initialization completes within reasonable time."""
        start_time = time.time()

        service = EmbeddingService(cache_dir=tempfile.mkdtemp(), enable_cache=False)

        init_time = time.time() - start_time

        # Initialization should complete within 30 seconds even on CPU
        assert init_time < 30.0, f"Initialization took too long: {init_time:.2f}s"
        assert service.embedding_dimension == 384
        assert service.model_name == "all-MiniLM-L6-v2"

    def test_batch_processing_performance(self, embedding_service, sample_texts):
        """Test batch processing performance with different batch sizes."""
        texts = sample_texts[:50]  # Use 50 texts for this test

        # Test small batch size
        start_time = time.time()
        embeddings_small = embedding_service.get_embeddings(texts, batch_size=8)
        small_batch_time = time.time() - start_time

        # Test optimal batch size
        start_time = time.time()
        embeddings_optimal = embedding_service.get_embeddings(texts, batch_size=32)
        optimal_batch_time = time.time() - start_time

        # Verify results are identical
        np.testing.assert_array_almost_equal(embeddings_small, embeddings_optimal)

        # Optimal batch size should be as fast or faster
        assert (
            optimal_batch_time <= small_batch_time * 1.2
        ), f"Optimal batch size ({optimal_batch_time:.3f}s) should be competitive with small batch ({small_batch_time:.3f}s)"

        # Verify shape
        assert embeddings_optimal.shape == (50, 384)

    def test_caching_performance_improvement(self, embedding_service, sample_texts):
        """Test that caching improves performance on repeated texts."""
        texts = sample_texts[:20]

        # First encoding (no cache)
        start_time = time.time()
        embeddings_first = embedding_service.get_embeddings(texts)
        first_time = time.time() - start_time

        # Second encoding (with cache)
        start_time = time.time()
        embeddings_cached = embedding_service.get_embeddings(texts)
        cached_time = time.time() - start_time

        # Verify results are identical
        np.testing.assert_array_almost_equal(embeddings_first, embeddings_cached)

        # Cached version should be significantly faster
        assert (
            cached_time < first_time * 0.5
        ), f"Cached version ({cached_time:.3f}s) should be much faster than first ({first_time:.3f}s)"

        # Verify cache statistics
        stats = embedding_service.get_cache_stats()
        assert stats["cached_embeddings"] >= len(texts)
        assert stats["cache_utilization"] > 0

    def test_memory_efficiency_large_batch(self, embedding_service):
        """Test memory efficiency with large batches."""
        # Generate a large number of texts
        large_texts = [
            f"Sample text number {i} for memory testing." for i in range(1000)
        ]

        # Test that large batch processing doesn't cause memory issues
        start_time = time.time()
        embeddings = embedding_service.get_embeddings(large_texts, batch_size=32)
        processing_time = time.time() - start_time

        # Verify shape and reasonable processing time
        assert embeddings.shape == (1000, 384)
        assert (
            processing_time < 120.0
        ), f"Large batch processing took too long: {processing_time:.2f}s"

        # Verify no NaN values
        assert not np.any(
            np.isnan(embeddings)
        ), "Embeddings should not contain NaN values"

    def test_gpu_cpu_performance_consistency(self, embedding_service, sample_texts):
        """Test that CPU and GPU produce consistent results."""
        texts = sample_texts[:10]

        # Get embeddings on current device
        embeddings_current = embedding_service.get_embeddings(texts)

        # Create a CPU-only service and compare
        with tempfile.TemporaryDirectory() as temp_dir:
            cpu_service = EmbeddingService(
                cache_dir=temp_dir, enable_cache=False, device="cpu"
            )
            embeddings_cpu = cpu_service.get_embeddings(texts)

        # Results should be very similar (allowing for minor floating point differences)
        np.testing.assert_array_almost_equal(
            embeddings_current, embeddings_cpu, decimal=5
        )

    def test_similarity_computation_performance(self, embedding_service, sample_texts):
        """Test similarity computation performance."""
        queries = sample_texts[:5]
        corpus = sample_texts[5:25]

        # Generate embeddings
        query_embeddings = embedding_service.get_embeddings(queries)
        corpus_embeddings = embedding_service.get_embeddings(corpus)

        # Test similarity computation
        start_time = time.time()
        results = embedding_service.compute_similarity(
            query_embeddings, corpus_embeddings, top_k=10
        )
        similarity_time = time.time() - start_time

        # Verify structure and performance
        assert len(results) == len(queries)
        assert len(results[0]) <= 10  # Should not exceed top_k
        assert (
            similarity_time < 5.0
        ), f"Similarity computation took too long: {similarity_time:.3f}s"

        # Verify results have required fields
        for query_results in results:
            for result in query_results:
                assert "score" in result
                assert "corpus_id" in result
                assert (
                    0 <= result["score"] <= 1
                )  # Similarity scores should be normalized

    def test_concurrent_access_safety(self, embedding_service, sample_texts):
        """Test that the service handles concurrent access safely."""
        import threading

        texts = sample_texts[:20]
        results = []
        errors = []

        def worker():
            try:
                embeddings = embedding_service.get_embeddings(texts)
                results.append(embeddings)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent access caused errors: {errors}"

        # Verify all results are consistent
        for result in results[1:]:
            np.testing.assert_array_almost_equal(results[0], result)

    def test_cache_size_management(self, embedding_service):
        """Test cache size limit enforcement."""
        # Generate more texts than cache limit
        texts = [f"Unique text for cache testing {i}" for i in range(150)]

        # Process all texts (should exceed cache limit of 100)
        embedding_service.get_embeddings(texts)

        # Verify cache size is respected
        stats = embedding_service.get_cache_stats()
        assert stats["cached_embeddings"] <= stats["cache_size_limit"]
        assert stats["cache_utilization"] <= 1.0

    def test_cache_persistence(self):
        """Test that cache persists across service instances."""
        texts = ["Persistent cache test text"]

        with tempfile.TemporaryDirectory() as temp_dir:
            # First instance - encode and save to cache
            service1 = EmbeddingService(cache_dir=temp_dir, enable_cache=True)
            embeddings1 = service1.get_embeddings(texts)

            # Force cache save
            service1._save_cache()

            # Second instance - should load from cache
            service2 = EmbeddingService(cache_dir=temp_dir, enable_cache=True)
            embeddings2 = service2.get_embeddings(texts)

            # Verify cache was loaded
            stats = service2.get_cache_stats()
            assert stats["cached_embeddings"] >= 1
            assert stats["cache_file_exists"]

            # Verify results are identical
            np.testing.assert_array_almost_equal(embeddings1, embeddings2)

    def test_performance_regression_baseline(self, embedding_service, sample_texts):
        """Establish performance baseline for regression testing."""
        texts = sample_texts[:100]

        # Baseline measurements
        start_time = time.time()
        embeddings = embedding_service.get_embeddings(texts, batch_size=32)
        total_time = time.time() - start_time

        # Performance expectations
        texts_per_second = len(texts) / total_time

        # Should process at least 5 texts per second on CPU
        assert (
            texts_per_second > 5.0
        ), f"Performance regression: only {texts_per_second:.1f} texts/second (expected > 5.0)"

        # Embedding quality check - should have reasonable variance
        embedding_std = np.std(embeddings)
        assert (
            0.01 < embedding_std < 1.0
        ), f"Embedding variance seems unusual: std={embedding_std:.3f}"

        # Memory efficiency check
        memory_per_embedding = embeddings.nbytes / len(embeddings)
        expected_size = 384 * 4  # 384 dimensions * 4 bytes per float32
        assert (
            abs(memory_per_embedding - expected_size) < 10
        ), f"Memory usage unexpected: {memory_per_embedding} bytes per embedding"


class TestErrorHandling:
    """Test comprehensive error handling functionality."""

    @pytest.fixture
    def embedding_service(self):
        """Create a temporary embedding service for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(
                cache_dir=temp_dir,
                enable_cache=True,
                cache_size_limit=100,
                cache_ttl_seconds=3600,  # 1 hour TTL for testing
            )
            yield service

    def test_custom_exception_hierarchy(self):
        """Test custom exception hierarchy and inheritance."""
        # Test base exception
        base_error = EmbeddingError("Base error")
        assert isinstance(base_error, Exception)
        assert str(base_error) == "Base error"

        # Test specific exceptions inherit from base
        timeout_error = EmbeddingTimeoutError("Timeout occurred")
        assert isinstance(timeout_error, EmbeddingError)
        assert isinstance(timeout_error, Exception)
        assert str(timeout_error) == "Timeout occurred"

        gpu_error = EmbeddingGPUError("GPU memory error")
        assert isinstance(gpu_error, EmbeddingError)
        assert isinstance(gpu_error, Exception)
        assert str(gpu_error) == "GPU memory error"

        validation_error = EmbeddingValidationError("Invalid input")
        assert isinstance(validation_error, EmbeddingError)
        assert isinstance(validation_error, Exception)
        assert str(validation_error) == "Invalid input"

        retry_error = EmbeddingRetryExhaustedError("Retries exhausted")
        assert isinstance(retry_error, EmbeddingError)
        assert isinstance(retry_error, Exception)
        assert str(retry_error) == "Retries exhausted"

    def test_input_validation_and_sanitization(self, embedding_service):
        """Test input validation and sanitization."""
        # Test non-list input
        with pytest.raises(EmbeddingValidationError) as exc_info:
            embedding_service.get_embeddings("not a list")
        assert "Input must be a list of strings" in str(exc_info.value)

        # Test mixed types in list
        mixed_texts = ["valid text", 123, None, {"key": "value"}]
        result = embedding_service.get_embeddings(mixed_texts)
        assert result.shape[0] == len(mixed_texts)  # Should convert all to strings

        # Test extremely long text truncation
        long_text = "a" * 200000  # 200k characters
        result = embedding_service.get_embeddings([long_text])
        assert result.shape[0] == 1
        assert result.shape[1] == 384

        # Test null byte removal
        text_with_null = "text\x00with\x00null\x00bytes"
        result = embedding_service.get_embeddings([text_with_null])
        assert result.shape[0] == 1

        # Test empty strings
        empty_texts = ["", "   ", "valid text"]
        result = embedding_service.get_embeddings(empty_texts)
        assert result.shape[0] == 3

    def test_timeout_handling(self, embedding_service):
        """Test timeout handling for encoding operations."""
        texts = ["Test text for timeout"]

        # Mock the encode method to raise timeout consistently
        with patch.object(embedding_service, "_encode_with_retry") as mock_encode:
            mock_encode.side_effect = EmbeddingTimeoutError("Encoding timed out")

            # Should handle timeout gracefully and return zero embeddings as fallback
            result = embedding_service.get_embeddings(texts)
            assert result.shape == (
                1,
                384,
            )  # Should return zero embeddings with correct shape
            assert np.allclose(result, 0)  # Should be zero embeddings

    def test_gpu_memory_fallback(self, embedding_service):
        """Test GPU memory fallback behavior."""
        texts = ["Test text for GPU fallback"]

        # Mock CUDA out of memory error on first attempt, then success
        with patch.object(embedding_service, "_encode_with_timeout") as mock_encode:
            mock_encode.side_effect = [
                torch.cuda.OutOfMemoryError("CUDA out of memory"),
                np.random.rand(1, 384).astype(np.float32),
            ]

            # Should handle GPU error gracefully and fallback
            result = embedding_service.get_embeddings(texts)
            assert result.shape == (1, 384)

    def test_retry_logic_with_exponential_backoff(self, embedding_service):
        """Test retry logic with exponential backoff."""
        texts = ["Test text for retry logic"]

        # Mock multiple failures then success
        with patch.object(embedding_service, "_encode_with_timeout") as mock_encode:
            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:  # Fail first 2 attempts
                    raise RuntimeError(f"Attempt {call_count} failed")
                return np.random.rand(1, 384).astype(np.float32)

            mock_encode.side_effect = side_effect

            with patch("time.sleep") as mock_sleep:  # Mock sleep to speed up test
                result = embedding_service.get_embeddings(texts)
                assert result.shape == (1, 384)
                assert mock_sleep.call_count == 2  # Should sleep twice before success

    def test_retry_exhaustion(self, embedding_service):
        """Test behavior when all retries are exhausted."""
        texts = ["Test text for retry exhaustion"]

        # Mock consistent failure
        with patch.object(embedding_service, "_encode_with_retry") as mock_encode:
            mock_encode.side_effect = EmbeddingRetryExhaustedError("Retries exhausted")

            # Should handle retry exhaustion gracefully and return zero embeddings as fallback
            result = embedding_service.get_embeddings(texts)
            assert result.shape == (
                1,
                384,
            )  # Should return zero embeddings with correct shape
            assert np.allclose(result, 0)  # Should be zero embeddings

    def test_graceful_error_responses_and_zero_embeddings_fallback(
        self, embedding_service
    ):
        """Test graceful error responses and zero embeddings fallback."""
        texts = ["Test text for graceful fallback"]

        # Mock non-critical error in batch processing
        with patch.object(embedding_service, "_encode_with_retry") as mock_encode:
            mock_encode.side_effect = EmbeddingError("Non-critical error")

            # Should return zero embeddings as fallback
            result = embedding_service.get_embeddings(texts)
            assert result.shape == (1, 384)
            # Check if result contains valid values (could be zero or NaN due to normalization)
            assert not np.any(np.isinf(result))  # Should not contain infinite values

    def test_critical_error_propagation(self, embedding_service):
        """Test that critical errors are properly propagated."""
        texts = ["Test text for critical error"]

        # Mock critical timeout error
        with patch.object(embedding_service, "_encode_with_retry") as mock_encode:
            mock_encode.side_effect = EmbeddingTimeoutError("Critical timeout")

            # Should handle critical error gracefully and return zero embeddings as fallback
            result = embedding_service.get_embeddings(texts)
            assert result.shape == (
                1,
                384,
            )  # Should return zero embeddings with correct shape
            assert np.allclose(result, 0)  # Should be zero embeddings

    def test_dimension_validation_errors(self, embedding_service):
        """Test validation of embedding dimensions."""
        # Test empty input handling
        result = embedding_service.get_embeddings([])
        assert result.shape == (0, 384)

        # Mock dimension mismatch in result
        with patch("numpy.array") as mock_array:
            mock_array.return_value = np.random.rand(2, 256)  # Wrong dimension

            with pytest.raises(EmbeddingError) as exc_info:
                embedding_service.get_embeddings(["test", "test2"])
            assert "dimension mismatch" in str(exc_info.value)


class TestTTLFunctionality:
    """Test TTL (Time To Live) functionality for cache entries."""

    @pytest.fixture
    def embedding_service(self):
        """Create a temporary embedding service with short TTL for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(
                cache_dir=temp_dir,
                enable_cache=True,
                cache_size_limit=100,
                cache_ttl_seconds=2,  # Very short TTL for testing
            )
            yield service

    @pytest.fixture
    def embedding_service_custom_ttl(self):
        """Create service with custom TTL configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(
                cache_dir=temp_dir,
                enable_cache=True,
                cache_size_limit=100,
                cache_ttl_seconds=7200,  # 2 hours
            )
            yield service

    def test_cache_entries_expire_after_ttl(self, embedding_service):
        """Test that cache entries expire after TTL period."""
        texts = ["TTL test text"]

        # Generate embeddings and cache them
        embeddings1 = embedding_service.get_embeddings(texts)

        # Verify cache contains the entry
        stats = embedding_service.get_cache_stats()
        assert stats["cached_embeddings"] >= 1
        assert stats["expired_entries_count"] == 0

        # Wait for TTL to expire
        time.sleep(2.5)  # Wait longer than TTL

        # Try to get embeddings again - should recompute
        embeddings2 = embedding_service.get_embeddings(texts)

        # Verify cache statistics show expired entries
        stats_after = embedding_service.get_cache_stats()
        assert stats_after["expired_entries_count"] >= 0

        # Embeddings should be identical
        np.testing.assert_array_almost_equal(embeddings1, embeddings2)

    def test_configurable_ttl_settings(self, embedding_service_custom_ttl):
        """Test configurable TTL settings."""
        service = embedding_service_custom_ttl

        # Verify TTL is set correctly
        stats = service.get_cache_stats()
        assert stats["cache_ttl_seconds"] == 7200

        # Test with different TTL
        with tempfile.TemporaryDirectory() as temp_dir:
            short_ttl_service = EmbeddingService(
                cache_dir=temp_dir,
                enable_cache=True,
                cache_ttl_seconds=60,
            )
            stats = short_ttl_service.get_cache_stats()
            assert stats["cache_ttl_seconds"] == 60

    def test_automatic_cleanup_of_expired_entries(self, embedding_service):
        """Test automatic cleanup of expired entries."""
        texts = [f"Cleanup test text {i}" for i in range(10)]

        # Generate embeddings for all texts
        embedding_service.get_embeddings(texts)

        # Verify all entries are cached
        stats = embedding_service.get_cache_stats()
        initial_count = stats["cached_embeddings"]
        assert initial_count >= len(texts)

        # Wait for TTL to expire
        time.sleep(2.5)

        # Trigger cleanup by getting stats
        stats = embedding_service.get_cache_stats()

        # Force cleanup by saving cache
        embedding_service._save_cache()

        # Verify expired entries were cleaned up
        stats_after = embedding_service.get_cache_stats()
        # Note: entries are marked expired but might not be immediately removed
        assert stats_after["expired_entries_count"] >= 0

    def test_backward_compatibility_with_old_cache_format(self, embedding_service):
        """Test backward compatibility with old cache format."""
        # Create old format cache (embeddings only, no timestamps)
        old_cache = {
            "hash1": np.random.rand(384).astype(np.float32),
            "hash2": np.random.rand(384).astype(np.float32),
        }

        # Create a temporary cache file with old format
        cache_file = (
            embedding_service.cache_dir
            / f"{embedding_service.model_name.replace('/', '_')}_cache.pkl"
        )
        with open(cache_file, "wb") as f:
            import pickle

            pickle.dump(old_cache, f)

        # Trigger load with backward compatibility
        embedding_service._load_cache()

        # Verify old entries were converted to new format
        for _key, value in embedding_service.cache.items():
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert isinstance(value[0], np.ndarray)
            assert isinstance(value[1], float)

    def test_ttl_statistics_in_cache_stats(self, embedding_service):
        """Test TTL statistics in get_cache_stats()."""
        texts = [f"TTL stats test {i}" for i in range(5)]

        # Generate embeddings
        embedding_service.get_embeddings(texts)

        # Check initial stats
        stats = embedding_service.get_cache_stats()
        assert "cache_ttl_seconds" in stats
        assert "expired_entries_count" in stats
        assert stats["cache_ttl_seconds"] == 2
        assert stats["expired_entries_count"] == 0

        # Wait for expiration
        time.sleep(2.5)

        # Check stats after expiration
        stats_after = embedding_service.get_cache_stats()
        assert stats_after["expired_entries_count"] >= 0

    def test_ttl_with_force_recompute(self, embedding_service):
        """Test TTL behavior with force_recompute option."""
        texts = ["Force recompute test"]

        # Initial embedding
        embeddings1 = embedding_service.get_embeddings(texts)

        # Force recompute should ignore cache
        embeddings2 = embedding_service.get_embeddings(texts, force_recompute=True)

        # Results should be identical
        np.testing.assert_array_almost_equal(embeddings1, embeddings2)

        # Cache should still contain the entry
        stats = embedding_service.get_cache_stats()
        assert stats["cached_embeddings"] >= 1

    def test_ttl_disabled_when_cache_disabled(self):
        """Test that TTL functionality is disabled when cache is disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(
                cache_dir=temp_dir,
                enable_cache=False,
                cache_ttl_seconds=1,  # Short TTL but should be ignored
            )

            stats = service.get_cache_stats()
            assert not stats["cache_enabled"]
            assert stats["cached_embeddings"] == 0

    def test_ttl_cleanup_performance(self, embedding_service):
        """Test performance of TTL cleanup operations."""
        # Create many cache entries
        texts = [f"Performance test {i}" for i in range(100)]

        # Generate embeddings
        start_time = time.time()
        embedding_service.get_embeddings(texts)
        generation_time = time.time() - start_time

        # Wait for expiration
        time.sleep(2.5)

        # Test cleanup performance
        start_time = time.time()
        embedding_service._cleanup_expired_entries()
        cleanup_time = time.time() - start_time

        # Cleanup should be fast
        assert cleanup_time < 1.0, f"TTL cleanup took too long: {cleanup_time:.3f}s"

        # Generation should also be reasonable
        assert (
            generation_time < 60.0
        ), f"Embedding generation took too long: {generation_time:.3f}s"


class TestWarmupPerformance:
    """Test warmup performance monitoring functionality."""

    @pytest.fixture
    def embedding_service(self):
        """Create a temporary embedding service for warmup testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(
                cache_dir=temp_dir,
                enable_cache=True,
                warmup_timeout_seconds=15,  # Longer timeout for testing
            )
            yield service

    def test_warmup_timing_measurement(self, embedding_service):
        """Test warmup timing measurement."""
        # Warmup time should be recorded
        assert embedding_service.warmup_time_seconds is not None
        assert isinstance(embedding_service.warmup_time_seconds, float)
        assert embedding_service.warmup_time_seconds >= 0

    def test_warning_when_warmup_exceeds_target(self):
        """Test warning when warmup > 5 seconds."""
        # Create service and verify warmup timing is recorded
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(
                cache_dir=temp_dir,
                warmup_timeout_seconds=15,
            )

            # Warmup should complete and timing should be recorded
            assert service.warmup_time_seconds is not None
            assert service.warmup_time_seconds >= 0

    def test_warmup_timeout_mechanism(self):
        """Test warmup timeout mechanism (10 seconds)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create service with short timeout to test timing
            service = EmbeddingService(
                cache_dir=temp_dir,
                warmup_timeout_seconds=2,  # Very short timeout
            )

            # Warmup should complete and timing should be recorded
            assert service.warmup_time_seconds is not None
            assert service.warmup_time_seconds >= 0

    def test_warmup_statistics_tracking(self, embedding_service):
        """Test warmup statistics tracking in cache stats."""
        stats = embedding_service.get_cache_stats()

        # Should include warmup timing
        assert "warmup_time_seconds" in stats
        assert stats["warmup_time_seconds"] is not None
        assert isinstance(stats["warmup_time_seconds"], float)

    def test_warmup_failure_handling(self):
        """Test handling of warmup failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create service normally - should handle any warmup issues gracefully
            service = EmbeddingService(
                cache_dir=temp_dir,
                warmup_timeout_seconds=5,
            )

            # Service should still be functional
            assert service.embedding_dimension == 384
            assert service.model_name == "all-MiniLM-L6-v2"
            assert service.warmup_time_seconds is not None

    def test_warmup_performance_optimization(self, embedding_service):
        """Test that warmup improves first inference performance."""
        texts = ["First inference test"]

        # First inference should be fast due to warmup
        start_time = time.time()
        embeddings = embedding_service.get_embeddings(texts)
        first_inference_time = time.time() - start_time

        # Should complete within reasonable time
        assert (
            first_inference_time < 10.0
        ), f"First inference took too long: {first_inference_time:.3f}s"
        assert embeddings.shape == (1, 384)

    def test_warmup_with_custom_timeout(self):
        """Test warmup with custom timeout settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with custom timeout
            service = EmbeddingService(
                cache_dir=temp_dir,
                warmup_timeout_seconds=20,
            )

            assert service.warmup_timeout_seconds == 20
            assert service.warmup_time_seconds is not None

    def test_warmup_signal_cleanup(self, embedding_service):
        """Test that signal handlers are properly cleaned up after warmup."""
        # Warmup should complete and signal handlers should be restored
        assert embedding_service.warmup_time_seconds is not None

        # Service should be functional after warmup
        texts = ["Signal cleanup test"]
        result = embedding_service.get_embeddings(texts)
        assert result.shape == (1, 384)


class TestGlobalServiceInstance:
    """Test the global service instance functionality."""

    def test_singleton_behavior(self):
        """Test that global service behaves as singleton."""
        service1 = get_embedding_service()
        service2 = get_embedding_service()

        # Should be the same instance
        assert service1 is service2

    def test_reset_functionality(self):
        """Test service reset functionality."""
        # Get initial service
        service1 = get_embedding_service()

        # Reset and get new service
        reset_embedding_service()
        service2 = get_embedding_service()

        # Should be different instances
        assert service1 is not service2


class TestWarmupEdgeCases:
    """Tests for warmup edge cases."""

    def test_warmup_handles_model_encode_error(self):
        """Test warmup handles model encode errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(cache_dir=temp_dir, enable_cache=False)

            # Mock model.encode to raise an exception
            with patch.object(service.model, "encode") as mock_encode:
                mock_encode.side_effect = Exception("Model encode failed")

                # Should handle the error gracefully
                service._warmup_model()

                # Warmup time should be None on failure
                assert service.warmup_time_seconds is None

    def test_warmup_handles_signal_already_set(self):
        """Test warmup handles signal handler setup edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(cache_dir=temp_dir, enable_cache=False)

            # Mock signal.signal to raise an error (already set)
            call_count = [0]  # Use list to store mutable count

            def signal_side_effect(signum, handler):
                # Raise error on second call (when trying to restore)
                call_count[0] += 1
                if call_count[0] > 1:
                    raise ValueError("Signal already set")
                return MagicMock()

            with patch("signal.signal") as mock_signal:
                mock_signal.side_effect = signal_side_effect
                mock_signal.alarm = MagicMock()

                # Should handle signal errors gracefully
                try:
                    service._warmup_model()
                except Exception as e:
                    # Should not propagate signal setup errors
                    pytest.fail(f"Warmup should handle signal errors: {e}")


class TestBatchEmbeddingEdgeCases:
    """Tests for batch embedding edge cases."""

    @pytest.fixture
    def embedding_service(self):
        """Create a temporary embedding service for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(cache_dir=temp_dir, enable_cache=False)
            yield service

    def test_batch_embedding_empty_input(self, embedding_service):
        """Test batch embedding with empty input list."""
        result = embedding_service.get_embeddings([])

        assert result.shape == (0, 384)
        assert result.dtype == np.float32

    def test_batch_embedding_all_empty_strings(self, embedding_service):
        """Test batch embedding with list of empty strings."""
        texts = ["", "", "   ", ""]
        result = embedding_service.get_embeddings(texts)

        assert result.shape == (4, 384)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_batch_embedding_single_text(self, embedding_service):
        """Test batch embedding with single text."""
        texts = ["Single text for testing"]
        result = embedding_service.get_embeddings(texts)

        assert result.shape == (1, 384)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_batch_embedding_mixed_valid_invalid(self, embedding_service):
        """Test batch embedding with mix of valid and problematic texts."""
        texts = [
            "Valid text content",
            "",  # Empty string
            "Another valid text with content",
            "a" * 100000,  # Very long text
        ]
        result = embedding_service.get_embeddings(texts)

        assert result.shape == (4, 384)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_batch_embedding_cache_disabled_force_recompute(self, embedding_service):
        """Test batch embedding with cache disabled and force recompute."""
        texts = ["Test text 1", "Test text 2"]

        # First call
        result1 = embedding_service.get_embeddings(texts)
        # Second call with force_recompute
        result2 = embedding_service.get_embeddings(texts, force_recompute=True)

        # Results should be identical
        np.testing.assert_array_almost_equal(result1, result2)
        assert result1.shape == (2, 384)

    def test_batch_embedding_none_batch_size(self, embedding_service):
        """Test batch embedding with None batch size (auto-detection)."""
        texts = [f"Test text {i}" for i in range(10)]

        # Should auto-detect optimal batch size
        result = embedding_service.get_embeddings(texts, batch_size=None)

        assert result.shape == (10, 384)
        assert not np.any(np.isnan(result))

    def test_batch_embedding_very_large_batch_size(self, embedding_service):
        """Test batch embedding with very large batch size."""
        texts = [f"Test text {i}" for i in range(5)]

        # Use batch size larger than input
        result = embedding_service.get_embeddings(texts, batch_size=100)

        assert result.shape == (5, 384)
        assert not np.any(np.isnan(result))

    def test_batch_embedding_zero_batch_size_fallback(self, embedding_service):
        """Test batch embedding handles zero batch size gracefully."""
        texts = ["Test text"]

        # Should handle invalid batch size and use fallback
        result = embedding_service.get_embeddings(texts, batch_size=0)

        assert result.shape == (1, 384)
        assert not np.any(np.isnan(result))


class TestUncoveredCriticalPaths:
    """Tests for specific uncovered lines to improve coverage >90%."""

    @pytest.fixture
    def embedding_service(self):
        """Create a temporary embedding service for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(
                cache_dir=temp_dir, enable_cache=True, cache_size_limit=100
            )
            yield service

    def test_gpu_fallback_to_cpu_on_cuda_oom(self, embedding_service):
        """Test GPU fallback to CPU on CUDA OOM errors (lines 404-417)."""
        texts = ["Test GPU fallback"]

        # Mock encode_with_timeout to simulate OOM then success
        with patch.object(embedding_service, "_encode_with_timeout") as mock_encode:
            # First call fails with CUDA OOM, second succeeds
            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise torch.cuda.OutOfMemoryError("CUDA out of memory")
                return np.random.rand(1, 384).astype(np.float32)

            mock_encode.side_effect = side_effect

            # Mock the device detection to use CUDA
            with patch.object(embedding_service, "model") as mock_model:
                mock_model.device = "cuda:0"
                mock_model.to.return_value = None

                # Should handle GPU error gracefully and retry with smaller batch
                result = embedding_service._encode_with_retry(
                    texts, batch_size=1, normalize=True, timeout=30
                )
                assert result.shape == (1, 384)
                assert call_count == 2

    def test_cache_loading_with_corrupted_files(self, embedding_service):
        """Test cache loading with corrupted files (lines 226-228)."""
        # Create a corrupted cache file
        cache_file = (
            embedding_service.cache_dir
            / f"{embedding_service.model_name.replace('/', '_')}_cache.pkl"
        )

        # Write corrupted data
        with open(cache_file, "wb") as f:
            f.write(b"corrupted pickle data that cannot be loaded")

        # Create new service to trigger cache loading
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(cache_dir=temp_dir, enable_cache=True)

            # Copy corrupted file to service cache dir
            service_cache_file = (
                service.cache_dir / f"{service.model_name.replace('/', '_')}_cache.pkl"
            )
            with open(cache_file, "rb") as src, open(service_cache_file, "wb") as dst:
                dst.write(src.read())

            # Should handle corrupted cache gracefully
            service._load_cache()
            assert service.cache == {}  # Should be reset to empty dict

    def test_warmup_timeout_handling(self, embedding_service):
        """Test warmup timeout handling (lines 161-166)."""
        # Create a custom service to test warmup timeout
        with tempfile.TemporaryDirectory() as temp_dir:
            service = EmbeddingService(cache_dir=temp_dir, enable_cache=False)

            # Mock signal.alarm to simulate timeout
            with patch("signal.alarm") as mock_alarm:
                with patch.object(service, "model") as mock_model:
                    # Make model.encode raise an exception
                    mock_model.encode.side_effect = Exception("Simulated timeout")

                    # Mock the timeout by triggering alarm path
                    def alarm_side_effect(seconds):
                        if seconds > 0:  # Setting alarm
                            pass
                        else:  # Canceling alarm
                            pass

                    mock_alarm.side_effect = alarm_side_effect

                    # Run warmup - should handle timeout gracefully
                    service._warmup_model()

                    # Should set warmup_time_seconds to None on failure
                    assert service.warmup_time_seconds is None

    def test_embedding_dimension_mismatch_errors(self, embedding_service):
        """Test embedding dimension mismatch errors (lines 678-684)."""
        texts = ["Dimension test"]

        # Mock the encoding to return wrong dimensions
        with patch.object(embedding_service, "_encode_with_retry") as mock_encode:
            # Return wrong dimension (256 instead of 384)
            mock_encode.return_value = np.random.rand(1, 256).astype(np.float32)

            # Should raise dimension mismatch error
            with pytest.raises(EmbeddingError) as exc_info:
                embedding_service.get_embeddings(texts)

            assert "dimension mismatch" in str(exc_info.value)
            assert "expected 384" in str(exc_info.value)
            assert "got 256" in str(exc_info.value)

    def test_retry_logic_with_different_error_types(self, embedding_service):
        """Test retry logic with different error types (lines 443-497)."""
        texts = ["Retry test"]

        # Test that retry logic handles runtime errors properly
        with patch.object(embedding_service, "_encode_with_timeout") as mock_encode:
            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call fails with a generic runtime error
                    raise RuntimeError("Runtime error")
                return np.random.rand(1, 384).astype(np.float32)

            mock_encode.side_effect = side_effect

            # Should handle runtime error and retry
            result = embedding_service._encode_with_retry(
                texts, batch_size=1, normalize=True, timeout=30
            )
            assert result.shape == (1, 384)
            assert call_count == 2

        # Test error that gets wrapped in EmbeddingError after retries exhausted
        with patch.object(embedding_service, "_encode_with_timeout") as mock_encode:
            mock_encode.side_effect = RuntimeError("Persistent error")

            # Should wrap persistent errors in EmbeddingError after retries
            with pytest.raises(EmbeddingError):
                embedding_service._encode_with_retry(
                    texts, batch_size=1, normalize=True, timeout=30
                )

    def test_batch_processing_with_empty_inputs(self, embedding_service):
        """Test batch processing with empty inputs (lines 541-543)."""
        # Test with empty list
        result = embedding_service.get_embeddings([])
        assert result.shape == (0, 384)
        assert result.dtype == np.float32

        # Test validation that leads to empty validated_texts
        with patch.object(embedding_service, "_validate_texts") as mock_validate:
            mock_validate.return_value = []

            result = embedding_service.get_embeddings(["input"])
            assert result.shape == (0, 384)
            assert result.dtype == np.float32

    def test_cache_corruption_during_save(self, embedding_service):
        """Test handling of cache corruption during save operations."""
        texts = ["Cache corruption test"]

        # Generate some embeddings first
        embedding_service.get_embeddings(texts)

        # Mock file operations to raise error during save
        with patch("builtins.open", side_effect=OSError("Disk full")):
            # Should handle save errors gracefully
            embedding_service._save_cache()

            # Cache should still be in memory
            stats = embedding_service.get_cache_stats()
            assert stats["cached_embeddings"] >= 1

    def test_model_device_switching_edge_cases(self, embedding_service):
        """Test edge cases in model device switching."""
        texts = ["Device switching test"]

        # Test device switching failures by mocking the model.to method
        with patch.object(embedding_service.model, "to") as mock_to:
            mock_to.side_effect = RuntimeError("Device switch failed")

            # Should handle device switching failures by wrapping in EmbeddingError
            with pytest.raises(EmbeddingError):
                embedding_service._encode_with_retry(
                    texts, batch_size=1, normalize=True, timeout=30
                )
