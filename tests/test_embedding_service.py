"""
Performance tests for embedding service.

Tests embedding generation performance, caching behavior, and batch processing.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from doc_server.search.embedding_service import (
    EmbeddingService,
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
        assert optimal_batch_time <= small_batch_time * 1.2, (
            f"Optimal batch size ({optimal_batch_time:.3f}s) should be competitive with small batch ({small_batch_time:.3f}s)"
        )

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
        assert cached_time < first_time * 0.5, (
            f"Cached version ({cached_time:.3f}s) should be much faster than first ({first_time:.3f}s)"
        )

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
        assert processing_time < 120.0, (
            f"Large batch processing took too long: {processing_time:.2f}s"
        )

        # Verify no NaN values
        assert not np.any(np.isnan(embeddings)), (
            "Embeddings should not contain NaN values"
        )

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
        assert similarity_time < 5.0, (
            f"Similarity computation took too long: {similarity_time:.3f}s"
        )

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
        assert texts_per_second > 5.0, (
            f"Performance regression: only {texts_per_second:.1f} texts/second (expected > 5.0)"
        )

        # Embedding quality check - should have reasonable variance
        embedding_std = np.std(embeddings)
        assert 0.01 < embedding_std < 1.0, (
            f"Embedding variance seems unusual: std={embedding_std:.3f}"
        )

        # Memory efficiency check
        memory_per_embedding = embeddings.nbytes / len(embeddings)
        expected_size = 384 * 4  # 384 dimensions * 4 bytes per float32
        assert abs(memory_per_embedding - expected_size) < 10, (
            f"Memory usage unexpected: {memory_per_embedding} bytes per embedding"
        )


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
