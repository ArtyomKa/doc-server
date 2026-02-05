"""
Embedding service for generating document embeddings.

Provides high-performance embedding generation using sentence-transformers
with all-MiniLM-L6-v2 model, including caching, batch processing,
and performance optimizations.
"""

import hashlib
import logging
import pickle
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from ..config import settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base exception for embedding service errors."""

    pass


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when embedding generation times out."""

    pass


class EmbeddingGPUError(EmbeddingError):
    """Raised when GPU-related errors occur during embedding generation."""

    pass


class EmbeddingValidationError(EmbeddingError):
    """Raised when input validation fails."""

    pass


class EmbeddingRetryExhaustedError(EmbeddingError):
    """Raised when all retry attempts are exhausted."""

    pass


class EmbeddingService:
    """
    High-performance embedding service with caching and batch processing.

    Features:
    - all-MiniLM-L6-v2 model (384 dimensions, fast inference)
    - Persistent caching for repeated embeddings
    - Batch processing with optimal memory management
    - GPU acceleration when available
    - Model warmup for optimal first inference
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
        cache_dir: str | Path | None = None,
        enable_cache: bool = True,
        cache_size_limit: int = 10000,
        cache_ttl_seconds: int = 86400,
        warmup_timeout_seconds: int = 10,
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device for computation ("auto", "cpu", "cuda")
            cache_dir: Directory for persistent cache storage
            enable_cache: Whether to enable embedding caching
            cache_size_limit: Maximum number of cached embeddings
            cache_ttl_seconds: TTL for cache entries in seconds (default 24h)
            warmup_timeout_seconds: Timeout for model warmup in seconds (default 10)
        """
        self.model_name = model_name
        self.enable_cache = enable_cache
        self.cache_size_limit = cache_size_limit
        self.cache_ttl_seconds = cache_ttl_seconds
        self.warmup_timeout_seconds = warmup_timeout_seconds
        self.warmup_time_seconds: float | None = None

        # Initialize model with device auto-detection
        actual_device = self._determine_device(device)
        logger.info(f"Loading model {model_name} on device: {actual_device}")

        self.model = SentenceTransformer(model_name, device=actual_device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

        # Setup caching
        self.cache: dict[str, Any] = {}
        if self.enable_cache:
            self.cache_dir = Path(
                cache_dir or settings.storage_path / "embeddings_cache"
            )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

        # Optimize settings and warmup
        self._optimize_settings()
        self._warmup_model()

        logger.info(
            f"EmbeddingService initialized with dimension: {self.embedding_dimension}"
        )

    def _determine_device(self, device: str) -> str:
        """Determine the optimal device for computation."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _optimize_settings(self) -> None:
        """Apply performance optimizations based on device."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.set_num_threads(4)  # Limit CPU threads when using GPU
            logger.info("GPU optimizations enabled")

    def _warmup_model(self) -> None:
        """Warm up the model with dummy inputs for optimal first inference."""
        logger.debug("Warming up embedding model...")

        class WarmupTimeoutError(Exception):
            pass

        def timeout_handler(signum: int, frame: Any) -> None:
            raise WarmupTimeoutError("Model warmup timed out")

        # Set up signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.warmup_timeout_seconds)

        start_time = time.time()

        try:
            dummy_texts = [
                "This is a warmup sentence for model optimization.",
                "Another warmup sentence to initialize GPU kernels.",
            ]

            with torch.no_grad():
                _ = self.model.encode(dummy_texts, show_progress_bar=False)

            signal.alarm(0)  # Cancel the alarm

        except WarmupTimeoutError:
            signal.alarm(0)  # Cancel the alarm
            warmup_time = self.warmup_timeout_seconds
            self.warmup_time_seconds = warmup_time
            logger.error(f"Model warmup timed out after {warmup_time}s (>10s limit)")
            return  # Don't raise exception, just log and continue

        except Exception as e:
            signal.alarm(0)  # Cancel the alarm
            logger.error(f"Model warmup failed with error: {e}")
            self.warmup_time_seconds = None
            return

        finally:
            # Restore original signal handler (may fail in some environments)
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, OSError):
                pass  # Ignore signal handler restoration errors

        actual_warmup_time = time.time() - start_time
        self.warmup_time_seconds = actual_warmup_time

        # Log timing with appropriate level
        if actual_warmup_time > 5.0:
            logger.warning(f"Model warmup took {actual_warmup_time:.2f}s (>5s target)")
        else:
            logger.info(f"Model warmup completed in {actual_warmup_time:.2f}s")

    def _get_text_hash(self, text: str) -> str:
        """Generate consistent hash for text caching."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    loaded_cache = pickle.load(f)

                # Handle backward compatibility for old cache format
                self.cache = {}
                expired_count = 0
                current_time = time.time()

                for key, value in loaded_cache.items():
                    if isinstance(value, tuple) and len(value) == 2:
                        # New format: (embedding, timestamp)
                        embedding, timestamp = value
                        if current_time - timestamp <= self.cache_ttl_seconds:
                            self.cache[key] = value
                        else:
                            expired_count += 1
                    else:
                        # Old format: just embedding - add current timestamp
                        self.cache[key] = (value, current_time)

                if expired_count > 0:
                    logger.info(
                        f"Removed {expired_count} expired cache entries during load"
                    )

                logger.debug(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Error loading embedding cache: {e}")
                self.cache = {}
        else:
            logger.debug("No existing cache file found")

    def _cleanup_expired_entries(self) -> int:
        """Remove expired entries from cache and return count of removed entries."""
        if not self.cache:
            return 0

        current_time = time.time()
        expired_keys = []

        for key, value in self.cache.items():
            if isinstance(value, tuple) and len(value) == 2:
                _, timestamp = value
                if current_time - timestamp > self.cache_ttl_seconds:
                    expired_keys.append(key)

        # Remove expired entries
        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if not self.enable_cache:
            return

        cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"

        try:
            # Remove expired entries
            expired_count = self._cleanup_expired_entries()

            # Limit cache size to prevent memory bloat
            if len(self.cache) > self.cache_size_limit:
                # Keep only the most recent embeddings (simple LRU simulation)
                items = list(self.cache.items())
                self.cache = dict(items[-self.cache_size_limit :])

            with open(cache_file, "wb") as f:
                pickle.dump(self.cache, f)
            logger.debug(f"Saved {len(self.cache)} embeddings to cache")
            if expired_count > 0:
                logger.info(f"Removed {expired_count} expired entries before saving")
        except Exception as e:
            logger.warning(f"Error saving embedding cache: {e}")

    def _get_optimal_batch_size(self, device: str) -> int:
        """Get optimal batch size based on device and memory."""
        if device.startswith("cuda"):
            return 32  # GPU can handle larger batches
        return 16  # CPU needs smaller batches

    def _validate_texts(self, texts: list[str]) -> list[str]:
        """
        Validate and sanitize input texts before encoding.

        Args:
            texts: List of text strings to validate

        Returns:
            List of validated texts

        Raises:
            EmbeddingValidationError: If texts are invalid
        """
        if not isinstance(texts, list):
            raise EmbeddingValidationError("Input must be a list of strings")

        validated_texts = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                logger.warning(
                    f"Text at index {i} is not a string, converting to string"
                )
                text = str(text)

            # Remove null bytes and other problematic characters
            text = text.replace("\x00", "")

            # Check for empty strings (but allow them as they might be meaningful)
            if text == "":
                logger.debug(f"Empty string at index {i}")

            # Check for extremely long texts that might cause memory issues
            if len(text) > 100000:  # 100k characters
                logger.warning(
                    f"Text at index {i} is extremely long ({len(text)} chars), truncating"
                )
                text = text[:100000]

            validated_texts.append(text)

        return validated_texts

    def _encode_with_timeout(self, texts: list[str], timeout: int = 30) -> np.ndarray:
        """
        Encode texts with timeout handling using signals.

        Args:
            texts: List of texts to encode
            timeout: Timeout in seconds

        Returns:
            Numpy array of embeddings

        Raises:
            EmbeddingTimeoutError: If encoding times out
        """

        class TimeoutError(Exception):
            pass

        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError("Encoding operation timed out")

        # Set up signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            result = self.model.encode(
                texts,
                batch_size=len(texts),  # Process all at once for this call
                normalize_embeddings=False,  # Handle normalization outside
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            signal.alarm(0)  # Cancel the alarm
            return result
        except TimeoutError:
            signal.alarm(0)  # Cancel the alarm
            raise EmbeddingTimeoutError(
                f"Encoding timed out after {timeout} seconds"
            ) from None
        finally:
            # Restore original signal handler (may fail in some environments)
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, OSError):
                pass  # Ignore signal handler restoration errors

    def _encode_with_retry(
        self,
        texts: list[str],
        batch_size: int,
        normalize: bool,
        max_retries: int = 3,
        initial_delay: float = 0.1,
        timeout: int = 30,
    ) -> np.ndarray:
        """
        Encode texts with retry logic and GPU fallback.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries (seconds)
            timeout: Timeout for each encoding attempt

        Returns:
            Numpy array of embeddings

        Raises:
            EmbeddingRetryExhaustedError: If all retries fail
        """
        current_device = str(self.model.device)
        fallback_to_cpu = False

        for attempt in range(max_retries):
            try:
                # Prepare texts for this attempt
                batch_texts = texts[:batch_size] if len(texts) > batch_size else texts

                if fallback_to_cpu and current_device.startswith("cuda"):
                    # Force CPU fallback
                    logger.info("Switching to CPU due to previous GPU errors")
                    original_device = self.model.device
                    self.model.to("cpu")
                    try:
                        embeddings = self._encode_with_timeout(batch_texts, timeout)
                        if normalize:
                            embeddings = embeddings / np.linalg.norm(
                                embeddings, axis=1, keepdims=True
                            )
                        return embeddings
                    finally:
                        self.model.to(original_device)
                else:
                    # Normal encoding path
                    embeddings = self._encode_with_timeout(batch_texts, timeout)
                    if normalize:
                        embeddings = embeddings / np.linalg.norm(
                            embeddings, axis=1, keepdims=True
                        )
                    return embeddings

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"CUDA out of memory on attempt {attempt + 1}: {e}")
                if not fallback_to_cpu and current_device.startswith("cuda"):
                    fallback_to_cpu = True
                    # Reduce batch size for next attempt
                    batch_size = max(1, batch_size // 2)
                    logger.info(
                        f"Reducing batch size to {batch_size} and enabling CPU fallback"
                    )
                    continue
                else:
                    # Already using CPU fallback, still getting OOM (unlikely)
                    raise EmbeddingGPUError(
                        f"CUDA out of memory even with CPU fallback: {e}"
                    ) from e

            except RuntimeError as e:
                error_msg = str(e).lower()
                if "cuda" in error_msg or "gpu" in error_msg:
                    logger.warning(f"GPU runtime error on attempt {attempt + 1}: {e}")
                    if not fallback_to_cpu and current_device.startswith("cuda"):
                        fallback_to_cpu = True
                        batch_size = max(1, batch_size // 2)
                        logger.info(
                            f"Switching to CPU due to GPU error, reduced batch size to {batch_size}"
                        )
                        continue
                    else:
                        raise EmbeddingGPUError(
                            f"GPU error even with CPU fallback: {e}"
                        ) from e
                else:
                    # Non-GPU runtime error
                    logger.warning(f"Runtime error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        delay = initial_delay * (2**attempt)  # Exponential backoff
                        logger.info(f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        raise EmbeddingError(
                            f"Runtime error after {max_retries} attempts: {e}"
                        ) from e

            except EmbeddingTimeoutError as e:
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    # Increase timeout for next attempt
                    timeout = min(timeout * 2, 60)  # Double timeout, max 60s
                    delay = initial_delay * (2**attempt)
                    logger.info(
                        f"Increasing timeout to {timeout}s and retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingTimeoutError(
                        f"Encoding timed out after {max_retries} attempts"
                    ) from None

            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = initial_delay * (2**attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    raise EmbeddingError(
                        f"Unexpected error after {max_retries} attempts: {e}"
                    ) from e

        # This should never be reached due to exceptions above
        raise EmbeddingRetryExhaustedError(
            f"Failed to encode after {max_retries} attempts"
        )

    def get_embeddings(
        self,
        texts: list[str],
        batch_size: int | None = None,
        force_recompute: bool = False,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts with caching and robust error handling.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (auto-detected if None)
            force_recompute: Force recompute even if cached
            normalize: Whether to normalize embeddings (recommended for cosine similarity)
            show_progress: Show progress bar for long operations

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dimension)

        Raises:
            EmbeddingValidationError: If input validation fails
            EmbeddingTimeoutError: If encoding times out
            EmbeddingGPUError: If GPU errors occur and fallback fails
            EmbeddingRetryExhaustedError: If all retry attempts fail
            EmbeddingError: For other embedding-related errors
        """
        # Input validation
        try:
            validated_texts = self._validate_texts(texts)
        except EmbeddingValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            raise EmbeddingValidationError(f"Input validation failed: {e}") from e

        if not validated_texts:
            return np.empty((0, self.embedding_dimension or 384), dtype=np.float32)

        # Determine optimal batch size
        if batch_size is None:
            batch_size = self._get_optimal_batch_size(str(self.model.device))

        # Ensure batch_size is valid (at least 1)
        if batch_size is not None and batch_size < 1:
            batch_size = 1

        embeddings: list[np.ndarray | None] = [None] * len(validated_texts)
        texts_to_encode: list[str] = []
        indices_to_encode: list[int] = []

        # Check cache for existing embeddings
        if self.enable_cache and not force_recompute:
            for i, text in enumerate(validated_texts):
                text_hash = self._get_text_hash(text)

                if text_hash in self.cache:
                    cache_entry = self.cache[text_hash]
                    if isinstance(cache_entry, tuple) and len(cache_entry) == 2:
                        # New format: (embedding, timestamp)
                        embedding, timestamp = cache_entry
                        # Check TTL
                        current_time = time.time()
                        if current_time - timestamp <= self.cache_ttl_seconds:
                            embeddings[i] = embedding
                        else:
                            # Expired entry, remove and mark for recompute
                            del self.cache[text_hash]
                            texts_to_encode.append(text)
                            indices_to_encode.append(i)
                    else:
                        # Old format: just embedding - treat as expired to force recompute
                        del self.cache[text_hash]
                        texts_to_encode.append(text)
                        indices_to_encode.append(i)
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
        else:
            # Force recompute all texts
            texts_to_encode = validated_texts
            indices_to_encode = list(range(len(validated_texts)))

        # Encode new texts in batches with error handling
        if texts_to_encode:
            logger.debug(
                f"Encoding {len(texts_to_encode)} new texts in batches of {batch_size}"
            )

            try:
                # Process texts in batches with error handling
                batch_embeddings_list: list[np.ndarray] = []
                batch_start = 0

                while batch_start < len(texts_to_encode):
                    batch_end = batch_start + batch_size
                    batch_texts = texts_to_encode[batch_start:batch_end]

                    try:
                        # Use the robust encoding method with retry and fallback
                        batch_embeddings = self._encode_with_retry(
                            batch_texts,
                            batch_size=len(batch_texts),  # Process entire batch at once
                            normalize=normalize,
                            max_retries=3,
                            initial_delay=0.1,
                            timeout=30,
                        )
                        batch_embeddings_list.append(batch_embeddings)

                    except (
                        EmbeddingTimeoutError,
                        EmbeddingGPUError,
                        EmbeddingRetryExhaustedError,
                    ) as e:
                        # Re-raise critical errors
                        logger.error(f"Critical error during batch encoding: {e}")
                        raise
                    except EmbeddingError as e:
                        # Log but continue with other batches for non-critical errors
                        logger.warning(
                            f"Embedding error in batch {batch_start}-{batch_end}: {e}"
                        )
                        # Create zero embeddings as fallback
                        dim = self.embedding_dimension or 384
                        zero_embeddings = np.zeros((len(batch_texts), dim))
                        if normalize:
                            zero_embeddings = zero_embeddings / np.linalg.norm(
                                zero_embeddings, axis=1, keepdims=True
                            )
                        batch_embeddings_list.append(zero_embeddings)

                    batch_start = batch_end

                # Concatenate all batch embeddings
                if batch_embeddings_list:
                    new_embeddings = np.concatenate(batch_embeddings_list, axis=0)
                else:
                    new_embeddings = np.empty((0, self.embedding_dimension or 384))

                # Update cache and fill result array
                for idx, text, embedding in zip(
                    indices_to_encode, texts_to_encode, new_embeddings, strict=True
                ):
                    embeddings[idx] = embedding

                    if self.enable_cache and not force_recompute:
                        text_hash = self._get_text_hash(text)
                        self.cache[text_hash] = (embedding, time.time())

                # Save updated cache periodically
                if self.enable_cache:
                    self._save_cache()

            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                # Return zero embeddings as graceful fallback
                logger.warning("Returning zero embeddings as fallback")
                dim = self.embedding_dimension or 384
                return np.zeros((len(validated_texts), dim))

        # Convert to numpy array and validate
        try:
            result = np.array(embeddings)
        except Exception as e:
            logger.error(f"Failed to convert embeddings to numpy array: {e}")
            raise EmbeddingError(f"Failed to process embeddings: {e}") from e

        # Validate result dimensions
        if result.shape[0] != len(validated_texts):
            logger.error(
                f"Embedding count mismatch: expected {len(validated_texts)}, got {result.shape[0]}"
            )
            raise EmbeddingError(
                f"Embedding count mismatch: expected {len(validated_texts)}, got {result.shape[0]}"
            )

        if result.shape[1] != self.embedding_dimension:
            logger.error(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {result.shape[1]}"
            )
            raise EmbeddingError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {result.shape[1]}"
            )

        return result

    def compute_similarity(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 10,
        score_function: str = "cosine",
    ) -> list[list[dict[str, Any]]]:
        """
        Compute similarity between queries and corpus.

        Args:
            query_embeddings: Query embeddings of shape (n_queries, embedding_dim)
            corpus_embeddings: Corpus embeddings of shape (n_corpus, embedding_dim)
            top_k: Number of top results to return per query
            score_function: Similarity function ("cosine" or "dot")

        Returns:
            List of similarity results, one list per query
        """
        # Convert to tensors for GPU processing
        query_tensor = torch.from_numpy(query_embeddings)
        corpus_tensor = torch.from_numpy(corpus_embeddings)

        # Move to GPU if available
        if torch.cuda.is_available():
            query_tensor = query_tensor.to("cuda")
            corpus_tensor = corpus_tensor.to("cuda")

        # Normalize for better similarity computation
        query_tensor = util.normalize_embeddings(query_tensor)
        corpus_tensor = util.normalize_embeddings(corpus_tensor)

        # Choose score function
        if score_function == "dot":
            score_fn = util.dot_score
        else:
            score_fn = util.cos_sim

        # Compute semantic search
        hits = util.semantic_search(
            query_tensor, corpus_tensor, score_function=score_fn, top_k=top_k
        )

        return hits

    def batch_process(
        self, text_batches: list[list[str]], batch_size: int | None = None
    ) -> list[np.ndarray]:
        """
        Process multiple text batches efficiently.

        Args:
            text_batches: List of text batches to process
            batch_size: Batch size for processing within each batch

        Returns:
            List of embedding arrays, one per input batch
        """
        results = []

        for batch in text_batches:
            embeddings = self.get_embeddings(batch, batch_size=batch_size)
            results.append(embeddings)

        return results

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache = {}
        if self.enable_cache and self.cache_dir.exists():
            cache_file = (
                self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"
            )
            if cache_file.exists():
                cache_file.unlink()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the embedding cache."""
        # Count expired entries
        expired_count = 0
        current_time = time.time()

        if self.cache:
            for value in self.cache.values():
                if isinstance(value, tuple) and len(value) == 2:
                    _, timestamp = value
                    if current_time - timestamp > self.cache_ttl_seconds:
                        expired_count += 1

        stats = {
            "cache_enabled": self.enable_cache,
            "cached_embeddings": len(self.cache),
            "cache_size_limit": self.cache_size_limit,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "expired_entries_count": expired_count,
            "cache_utilization": (
                len(self.cache) / self.cache_size_limit if self.enable_cache else 0
            ),
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": str(self.model.device),
        }

        # Add warmup timing if available
        if self.warmup_time_seconds is not None:
            stats["warmup_time_seconds"] = self.warmup_time_seconds

        if self.enable_cache:
            cache_file = (
                self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"
            )
            stats["cache_file_exists"] = cache_file.exists()
            if cache_file.exists():
                stats["cache_file_size_mb"] = cache_file.stat().st_size / (1024 * 1024)

        return stats


# Global embedding service instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService()

    return _embedding_service


def reset_embedding_service() -> None:
    """Reset the global embedding service instance."""
    global _embedding_service
    _embedding_service = None
