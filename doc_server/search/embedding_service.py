"""
Embedding service for generating document embeddings.

Provides high-performance embedding generation using sentence-transformers
with all-MiniLM-L6-v2 model, including caching, batch processing,
and performance optimizations.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from ..config import settings

logger = logging.getLogger(__name__)


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
        cache_dir: Optional[Union[str, Path]] = None,
        enable_cache: bool = True,
        cache_size_limit: int = 10000,
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device for computation ("auto", "cpu", "cuda")
            cache_dir: Directory for persistent cache storage
            enable_cache: Whether to enable embedding caching
            cache_size_limit: Maximum number of cached embeddings
        """
        self.model_name = model_name
        self.enable_cache = enable_cache
        self.cache_size_limit = cache_size_limit

        # Initialize model with device auto-detection
        actual_device = self._determine_device(device)
        logger.info(f"Loading model {model_name} on device: {actual_device}")

        self.model = SentenceTransformer(model_name, device=actual_device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

        # Setup caching
        self.cache = {}
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

    def _optimize_settings(self):
        """Apply performance optimizations based on device."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.set_num_threads(4)  # Limit CPU threads when using GPU
            logger.info("GPU optimizations enabled")

    def _warmup_model(self):
        """Warm up the model with dummy inputs for optimal first inference."""
        logger.debug("Warming up embedding model...")
        dummy_texts = [
            "This is a warmup sentence for model optimization.",
            "Another warmup sentence to initialize GPU kernels.",
        ]

        with torch.no_grad():
            _ = self.model.encode(dummy_texts, show_progress_bar=False)

        logger.debug("Model warmup completed")

    def _get_text_hash(self, text: str) -> str:
        """Generate consistent hash for text caching."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_cache(self):
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                logger.debug(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Error loading embedding cache: {e}")
                self.cache = {}
        else:
            logger.debug("No existing cache file found")

    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self.enable_cache:
            return

        cache_file = self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"

        try:
            # Limit cache size to prevent memory bloat
            if len(self.cache) > self.cache_size_limit:
                # Keep only the most recent embeddings (simple LRU simulation)
                items = list(self.cache.items())
                self.cache = dict(items[-self.cache_size_limit :])

            with open(cache_file, "wb") as f:
                pickle.dump(self.cache, f)
            logger.debug(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Error saving embedding cache: {e}")

    def _get_optimal_batch_size(self, device: str) -> int:
        """Get optimal batch size based on device and memory."""
        if device.startswith("cuda"):
            return 32  # GPU can handle larger batches
        return 16  # CPU needs smaller batches

    def get_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        force_recompute: bool = False,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts with caching.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (auto-detected if None)
            force_recompute: Force recompute even if cached
            normalize: Whether to normalize embeddings (recommended for cosine similarity)
            show_progress: Show progress bar for long operations

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.empty((0, self.embedding_dimension or 384))

        # Determine optimal batch size
        if batch_size is None:
            batch_size = self._get_optimal_batch_size(str(self.model.device))

        embeddings = [None] * len(texts)
        texts_to_encode = []
        indices_to_encode = []

        # Check cache for existing embeddings
        if self.enable_cache and not force_recompute:
            for i, text in enumerate(texts):
                text_hash = self._get_text_hash(text)

                if text_hash in self.cache:
                    embeddings[i] = self.cache[text_hash]
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
        else:
            # Force recompute all texts
            texts_to_encode = texts
            indices_to_encode = list(range(len(texts)))

        # Encode new texts in batches
        if texts_to_encode:
            logger.debug(
                f"Encoding {len(texts_to_encode)} new texts in batches of {batch_size}"
            )

            new_embeddings = self.model.encode(
                texts_to_encode,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

            # Update cache and fill result array
            for idx, text, embedding in zip(
                indices_to_encode, texts_to_encode, new_embeddings
            ):
                embeddings[idx] = embedding

                if self.enable_cache and not force_recompute:
                    text_hash = self._get_text_hash(text)
                    self.cache[text_hash] = embedding

            # Save updated cache periodically
            if self.enable_cache:
                self._save_cache()

        # Convert to numpy array and validate
        result = np.array(embeddings)

        if result.shape[0] != len(texts):
            raise ValueError(
                f"Embedding count mismatch: expected {len(texts)}, got {result.shape[0]}"
            )

        if result.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {result.shape[1]}"
            )

        return result

    def compute_similarity(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 10,
        score_function: str = "cosine",
    ) -> List[List[dict]]:
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
        self, text_batches: List[List[str]], batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
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

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache = {}
        if self.enable_cache and self.cache_dir.exists():
            cache_file = (
                self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"
            )
            if cache_file.exists():
                cache_file.unlink()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> dict:
        """Get statistics about the embedding cache."""
        stats = {
            "cache_enabled": self.enable_cache,
            "cached_embeddings": len(self.cache),
            "cache_size_limit": self.cache_size_limit,
            "cache_utilization": len(self.cache) / self.cache_size_limit
            if self.enable_cache
            else 0,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": str(self.model.device),
        }

        if self.enable_cache:
            cache_file = (
                self.cache_dir / f"{self.model_name.replace('/', '_')}_cache.pkl"
            )
            stats["cache_file_exists"] = cache_file.exists()
            if cache_file.exists():
                stats["cache_file_size_mb"] = cache_file.stat().st_size / (1024 * 1024)

        return stats


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService()

    return _embedding_service


def reset_embedding_service():
    """Reset the global embedding service instance."""
    global _embedding_service
    _embedding_service = None
