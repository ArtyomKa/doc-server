"""
Search and retrieval module.

Provides embedding generation, vector storage, and hybrid search
capabilities for processed documentation.
"""

from .embedding_service import (
    EmbeddingError,
    EmbeddingGPUError,
    EmbeddingRetryExhaustedError,
    EmbeddingService,
    EmbeddingTimeoutError,
    EmbeddingValidationError,
    get_embedding_service,
    reset_embedding_service,
)
from .vector_store import (
    ChromaEmbeddingFunction,
    ChromaVectorStore,
    CollectionAlreadyExistsError,
    CollectionCreationError,
    CollectionDeletionError,
    CollectionNotFoundError,
    DocumentAdditionError,
    DocumentQueryError,
    VectorStoreConnectionError,
    VectorStoreError,
    get_vector_store,
    reset_vector_store,
)

__all__ = [
    # Embedding service
    "EmbeddingError",
    "EmbeddingGPUError",
    "EmbeddingRetryExhaustedError",
    "EmbeddingService",
    "EmbeddingTimeoutError",
    "EmbeddingValidationError",
    "get_embedding_service",
    "reset_embedding_service",
    # Vector store
    "ChromaEmbeddingFunction",
    "ChromaVectorStore",
    "CollectionAlreadyExistsError",
    "CollectionCreationError",
    "CollectionDeletionError",
    "CollectionNotFoundError",
    "DocumentAdditionError",
    "DocumentQueryError",
    "VectorStoreConnectionError",
    "VectorStoreError",
    "get_vector_store",
    "reset_vector_store",
]
