"""
Vector store implementation using ChromaDB for persistent document storage.

Provides high-performance vector storage and retrieval with ChromaDB integration,
including collection management, batch operations, and robust error handling.
"""

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import ChromaError, NotFoundError

from ..config import settings
from .embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Base exception for vector store errors."""

    pass


class CollectionNotFoundError(VectorStoreError):
    """Raised when a requested collection is not found."""

    pass


class CollectionCreationError(VectorStoreError):
    """Raised when collection creation fails."""

    pass


class CollectionDeletionError(VectorStoreError):
    """Raised when collection deletion fails."""

    pass


class DocumentAdditionError(VectorStoreError):
    """Raised when document addition fails."""

    pass


class DocumentQueryError(VectorStoreError):
    """Raised when document query fails."""

    pass


class VectorStoreConnectionError(VectorStoreError):
    """Raised when connection to vector store fails."""

    pass


class CollectionAlreadyExistsError(VectorStoreError):
    """Raised when trying to create a collection that already exists."""

    pass


class ChromaEmbeddingFunction:
    """Custom embedding function adapter for ChromaDB."""

    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize the embedding function adapter.

        Args:
            embedding_service: EmbeddingService instance for generating embeddings
        """
        self.embedding_service = embedding_service

    def __call__(self, input: list[str]) -> list[list[float]]:
        """
        Generate embeddings for the given texts.

        Args:
            input: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_service.get_embeddings(input)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings for ChromaDB: {e}")
            raise VectorStoreError(f"Embedding generation failed: {e}")


class ChromaVectorStore:
    """
    ChromaDB-based vector store with persistent storage and robust error handling.

    Features:
    - Persistent storage using ChromaDB PersistentClient
    - Collection management with library ID normalization
    - Integration with EmbeddingService
    - Batch operations for performance
    - Health checks and statistics
    - Comprehensive error handling with custom exceptions
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        embedding_service: EmbeddingService | None = None,
        chroma_settings: dict[str, Any] | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the ChromaDB vector store.

        Args:
            persist_directory: Directory for persistent storage (uses config default if None)
            embedding_service: EmbeddingService instance (creates global instance if None)
            chroma_settings: Additional ChromaDB configuration settings
            max_retries: Maximum number of retry attempts for operations
            retry_delay: Initial delay between retries in seconds
        """
        self.persist_directory = Path(persist_directory or settings.chroma_db_path)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize embedding service
        self.embedding_service = embedding_service or get_embedding_service()
        # Note: We'll use ChromaDB's default embedding function for now
        # self.embedding_function = ChromaEmbeddingFunction(self.embedding_service)
        self.embedding_function = None

        # Configure ChromaDB settings
        self.chroma_settings = ChromaSettings(
            persist_directory=str(self.persist_directory),
            allow_reset=True,
            anonymized_telemetry=False,
            **(chroma_settings or {}),
        )

        # Initialize ChromaDB client with retry logic
        self.client = self._initialize_client()

        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ChromaVectorStore initialized with persist directory: {self.persist_directory}"
        )

    def _initialize_client(self) -> chromadb.PersistentClient:
        """
        Initialize ChromaDB client with retry logic.

        Returns:
            Initialized ChromaDB PersistentClient

        Raises:
            VectorStoreConnectionError: If connection fails after all retries
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Attempting to initialize ChromaDB client (attempt {attempt + 1})"
                )
                client = chromadb.PersistentClient(settings=self.chroma_settings)

                # Test the connection
                client.heartbeat()
                logger.info("ChromaDB client initialized successfully")
                return client

            except Exception as e:
                last_error = e
                logger.warning(
                    f"ChromaDB client initialization attempt {attempt + 1} failed: {e}"
                )

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)

        raise VectorStoreConnectionError(
            f"Failed to initialize ChromaDB client after {self.max_retries} attempts: {last_error}"
        )

    def _normalize_collection_name(self, library_id: str) -> str:
        """
        Normalize library ID to ChromaDB-compatible collection name.

        Args:
            library_id: Library identifier

        Returns:
            Normalized collection name
        """
        # Handle empty input
        if not library_id or not library_id.strip():
            return "default"

        # Remove leading/trailing whitespace
        library_id = library_id.strip()

        # Ensure it starts with '/' for validation
        if not library_id.startswith("/"):
            library_id = "/" + library_id

        # Try to normalize using settings method, but handle validation errors gracefully
        try:
            normalized_id = settings.normalize_library_id(library_id)
        except ValueError:
            # If validation fails, do basic normalization ourselves
            # Replace invalid characters with underscores
            library_id = re.sub(r"[^a-zA-Z0-9_/.-]", "_", library_id)
            # Normalize leading slash
            if not library_id.startswith("/"):
                library_id = "/" + library_id
            normalized_id = library_id

        # Convert to ChromaDB-compatible format
        # Remove leading slash and convert to lowercase
        collection_name = normalized_id.lstrip("/").lower()

        # Replace invalid characters with underscores
        collection_name = re.sub(r"[^a-z0-9-]", "-", collection_name)

        # Replace multiple consecutive hyphens with single hyphen
        collection_name = re.sub(r"-+", "-", collection_name)

        # Remove leading/trailing hyphens
        collection_name = collection_name.strip("-")

        # Ensure it's not empty
        if not collection_name:
            collection_name = "default"

        # Ensure it starts with a letter
        if collection_name[0].isdigit():
            collection_name = f"lib-{collection_name}"

        # Truncate if too long (ChromaDB limit is 63)
        if len(collection_name) > 63:
            collection_name = collection_name[:60] + "..."

        logger.debug(
            f"Normalized library ID '{library_id}' to collection name '{collection_name}'"
        )
        return collection_name

    def _retry_operation(self, operation_func, *args, **kwargs):
        """
        Execute an operation with retry logic.

        Args:
            operation_func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the operation

        Raises:
            VectorStoreError: If operation fails after all retries
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return operation_func(*args, **kwargs)

            except (NotFoundError, ChromaError):
                # These are often not retryable
                raise

            except Exception as e:
                last_error = e
                logger.warning(f"Operation attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)

        raise VectorStoreError(
            f"Operation failed after {self.max_retries} attempts: {last_error}"
        )

    def create_collection(
        self,
        library_id: str,
        metadata: dict[str, Any] | None = None,
        get_or_create: bool = False,
    ) -> chromadb.Collection:
        """
        Create a new collection for a library.

        Args:
            library_id: Library identifier
            metadata: Optional metadata to attach to the collection
            get_or_create: If True, return existing collection if it exists

        Returns:
            ChromaDB Collection object

        Raises:
            CollectionCreationError: If collection creation fails
            CollectionAlreadyExistsError: If collection already exists and get_or_create=False
        """
        try:
            collection_name = self._normalize_collection_name(library_id)

            # Prepare metadata with library information
            collection_metadata = {
                "library_id": library_id,
                "created_at": time.time(),
                "embedding_model": self.embedding_service.model_name,
                "embedding_dimension": self.embedding_service.embedding_dimension,
            }
            if metadata:
                collection_metadata.update(metadata)

            logger.info(
                f"Creating collection '{collection_name}' for library '{library_id}'"
            )

            def _create():
                kwargs = {
                    "name": collection_name,
                    "metadata": collection_metadata,
                }
                if self.embedding_function is not None:
                    kwargs["embedding_function"] = self.embedding_function
                return self.client.get_or_create_collection(**kwargs)

            collection = self._retry_operation(_create)

            logger.info(f"Collection '{collection_name}' created successfully")
            return collection

        except Exception as e:
            if "already exists" in str(e).lower() and not get_or_create:
                raise CollectionAlreadyExistsError(
                    f"Collection for library '{library_id}' already exists"
                )
            raise CollectionCreationError(
                f"Failed to create collection for library '{library_id}': {e}"
            )

    def get_collection(self, library_id: str) -> chromadb.Collection:
        """
        Get an existing collection by library ID.

        Args:
            library_id: Library identifier

        Returns:
            ChromaDB Collection object

        Raises:
            CollectionNotFoundError: If collection is not found
        """
        try:
            collection_name = self._normalize_collection_name(library_id)

            logger.debug(
                f"Getting collection '{collection_name}' for library '{library_id}'"
            )

            def _get():
                kwargs = {
                    "name": collection_name,
                }
                if self.embedding_function is not None:
                    kwargs["embedding_function"] = self.embedding_function
                return self.client.get_collection(**kwargs)

            collection = self._retry_operation(_get)

            logger.debug(f"Collection '{collection_name}' retrieved successfully")
            return collection

        except (NotFoundError, ChromaError) as e:
            raise CollectionNotFoundError(
                f"Collection for library '{library_id}' not found: {e}"
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to get collection for library '{library_id}': {e}"
            )

    def delete_collection(self, library_id: str) -> bool:
        """
        Delete a collection by library ID.

        Args:
            library_id: Library identifier

        Returns:
            True if collection was deleted, False if it didn't exist

        Raises:
            CollectionDeletionError: If deletion fails for reasons other than non-existence
        """
        try:
            collection_name = self._normalize_collection_name(library_id)

            logger.info(
                f"Deleting collection '{collection_name}' for library '{library_id}'"
            )

            def _delete():
                self.client.delete_collection(name=collection_name)

            try:
                self._retry_operation(_delete)
                logger.info(f"Collection '{collection_name}' deleted successfully")
                return True

            except (NotFoundError, ChromaError):
                logger.info(f"Collection '{collection_name}' did not exist")
                return False

        except Exception as e:
            raise CollectionDeletionError(
                f"Failed to delete collection for library '{library_id}': {e}"
            )

    def list_collections(self) -> list[dict[str, Any]]:
        """
        List all collections with their metadata.

        Returns:
            List of collection information dictionaries
        """
        try:
            logger.debug("Listing all collections")

            def _list():
                return self.client.list_collections()

            collections = self._retry_operation(_list)

            result = []
            for collection in collections:
                try:
                    metadata = collection.metadata or {}
                    result.append(
                        {
                            "id": collection.id,
                            "name": collection.name,
                            "library_id": metadata.get("library_id", collection.name),
                            "metadata": metadata,
                            "count": collection.count(),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Error getting info for collection {collection.name}: {e}"
                    )
                    continue

            logger.debug(f"Found {len(result)} collections")
            return result

        except Exception as e:
            raise VectorStoreError(f"Failed to list collections: {e}")

    def add_documents(
        self,
        library_id: str,
        documents: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        batch_size: int = 100,
    ) -> list[str]:
        """
        Add documents to a collection.

        Args:
            library_id: Library identifier
            documents: List of document texts to add
            ids: Optional list of document IDs (generated if None)
            metadatas: Optional list of metadata dictionaries for each document
            batch_size: Number of documents to process in each batch

        Returns:
            List of document IDs that were added

        Raises:
            CollectionNotFoundError: If collection is not found
            DocumentAdditionError: If document addition fails
        """
        if not documents:
            return []

        try:
            collection = self.get_collection(library_id)

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]

            if len(ids) != len(documents):
                raise DocumentAdditionError(
                    f"Number of IDs ({len(ids)}) must match number of documents ({len(documents)})"
                )

            # Prepare metadatas if not provided
            if metadatas is None:
                metadatas = [{}] * len(documents)
            elif len(metadatas) != len(documents):
                raise DocumentAdditionError(
                    f"Number of metadatas ({len(metadatas)}) must match number of documents ({len(documents)})"
                )

            # Add common metadata
            current_time = time.time()
            for i, metadata in enumerate(metadatas):
                metadata.update(
                    {
                        "added_at": current_time,
                        "library_id": library_id,
                        "embedding_model": self.embedding_service.model_name,
                    }
                )

            logger.info(f"Adding {len(documents)} documents to library '{library_id}'")

            added_ids = []

            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch_end = i + batch_size
                batch_docs = documents[i:batch_end]
                batch_ids = ids[i:batch_end]
                batch_metas = metadatas[i:batch_end]

                logger.debug(
                    f"Processing batch {i // batch_size + 1}: {len(batch_docs)} documents"
                )

                try:
                    # Generate embeddings manually if not using ChromaDB's embedding function
                    if self.embedding_function is None:
                        embeddings = self.embedding_service.get_embeddings(batch_docs)

                        def _add_batch():
                            collection.add(
                                documents=batch_docs,
                                ids=batch_ids,
                                metadatas=batch_metas,
                                embeddings=embeddings.tolist(),
                            )
                    else:

                        def _add_batch():
                            collection.add(
                                documents=batch_docs,
                                ids=batch_ids,
                                metadatas=batch_metas,
                            )

                    self._retry_operation(_add_batch)
                    added_ids.extend(batch_ids)

                except Exception as e:
                    logger.error(f"Failed to add batch {i // batch_size + 1}: {e}")
                    raise DocumentAdditionError(
                        f"Failed to add documents to library '{library_id}': {e}"
                    )

            logger.info(
                f"Successfully added {len(added_ids)} documents to library '{library_id}'"
            )
            return added_ids

        except Exception as e:
            if isinstance(e, (CollectionNotFoundError, DocumentAdditionError)):
                raise
            raise DocumentAdditionError(
                f"Failed to add documents to library '{library_id}': {e}"
            )

    def query_documents(
        self,
        library_id: str,
        query_texts: list[str],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list[Any]]:
        """
        Query documents in a collection.

        Args:
            library_id: Library identifier
            query_texts: List of query texts
            n_results: Number of results to return per query
            where: Optional metadata filter
            where_document: Optional document content filter
            include: List of fields to include in results

        Returns:
            Dictionary containing query results

        Raises:
            CollectionNotFoundError: If collection is not found
            DocumentQueryError: If query fails
        """
        if not query_texts:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        try:
            collection = self.get_collection(library_id)

            # Default include fields if not specified
            if include is None:
                include = ["documents", "metadatas", "distances"]

            logger.debug(
                f"Querying library '{library_id}' with {len(query_texts)} queries, n_results={n_results}"
            )

            def _query():
                return collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where,
                    where_document=where_document,
                    include=include,
                )

            results = self._retry_operation(_query)

            # Log query summary
            total_results = (
                len(results.get("ids", [[]])[0]) if results.get("ids") else 0
            )
            logger.debug(
                f"Query returned {total_results} results for library '{library_id}'"
            )

            return results

        except Exception as e:
            if isinstance(e, CollectionNotFoundError):
                raise
            raise DocumentQueryError(f"Failed to query library '{library_id}': {e}")

    def get_documents(
        self,
        library_id: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list[Any]]:
        """
        Get documents from a collection by ID or filter.

        Args:
            library_id: Library identifier
            ids: Optional list of document IDs to retrieve
            where: Optional metadata filter
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            include: List of fields to include in results

        Returns:
            Dictionary containing document data

        Raises:
            CollectionNotFoundError: If collection is not found
            DocumentQueryError: If retrieval fails
        """
        try:
            collection = self.get_collection(library_id)

            # Default include fields if not specified
            if include is None:
                include = ["documents", "metadatas"]

            logger.debug(f"Getting documents from library '{library_id}'")

            def _get():
                return collection.get(
                    ids=ids,
                    where=where,
                    limit=limit,
                    offset=offset,
                    include=include,
                )

            results = self._retry_operation(_get)

            # Log retrieval summary
            total_results = len(results.get("ids", []))
            logger.debug(
                f"Retrieved {total_results} documents from library '{library_id}'"
            )

            return results

        except Exception as e:
            if isinstance(e, CollectionNotFoundError):
                raise
            raise DocumentQueryError(
                f"Failed to get documents from library '{library_id}': {e}"
            )

    def delete_documents(
        self,
        library_id: str,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> bool:
        """
        Delete documents from a collection.

        Args:
            library_id: Library identifier
            ids: Optional list of document IDs to delete
            where: Optional metadata filter for deletion

        Returns:
            True if documents were deleted, False otherwise

        Raises:
            CollectionNotFoundError: If collection is not found
        """
        try:
            collection = self.get_collection(library_id)

            if not ids and not where:
                logger.warning("No IDs or filter provided for document deletion")
                return False

            logger.debug(f"Deleting documents from library '{library_id}'")

            def _delete():
                collection.delete(ids=ids, where=where)

            self._retry_operation(_delete)

            logger.info(f"Documents deleted from library '{library_id}'")
            return True

        except Exception as e:
            if isinstance(e, CollectionNotFoundError):
                raise
            logger.error(f"Failed to delete documents from library '{library_id}': {e}")
            return False

    def update_documents(
        self,
        library_id: str,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> bool:
        """
        Update documents in a collection.

        Args:
            library_id: Library identifier
            ids: List of document IDs to update
            documents: Optional list of new document texts
            metadatas: Optional list of new metadata dictionaries

        Returns:
            True if documents were updated, False otherwise

        Raises:
            CollectionNotFoundError: If collection is not found
        """
        if not ids:
            return False

        try:
            collection = self.get_collection(library_id)

            if not documents and not metadatas:
                logger.warning("No documents or metadatas provided for update")
                return False

            # Prepare update data
            current_time = time.time()

            if metadatas is not None:
                for metadata in metadatas:
                    metadata.update(
                        {
                            "updated_at": current_time,
                            "library_id": library_id,
                        }
                    )

            logger.debug(f"Updating {len(ids)} documents in library '{library_id}'")

            def _update():
                collection.update(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )

            self._retry_operation(_update)

            logger.info(f"Documents updated in library '{library_id}'")
            return True

        except Exception as e:
            if isinstance(e, CollectionNotFoundError):
                raise
            logger.error(f"Failed to update documents in library '{library_id}': {e}")
            return False

    def count_documents(self, library_id: str) -> int:
        """
        Count documents in a collection.

        Args:
            library_id: Library identifier

        Returns:
            Number of documents in the collection

        Raises:
            CollectionNotFoundError: If collection is not found
        """
        try:
            collection = self.get_collection(library_id)

            def _count():
                return collection.count()

            count = self._retry_operation(_count)

            logger.debug(f"Collection '{library_id}' contains {count} documents")
            return count

        except Exception as e:
            if isinstance(e, CollectionNotFoundError):
                raise
            raise VectorStoreError(
                f"Failed to count documents in library '{library_id}': {e}"
            )

    def health_check(self) -> dict[str, Any]:
        """
        Perform a health check on the vector store.

        Returns:
            Dictionary containing health status information
        """
        health_info = {
            "status": "healthy",
            "persist_directory": str(self.persist_directory),
            "directory_exists": self.persist_directory.exists(),
            "client_connected": False,
            "embedding_service": None,
            "collections_count": 0,
            "total_documents": 0,
            "errors": [],
        }

        try:
            # Check ChromaDB client connection
            self.client.heartbeat()
            health_info["client_connected"] = True

            # Check embedding service
            if self.embedding_service:
                health_info["embedding_service"] = {
                    "model_name": self.embedding_service.model_name,
                    "embedding_dimension": self.embedding_service.embedding_dimension,
                    "cache_stats": self.embedding_service.get_cache_stats(),
                }

            # Count collections and documents
            collections = self.list_collections()
            health_info["collections_count"] = len(collections)
            health_info["total_documents"] = sum(
                collection.get("count", 0) for collection in collections
            )

        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["errors"].append(str(e))

        return health_info

    def get_collection_stats(self, library_id: str) -> dict[str, Any]:
        """
        Get statistics for a specific collection.

        Args:
            library_id: Library identifier

        Returns:
            Dictionary containing collection statistics

        Raises:
            CollectionNotFoundError: If collection is not found
        """
        try:
            collection = self.get_collection(library_id)

            # Get basic info
            def _get_info():
                return {
                    "count": collection.count(),
                    "metadata": collection.metadata or {},
                    "name": collection.name,
                    "id": collection.id,
                }

            info = self._retry_operation(_get_info)

            # Add additional statistics
            stats = {
                "library_id": library_id,
                "collection_name": info["name"],
                "collection_id": info["id"],
                "document_count": info["count"],
                "metadata": info["metadata"],
                "embedding_model": self.embedding_service.model_name,
                "embedding_dimension": self.embedding_service.embedding_dimension,
            }

            return stats

        except Exception as e:
            if isinstance(e, CollectionNotFoundError):
                raise
            raise VectorStoreError(
                f"Failed to get stats for library '{library_id}': {e}"
            )

    def clear_collection(self, library_id: str) -> bool:
        """
        Clear all documents from a collection without deleting it.

        Args:
            library_id: Library identifier

        Returns:
            True if collection was cleared, False otherwise

        Raises:
            CollectionNotFoundError: If collection is not found
        """
        try:
            collection = self.get_collection(library_id)

            # Get all document IDs and delete them
            def _get_all():
                return collection.get(include=[])

            result = self._retry_operation(_get_all)
            all_ids = result.get("ids", [])

            if all_ids:

                def _delete_all():
                    collection.delete(ids=all_ids)

                self._retry_operation(_delete_all)
                logger.info(
                    f"Cleared {len(all_ids)} documents from library '{library_id}'"
                )
            else:
                logger.info(f"Collection '{library_id}' was already empty")

            return True

        except Exception as e:
            if isinstance(e, CollectionNotFoundError):
                raise
            logger.error(f"Failed to clear collection '{library_id}': {e}")
            return False

    def backup_collection(
        self, library_id: str, backup_path: str | Path | None = None
    ) -> Path:
        """
        Create a backup of a collection.

        Args:
            library_id: Library identifier
            backup_path: Optional custom backup path

        Returns:
            Path to the backup file

        Raises:
            CollectionNotFoundError: If collection is not found
        """
        try:
            import json

            collection = self.get_collection(library_id)

            if backup_path is None:
                timestamp = int(time.time())
                backup_filename = (
                    f"{library_id.replace('/', '_')}_backup_{timestamp}.json"
                )
                backup_path = self.persist_directory / "backups" / backup_filename
                backup_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                backup_path = Path(backup_path)

            logger.info(
                f"Creating backup of collection '{library_id}' to {backup_path}"
            )

            # Get all documents
            def _get_all():
                return collection.get(include=["documents", "metadatas"])

            result = self._retry_operation(_get_all)

            # Prepare backup data
            backup_data = {
                "library_id": library_id,
                "backup_time": time.time(),
                "collection_metadata": collection.metadata,
                "embedding_model": self.embedding_service.model_name,
                "documents": result.get("documents", []),
                "metadatas": result.get("metadatas", []),
                "ids": result.get("ids", []),
            }

            # Write backup
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Backup created successfully: {backup_path}")
            return backup_path

        except Exception as e:
            if isinstance(e, CollectionNotFoundError):
                raise
            raise VectorStoreError(f"Failed to backup collection '{library_id}': {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # ChromaDB PersistentClient doesn't need explicit cleanup
        pass


# Global vector store instance
_vector_store: ChromaVectorStore | None = None


def get_vector_store() -> ChromaVectorStore:
    """Get the global vector store instance."""
    global _vector_store

    if _vector_store is None:
        _vector_store = ChromaVectorStore()

    return _vector_store


def reset_vector_store():
    """Reset the global vector store instance."""
    global _vector_store
    _vector_store = None
