"""
Comprehensive unit tests for Vector Store module.

Tests cover:
- Vector store initialization and configuration
- Collection management (create, get, delete, list)
- Document operations (add, query, get, delete, update)
- Embedding integration with EmbeddingService
- Error handling and retry logic
- Health checks and performance monitoring
- Edge cases and error scenarios
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from doc_server.config import Settings
from doc_server.search.embedding_service import EmbeddingService, EmbeddingError
from doc_server.search.vector_store import (
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


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_settings(temp_dir: Path) -> Settings:
    """Create Settings instance with temporary directory."""
    return Settings(storage_path=temp_dir)


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    """Create a mock EmbeddingService."""
    service = Mock(spec=EmbeddingService)
    service.model_name = "test-model"
    service.embedding_dimension = 384
    service.get_embeddings.return_value = np.random.rand(2, 384)
    service.get_cache_stats.return_value = {
        "cache_enabled": True,
        "cached_embeddings": 10,
        "cache_size_limit": 1000,
    }
    return service


@pytest.fixture
def vector_store(
    temp_dir: Path, mock_embedding_service: EmbeddingService
) -> ChromaVectorStore:
    """Create ChromaVectorStore instance for testing."""
    return ChromaVectorStore(
        persist_directory=temp_dir,
        embedding_service=mock_embedding_service,
        max_retries=2,
        retry_delay=0.1,
    )


@pytest.fixture
def mock_chroma_client() -> Mock:
    """Create a mock ChromaDB client."""
    client = Mock()
    client.heartbeat.return_value = True
    return client


@pytest.fixture
def mock_collection() -> Mock:
    """Create a mock ChromaDB collection."""
    collection = Mock()
    collection.id = "test-collection-id"
    collection.name = "test-collection"
    collection.metadata = {"library_id": "test-library", "created_at": time.time()}
    collection.count.return_value = 0
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    collection.add.return_value = None
    collection.delete.return_value = None
    collection.update.return_value = None
    return collection


@pytest.fixture
def sample_documents() -> List[str]:
    """Sample documents for testing."""
    return [
        "This is a test document about machine learning.",
        "Python is a popular programming language for data science.",
        "Vector databases enable efficient similarity search.",
        "Embeddings represent text as numerical vectors.",
    ]


@pytest.fixture
def sample_metadata() -> List[Dict[str, Any]]:
    """Sample metadata for testing."""
    return [
        {"source": "doc1.txt", "chapter": "introduction", "page": 1},
        {"source": "doc2.txt", "chapter": "programming", "page": 5},
        {"source": "doc3.txt", "chapter": "databases", "page": 10},
        {"source": "doc4.txt", "chapter": "embeddings", "page": 15},
    ]


class TestChromaEmbeddingFunction:
    """Tests for ChromaEmbeddingFunction."""

    def test_embedding_function_initialization(
        self, mock_embedding_service: EmbeddingService
    ):
        """Test embedding function initialization."""
        func = ChromaEmbeddingFunction(mock_embedding_service)
        assert func.embedding_service == mock_embedding_service

    def test_embedding_function_call_success(
        self, mock_embedding_service: EmbeddingService
    ):
        """Test successful embedding generation."""
        func = ChromaEmbeddingFunction(mock_embedding_service)
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_embedding_service.get_embeddings.return_value = mock_embeddings

        result = func(["text1", "text2"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_embedding_service.get_embeddings.assert_called_once_with(
            ["text1", "text2"]
        )

    def test_embedding_function_call_error(
        self, mock_embedding_service: EmbeddingService
    ):
        """Test embedding function error handling."""
        func = ChromaEmbeddingFunction(mock_embedding_service)
        mock_embedding_service.get_embeddings.side_effect = EmbeddingError("Test error")

        with pytest.raises(VectorStoreError, match="Embedding generation failed"):
            func(["text1"])


class TestChromaVectorStoreInitialization:
    """Tests for ChromaVectorStore initialization."""

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_initialization_success(
        self,
        mock_client_class,
        temp_dir: Path,
        mock_embedding_service: EmbeddingService,
    ):
        """Test successful vector store initialization."""
        mock_client = Mock()
        mock_client.heartbeat.return_value = True
        mock_client_class.return_value = mock_client

        store = ChromaVectorStore(
            persist_directory=temp_dir,
            embedding_service=mock_embedding_service,
        )

        assert store.persist_directory == temp_dir
        assert store.embedding_service == mock_embedding_service
        assert store.max_retries == 3
        assert store.retry_delay == 1.0
        mock_client_class.assert_called_once()
        mock_client.heartbeat.assert_called_once()

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_initialization_with_retry_success(
        self,
        mock_client_class,
        temp_dir: Path,
        mock_embedding_service: EmbeddingService,
    ):
        """Test initialization with retry on first failure."""
        mock_client = Mock()
        mock_client.heartbeat.return_value = True
        mock_client_class.side_effect = [Exception("Connection failed"), mock_client]

        store = ChromaVectorStore(
            persist_directory=temp_dir,
            embedding_service=mock_embedding_service,
            max_retries=3,
            retry_delay=0.01,
        )

        assert mock_client_class.call_count == 2

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    @patch("doc_server.search.vector_store.time.sleep")
    def test_initialization_retry_exhausted(
        self,
        mock_sleep,
        mock_client_class,
        temp_dir: Path,
        mock_embedding_service: EmbeddingService,
    ):
        """Test initialization when all retries are exhausted."""
        mock_client_class.side_effect = Exception("Persistent connection failure")

        with pytest.raises(
            VectorStoreConnectionError, match="Failed to initialize ChromaDB client"
        ):
            ChromaVectorStore(
                persist_directory=temp_dir,
                embedding_service=mock_embedding_service,
                max_retries=2,
                retry_delay=0.01,
            )

    def test_initialization_creates_directory(
        self, temp_dir: Path, mock_embedding_service: EmbeddingService
    ):
        """Test that initialization creates persist directory."""
        persist_dir = temp_dir / "chroma_test"
        assert not persist_dir.exists()

        with patch(
            "doc_server.search.vector_store.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.heartbeat.return_value = True
            mock_client_class.return_value = mock_client

            ChromaVectorStore(
                persist_directory=persist_dir,
                embedding_service=mock_embedding_service,
            )

        assert persist_dir.exists()


class TestCollectionNameNormalization:
    """Tests for collection name normalization."""

    def test_normalize_simple_library_id(self, vector_store: ChromaVectorStore):
        """Test normalization of simple library ID."""
        result = vector_store._normalize_collection_name("pandas")
        assert result == "pandas"

    def test_normalize_library_id_with_slash(self, vector_store: ChromaVectorStore):
        """Test normalization of library ID with slash."""
        result = vector_store._normalize_collection_name("/pandas")
        assert result == "pandas"

    def test_normalize_library_id_with_path(self, vector_store: ChromaVectorStore):
        """Test normalization of library ID with path."""
        result = vector_store._normalize_collection_name("pandas/v1.2.3")
        assert result == "pandas-v1-2-3"

    def test_normalize_library_id_with_special_characters(
        self, vector_store: ChromaVectorStore
    ):
        """Test normalization of library ID with special characters."""
        result = vector_store._normalize_collection_name("my_library@2023#test")
        assert result == "my-library-2023-test"

    def test_normalize_library_id_empty_string(self, vector_store: ChromaVectorStore):
        """Test normalization of empty string."""
        result = vector_store._normalize_collection_name("")
        assert result == "default"

    def test_normalize_library_id_starts_with_number(
        self, vector_store: ChromaVectorStore
    ):
        """Test normalization of library ID starting with number."""
        result = vector_store._normalize_collection_name("123library")
        assert result == "lib-123library"

    def test_normalize_library_id_very_long(self, vector_store: ChromaVectorStore):
        """Test normalization of very long library ID."""
        long_id = "a" * 100
        result = vector_store._normalize_collection_name(long_id)
        assert len(result) <= 63
        assert result.endswith("...")

    def test_normalize_library_id_with_consecutive_special_chars(
        self, vector_store: ChromaVectorStore
    ):
        """Test normalization with consecutive special characters."""
        result = vector_store._normalize_collection_name("library@@@test")
        assert result == "library-test"


class TestRetryOperation:
    """Tests for retry operation logic."""

    def test_retry_operation_success(self, vector_store: ChromaVectorStore):
        """Test successful operation on first attempt."""
        mock_func = Mock(return_value="success")
        result = vector_store._retry_operation(mock_func, "arg1", kwarg1="value1")
        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_retry_operation_with_retry(self, vector_store: ChromaVectorStore):
        """Test operation succeeds after retry."""
        mock_func = Mock(side_effect=[Exception("First failure"), "success"])

        with patch("doc_server.search.vector_store.time.sleep"):
            result = vector_store._retry_operation(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    def test_retry_operation_exhausted(self, vector_store: ChromaVectorStore):
        """Test operation fails after all retries."""
        mock_func = Mock(side_effect=Exception("Persistent failure"))

        with patch("doc_server.search.vector_store.time.sleep"):
            with pytest.raises(VectorStoreError, match="Operation failed after"):
                vector_store._retry_operation(mock_func)

    def test_retry_operation_not_retryable_error(self, vector_store: ChromaVectorStore):
        """Test that non-retryable errors are raised immediately."""
        from chromadb.errors import NotFoundError

        mock_func = Mock(side_effect=NotFoundError("Not found"))

        with pytest.raises(NotFoundError):
            vector_store._retry_operation(mock_func)

        mock_func.assert_called_once()


class TestCollectionManagement:
    """Tests for collection management operations."""

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_create_collection_success(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test successful collection creation."""
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.create_collection("test-library")

        assert result == mock_collection
        mock_client.get_or_create_collection.assert_called_once()
        call_args = mock_client.get_or_create_collection.call_args[1]
        assert call_args["name"] == "test-library"
        assert "library_id" in call_args["metadata"]
        assert call_args["metadata"]["library_id"] == "test-library"

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_create_collection_with_metadata(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test collection creation with custom metadata."""
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        custom_metadata = {"version": "1.0", "description": "Test library"}
        result = vector_store.create_collection(
            "test-library", metadata=custom_metadata
        )

        assert result == mock_collection
        call_args = mock_client.get_or_create_collection.call_args[1]
        assert call_args["metadata"]["version"] == "1.0"
        assert call_args["metadata"]["description"] == "Test library"

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_create_collection_already_exists(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test collection creation when collection already exists."""
        mock_client = Mock()
        mock_client.get_or_create_collection.side_effect = Exception(
            "Collection already exists"
        )
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        with pytest.raises(CollectionAlreadyExistsError, match="already exists"):
            vector_store.create_collection("test-library", get_or_create=False)

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_create_collection_get_or_create(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test collection creation with get_or_create=True."""
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        # Should not raise exception even if collection exists
        result = vector_store.create_collection("test-library", get_or_create=True)
        assert result == mock_collection

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_get_collection_success(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test successful collection retrieval."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.get_collection("test-library")

        assert result == mock_collection
        mock_client.get_collection.assert_called_once_with(
            name="test-library",
            embedding_function=vector_store.embedding_function,
        )

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_get_collection_not_found(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test collection retrieval when collection doesn't exist."""
        from chromadb.errors import NotFoundError

        mock_client = Mock()
        mock_client.get_collection.side_effect = NotFoundError("Collection not found")
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        with pytest.raises(CollectionNotFoundError, match="not found"):
            vector_store.get_collection("nonexistent-library")

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_delete_collection_success(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test successful collection deletion."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.delete_collection("test-library")

        assert result is True
        mock_client.delete_collection.assert_called_once_with(name="test-library")

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_delete_collection_not_found(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test collection deletion when collection doesn't exist."""
        from chromadb.errors import NotFoundError

        mock_client = Mock()
        mock_client.delete_collection.side_effect = NotFoundError(
            "Collection not found"
        )
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.delete_collection("nonexistent-library")

        assert result is False

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_list_collections_success(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test successful collection listing."""
        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_collection]
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.list_collections()

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "test-collection-id"
        assert result[0]["name"] == "test-collection"
        mock_client.list_collections.assert_called_once()

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_list_collections_empty(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test listing collections when none exist."""
        mock_client = Mock()
        mock_client.list_collections.return_value = []
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.list_collections()

        assert result == []


class TestDocumentOperations:
    """Tests for document operations."""

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_add_documents_success(
        self,
        mock_client_class,
        vector_store: ChromaVectorStore,
        mock_collection: Mock,
        sample_documents: List[str],
        sample_metadata: List[Dict[str, Any]],
    ):
        """Test successful document addition."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.add_documents(
            library_id="test-library",
            documents=sample_documents,
            metadatas=sample_metadata,
        )

        assert len(result) == len(sample_documents)
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert call_args["documents"] == sample_documents
        assert len(call_args["ids"]) == len(sample_documents)
        assert len(call_args["metadatas"]) == len(sample_documents)

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_add_documents_with_custom_ids(
        self,
        mock_client_class,
        vector_store: ChromaVectorStore,
        mock_collection: Mock,
        sample_documents: List[str],
    ):
        """Test document addition with custom IDs."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        custom_ids = ["doc1", "doc2", "doc3", "doc4"]
        result = vector_store.add_documents(
            library_id="test-library",
            documents=sample_documents,
            ids=custom_ids,
        )

        assert result == custom_ids
        call_args = mock_collection.add.call_args[1]
        assert call_args["ids"] == custom_ids

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_add_documents_empty_list(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test adding empty document list."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.add_documents("test-library", [])

        assert result == []

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_add_documents_id_mismatch(
        self,
        mock_client_class,
        vector_store: ChromaVectorStore,
        sample_documents: List[str],
    ):
        """Test document addition with mismatched ID count."""
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        with pytest.raises(DocumentAdditionError, match="Number of IDs.*must match"):
            vector_store.add_documents(
                "test-library",
                documents=sample_documents,
                ids=["only_one_id"],
            )

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_add_documents_metadata_mismatch(
        self,
        mock_client_class,
        vector_store: ChromaVectorStore,
        sample_documents: List[str],
    ):
        """Test document addition with mismatched metadata count."""
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        with pytest.raises(
            DocumentAdditionError, match="Number of metadatas.*must match"
        ):
            vector_store.add_documents(
                "test-library",
                documents=sample_documents,
                metadatas=[{"meta": "data"}],  # Only one metadata for multiple docs
            )

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_add_documents_batch_processing(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test document addition in batches."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        # Create more documents than batch size
        documents = [f"Document {i}" for i in range(10)]

        result = vector_store.add_documents(
            "test-library",
            documents,
            batch_size=3,
        )

        assert len(result) == 10
        assert mock_collection.add.call_count == 4  # 3+3+3+1

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_query_documents_success(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test successful document querying."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        query_texts = ["machine learning", "python programming"]
        result = vector_store.query_documents(
            library_id="test-library",
            query_texts=query_texts,
            n_results=5,
        )

        assert "ids" in result
        assert "documents" in result
        assert "metadatas" in result
        assert "distances" in result
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args[1]
        assert call_args["query_texts"] == query_texts
        assert call_args["n_results"] == 5

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_query_documents_empty_list(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test querying with empty query list."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.query_documents("test-library", [])

        assert result == {"ids": [], "documents": [], "metadatas": [], "distances": []}

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_get_documents_success(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test successful document retrieval."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        document_ids = ["doc1", "doc2"]
        result = vector_store.get_documents(
            library_id="test-library",
            ids=document_ids,
        )

        assert "ids" in result
        assert "documents" in result
        assert "metadatas" in result
        mock_collection.get.assert_called_once()
        call_args = mock_collection.get.call_args[1]
        assert call_args["ids"] == document_ids

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_delete_documents_success(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test successful document deletion."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.delete_documents(
            library_id="test-library",
            ids=["doc1", "doc2"],
        )

        assert result is True
        mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2"], where=None)

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_delete_documents_no_filter(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test document deletion with no IDs or filter."""
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.delete_documents("test-library")

        assert result is False
        mock_client.delete.assert_not_called()

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_update_documents_success(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test successful document update."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.update_documents(
            library_id="test-library",
            ids=["doc1", "doc2"],
            documents=["Updated content 1", "Updated content 2"],
            metadatas=[{"updated": True}, {"updated": True}],
        )

        assert result is True
        mock_collection.update.assert_called_once()
        call_args = mock_collection.update.call_args[1]
        assert call_args["ids"] == ["doc1", "doc2"]
        assert call_args["documents"] == ["Updated content 1", "Updated content 2"]
        assert call_args["metadatas"][0]["updated"] is True

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_count_documents_success(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test successful document counting."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_collection.count.return_value = 42
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.count_documents("test-library")

        assert result == 42
        mock_collection.count.assert_called_once()

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_clear_collection_success(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test successful collection clearing."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_collection.get.return_value = {"ids": ["doc1", "doc2", "doc3"]}
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.clear_collection("test-library")

        assert result is True
        mock_collection.get.assert_called_once_with(include=[])
        mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2", "doc3"])


class TestHealthAndStats:
    """Tests for health checks and statistics."""

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_health_check_success(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test successful health check."""
        mock_client = Mock()
        mock_client.heartbeat.return_value = True
        mock_client.list_collections.return_value = [mock_collection]
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.health_check()

        assert result["status"] == "healthy"
        assert result["client_connected"] is True
        assert result["collections_count"] == 1
        assert result["total_documents"] == 0
        assert "embedding_service" in result

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_health_check_connection_failed(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test health check with connection failure."""
        mock_client = Mock()
        mock_client.heartbeat.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.health_check()

        assert result["status"] == "unhealthy"
        assert result["client_connected"] is False
        assert len(result["errors"]) > 0

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_get_collection_stats_success(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test successful collection statistics retrieval."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.get_collection_stats("test-library")

        assert result["library_id"] == "test-library"
        assert result["collection_name"] == "test-collection"
        assert result["document_count"] == 0
        assert result["embedding_model"] == "test-model"
        assert result["embedding_dimension"] == 384


class TestBackupOperations:
    """Tests for backup operations."""

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_backup_collection_success(
        self,
        mock_client_class,
        vector_store: ChromaVectorStore,
        mock_collection: Mock,
        temp_dir: Path,
    ):
        """Test successful collection backup."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_collection.get.return_value = {
            "ids": ["doc1", "doc2"],
            "documents": ["Content 1", "Content 2"],
            "metadatas": [{"meta": 1}, {"meta": 2}],
        }
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        result = vector_store.backup_collection("test-library")

        assert isinstance(result, Path)
        assert result.exists()
        assert result.parent.name == "backups"

        # Verify backup content
        with open(result, "r") as f:
            backup_data = json.load(f)

        assert backup_data["library_id"] == "test-library"
        assert backup_data["documents"] == ["Content 1", "Content 2"]
        assert backup_data["ids"] == ["doc1", "doc2"]

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_backup_collection_custom_path(
        self,
        mock_client_class,
        vector_store: ChromaVectorStore,
        mock_collection: Mock,
        temp_dir: Path,
    ):
        """Test collection backup to custom path."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        custom_path = temp_dir / "custom_backup.json"
        result = vector_store.backup_collection("test-library", backup_path=custom_path)

        assert result == custom_path
        assert custom_path.exists()


class TestGlobalInstance:
    """Tests for global vector store instance management."""

    def test_get_vector_store_creates_instance(self):
        """Test that get_vector_store creates a new instance when none exists."""
        reset_vector_store()
        store = get_vector_store()
        assert isinstance(store, ChromaVectorStore)

    def test_get_vector_store_returns_cached_instance(self):
        """Test that get_vector_store returns the same instance."""
        reset_vector_store()
        store1 = get_vector_store()
        store2 = get_vector_store()
        assert store1 is store2

    def test_reset_vector_store(self):
        """Test that reset_vector_store clears the cached instance."""
        reset_vector_store()
        store1 = get_vector_store()
        reset_vector_store()
        store2 = get_vector_store()
        assert store1 is not store2


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_collection_creation_error(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test collection creation error handling."""
        mock_client = Mock()
        mock_client.get_or_create_collection.side_effect = Exception("Creation failed")
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        with pytest.raises(
            CollectionCreationError, match="Failed to create collection"
        ):
            vector_store.create_collection("test-library")

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_collection_deletion_error(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test collection deletion error handling."""
        mock_client = Mock()
        mock_client.delete_collection.side_effect = Exception("Deletion failed")
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        with pytest.raises(
            CollectionDeletionError, match="Failed to delete collection"
        ):
            vector_store.delete_collection("test-library")

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_document_query_error(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test document query error handling."""
        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()
        mock_client.get_collection.return_value.query.side_effect = Exception(
            "Query failed"
        )
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        with pytest.raises(DocumentQueryError, match="Failed to query library"):
            vector_store.query_documents("test-library", ["query"])


class TestPerformanceAndEdgeCases:
    """Tests for performance characteristics and edge cases."""

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_large_batch_processing(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test processing large document batches."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        # Create large document set
        large_documents = [f"Document {i}" for i in range(1000)]

        start_time = time.time()
        result = vector_store.add_documents(
            "test-library",
            large_documents,
            batch_size=100,
        )
        end_time = time.time()

        assert len(result) == 1000
        assert mock_collection.add.call_count == 10  # 1000 docs / 100 batch_size
        # Should complete reasonably quickly (adjust threshold as needed)
        assert end_time - start_time < 10.0

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_concurrent_operations(
        self, mock_client_class, vector_store: ChromaVectorStore, mock_collection: Mock
    ):
        """Test concurrent operations on the same collection."""
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        # Simulate concurrent adds
        import threading

        results = []

        def add_docs(thread_id):
            docs = [f"Thread {thread_id} Document {i}" for i in range(10)]
            result = vector_store.add_documents("test-library", docs)
            results.append(result)

        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_docs, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 3
        assert all(len(result) == 10 for result in results)

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_malformed_library_ids(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test handling of malformed library IDs."""
        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        malformed_ids = [
            "",
            "   ",
            "library@with#special$chars",
            "a" * 100,  # Very long
            "123starts_with_number",
        ]

        for lib_id in malformed_ids:
            # Should handle gracefully or raise appropriate error
            with pytest.raises((CollectionNotFoundError, VectorStoreError)):
                vector_store.get_collection(lib_id)


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_enter_exit(self, vector_store: ChromaVectorStore):
        """Test context manager enter and exit."""
        with vector_store as store:
            assert store is vector_store
        # Should exit without errors

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_context_manager_cleanup_on_error(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test context manager behavior with exceptions."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        vector_store.client = mock_client

        try:
            with vector_store as store:
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        # Context should handle cleanup gracefully


class TestInitializeClient:
    """Tests for _initialize_client method."""

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_initialize_client_success(
        self, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test successful client initialization."""
        mock_client = Mock()
        mock_client.heartbeat.return_value = True
        mock_client_class.return_value = mock_client

        result = vector_store._initialize_client()

        assert result == mock_client
        mock_client_class.assert_called_once_with(settings=vector_store.chroma_settings)
        mock_client.heartbeat.assert_called_once()

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    @patch("doc_server.search.vector_store.time.sleep")
    def test_initialize_client_with_retry(
        self, mock_sleep, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test client initialization with retry logic."""
        mock_client = Mock()
        mock_client.heartbeat.return_value = True
        mock_client_class.side_effect = [Exception("First failure"), mock_client]

        result = vector_store._initialize_client()

        assert result == mock_client
        assert mock_client_class.call_count == 2
        mock_sleep.assert_called_once()

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    @patch("doc_server.search.vector_store.time.sleep")
    def test_initialize_client_exhausted(
        self, mock_sleep, mock_client_class, vector_store: ChromaVectorStore
    ):
        """Test client initialization when retries are exhausted."""
        mock_client_class.side_effect = Exception("Persistent failure")

        with pytest.raises(
            VectorStoreConnectionError, match="Failed to initialize ChromaDB client"
        ):
            vector_store._initialize_client()

        assert mock_client_class.call_count == 2  # max_retries is 2 in test fixture


class TestIntegrationWithEmbeddingService:
    """Integration tests with EmbeddingService."""

    @patch("doc_server.search.vector_store.chromadb.PersistentClient")
    def test_real_embedding_service_integration(
        self, mock_client_class, temp_dir: Path
    ):
        """Test integration with real EmbeddingService."""
        from doc_server.search.embedding_service import get_embedding_service

        mock_client = Mock()
        mock_client.heartbeat.return_value = True
        mock_client_class.return_value = mock_client

        # Use real embedding service (will download model if needed)
        embedding_service = get_embedding_service()

        store = ChromaVectorStore(
            persist_directory=temp_dir,
            embedding_service=embedding_service,
            max_retries=1,
            retry_delay=0.01,
        )

        assert store.embedding_service is embedding_service
        assert store.embedding_function.embedding_service is embedding_service

    def test_embedding_function_service_properties(
        self, mock_embedding_service: EmbeddingService
    ):
        """Test that embedding function correctly delegates to service."""
        func = ChromaEmbeddingFunction(mock_embedding_service)

        # Verify the function has access to service properties
        assert hasattr(func.embedding_service, "model_name")
        assert hasattr(func.embedding_service, "embedding_dimension")
