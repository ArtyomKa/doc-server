"""
Comprehensive unit tests for Hybrid Search module.

Tests cover:
- BM25 keyword scoring
- Vector similarity search
- Hybrid search with fusion
- Exact term boosting
- Metadata enrichment
- Error handling for empty queries and invalid parameters
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from doc_server.config import Settings
from doc_server.search.embedding_service import EmbeddingService
from doc_server.search.hybrid_search import (
    BM25Scorer,
    HybridSearch,
    HybridSearchQueryError,
    HybridSearchValidationError,
    SearchResult,
    get_hybrid_search,
    reset_hybrid_search,
)
from doc_server.search.vector_store import ChromaVectorStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_settings(temp_dir: Path) -> Settings:
    """Create Settings instance with temporary directory."""
    return Settings(storage_path=temp_dir)


@pytest.fixture
def sample_documents() -> list[str]:
    """Generate sample documents for testing."""
    return [
        "Python is a popular programming language for data science.",
        "FastAPI is a modern web framework for building APIs.",
        "Vector embeddings enable semantic search capabilities.",
        "ChromaDB provides efficient vector storage solutions.",
        "Batch processing improves embedding generation performance.",
        "Caching reduces redundant computation overhead.",
        "GPU acceleration speeds up model inference.",
        "Normalization ensures consistent similarity scores.",
        "Warmup optimization improves first inference latency.",
        "BM25 algorithm ranks documents by keyword relevance.",
    ]


@pytest.fixture
def mock_embedding_service() -> Mock:
    """Create a mock EmbeddingService."""
    service = Mock(spec=EmbeddingService)
    service.model_name = "all-MiniLM-L6-v2"
    service.embedding_dimension = 384
    service.get_embeddings.return_value = np.random.rand(5, 384)
    return service


@pytest.fixture
def mock_vector_store(mock_embedding_service: Mock) -> Mock:
    """Create a mock VectorStore."""
    store = Mock(spec=ChromaVectorStore)
    store.embedding_service = mock_embedding_service
    return store


@pytest.fixture
def hybrid_search(
    mock_embedding_service: Mock, mock_vector_store: Mock
) -> HybridSearch:
    """Create HybridSearch instance for testing."""
    return HybridSearch(
        vector_weight=0.7,
        keyword_weight=0.3,
        top_k=10,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )


class TestBM25Scorer:
    """Test suite for BM25 scorer."""

    def test_build_index(self, sample_documents: list[str]):
        """Test building BM25 index from documents."""
        scorer = BM25Scorer()
        scorer.build_index(sample_documents)

        assert scorer.corpus_size == len(sample_documents)
        assert scorer.avg_doc_length > 0
        assert len(scorer.doc_freqs) > 0
        assert len(scorer.idf) > 0
        assert len(scorer.doc_lengths) == len(sample_documents)

    def test_build_empty_index(self):
        """Test building index with empty documents."""
        scorer = BM25Scorer()
        scorer.build_index([])

        assert scorer.corpus_size == 0
        assert scorer.avg_doc_length == 0

    def test_score_document(self, sample_documents: list[str]):
        """Test scoring a single document."""
        scorer = BM25Scorer()
        scorer.build_index(sample_documents)

        query = "Python data science"
        document = sample_documents[0]
        score = scorer.score(query, document)

        assert isinstance(score, float)
        assert score >= 0

    def test_score_batch(self, sample_documents: list[str]):
        """Test scoring multiple documents."""
        scorer = BM25Scorer()
        scorer.build_index(sample_documents)

        query = "FastAPI web framework"
        scores = scorer.score_batch(query, sample_documents[:5])

        assert len(scores) == 5
        assert all(isinstance(s, float) for s in scores)
        assert all(s >= 0 for s in scores)

    def test_score_terms_exact_match(self, sample_documents: list[str]):
        """Test scoring with exact term match boost."""
        scorer = BM25Scorer()
        scorer.build_index(sample_documents)

        query_terms = ["Python", "data"]
        document = sample_documents[0]  # Contains "Python" and "data"

        score = scorer.score_terms(query_terms, document)

        assert isinstance(score, float)
        assert score > 0  # Should have exact matches
        assert score <= 1.0  # Normalized

    def test_score_terms_no_match(self, sample_documents: list[str]):
        """Test scoring with no matching terms."""
        scorer = BM25Scorer()
        scorer.build_index(sample_documents)

        query_terms = ["nonexistent", "terms"]
        document = sample_documents[0]

        score = scorer.score_terms(query_terms, document)

        assert isinstance(score, float)
        assert score == 0.0  # No matches

    def test_custom_parameters(self, sample_documents: list[str]):
        """Test BM25 with custom parameters."""
        scorer = BM25Scorer(k1=2.0, b=0.5, epsilon=0.5)
        scorer.build_index(sample_documents)

        query = "vector search"
        document = sample_documents[2]
        score = scorer.score(query, document)

        assert isinstance(score, float)
        assert score >= 0


class TestSearchResult:
    """Test suite for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            content="Test content",
            file_path="/path/to/file.py",
            library_id="/test-lib",
            relevance_score=0.85,
            vector_score=0.9,
            keyword_score=0.7,
            line_numbers=(10, 20),
            metadata={"key": "value"},
        )

        assert result.content == "Test content"
        assert result.file_path == "/path/to/file.py"
        assert result.library_id == "/test-lib"
        assert result.relevance_score == 0.85
        assert result.vector_score == 0.9
        assert result.keyword_score == 0.7
        assert result.line_numbers == (10, 20)
        assert result.metadata == {"key": "value"}

    def test_search_result_to_dict(self):
        """Test converting SearchResult to dictionary."""
        result = SearchResult(
            content="Test content",
            file_path="/path/to/file.py",
            library_id="/test-lib",
            relevance_score=0.85,
            vector_score=0.9,
            keyword_score=0.7,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["content"] == "Test content"
        assert result_dict["file_path"] == "/path/to/file.py"
        assert result_dict["library_id"] == "/test-lib"
        assert result_dict["relevance_score"] == 0.85
        assert result_dict["vector_score"] == 0.9
        assert result_dict["keyword_score"] == 0.7

    def test_search_result_defaults(self):
        """Test SearchResult with default values."""
        result = SearchResult(
            content="Test content",
            file_path="/path/to/file.py",
            library_id="/test-lib",
            relevance_score=0.5,
            vector_score=0.5,
            keyword_score=0.5,
        )

        assert result.line_numbers is None
        assert result.metadata == {}


class TestHybridSearch:
    """Test suite for HybridSearch."""

    def test_initialization(
        self, mock_embedding_service: Mock, mock_vector_store: Mock
    ):
        """Test HybridSearch initialization."""
        search = HybridSearch(
            vector_weight=0.7,
            keyword_weight=0.3,
            top_k=10,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )

        assert search.vector_weight == 0.7
        assert search.keyword_weight == 0.3
        assert search.top_k == 10
        assert search.embedding_service == mock_embedding_service
        assert search.vector_store == mock_vector_store
        assert isinstance(search.bm25, BM25Scorer)

    def test_initialization_weight_normalization(
        self, mock_embedding_service: Mock, mock_vector_store: Mock
    ):
        """Test weight normalization when weights don't sum to 1.0."""
        search = HybridSearch(
            vector_weight=0.5,
            keyword_weight=0.5,
            top_k=10,
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
        )

        assert search.vector_weight == 0.5
        assert search.keyword_weight == 0.5

    def test_validate_query_valid(self, hybrid_search: HybridSearch):
        """Test validating a valid query."""
        # Should not raise
        hybrid_search._validate_query("test query")

    def test_validate_query_empty(self, hybrid_search: HybridSearch):
        """Test validating an empty query raises error."""
        with pytest.raises(HybridSearchValidationError) as exc_info:
            hybrid_search._validate_query("")

        assert "cannot be empty" in str(exc_info.value).lower()

    def test_validate_query_whitespace(self, hybrid_search: HybridSearch):
        """Test validating whitespace-only query raises error."""
        with pytest.raises(HybridSearchValidationError) as exc_info:
            hybrid_search._validate_query("   ")

        assert "cannot be empty" in str(exc_info.value).lower()

    def test_validate_query_not_string(self, hybrid_search: HybridSearch):
        """Test validating non-string query raises error."""
        with pytest.raises(HybridSearchValidationError) as exc_info:
            hybrid_search._validate_query(["test", "query"])  # type: ignore

        assert "must be a string" in str(exc_info.value).lower()

    def test_validate_query_too_long(self, hybrid_search: HybridSearch):
        """Test validating excessively long query raises error."""
        long_query = "x" * 10001

        with pytest.raises(HybridSearchValidationError) as exc_info:
            hybrid_search._validate_query(long_query)

        assert "too long" in str(exc_info.value).lower()

    def test_extract_query_terms(self, hybrid_search: HybridSearch):
        """Test extracting terms from query."""
        query = "Python FastAPI web framework"
        terms = hybrid_search._extract_query_terms(query)

        assert isinstance(terms, list)
        assert "Python" in terms
        assert "FastAPI" in terms
        assert "web" in terms
        assert "framework" in terms

    def test_extract_query_terms_removes_stop_words(self, hybrid_search: HybridSearch):
        """Test that stop words are removed."""
        query = "the and is a Python for the data"
        terms = hybrid_search._extract_query_terms(query)

        # Should contain content words but not stop words
        assert "Python" in terms
        assert "data" in terms
        assert "the" not in terms
        assert "and" not in terms
        assert "is" not in terms

    def test_enrich_metadata(self, hybrid_search: HybridSearch):
        """Test metadata enrichment."""
        document = "Test document content"
        file_path = "/path/to/file.py"
        metadata = {
            "line_start": 10,
            "line_end": 20,
            "custom_key": "custom_value",
        }

        enriched = hybrid_search._enrich_metadata(document, file_path, metadata)

        assert enriched["line_numbers"] == (10, 20)
        assert enriched["document_length"] == len(document)
        assert enriched["word_count"] == len(document.split())
        assert enriched["file_path"] == file_path
        assert enriched["custom_key"] == "custom_value"

    def test_fusion_rank(self, hybrid_search: HybridSearch):
        """Test result fusion algorithm."""
        vector_scores = {"doc1": 0.9, "doc2": 0.7, "doc3": 0.5}
        keyword_scores = {"doc1": 0.3, "doc2": 0.8, "doc3": 0.6}

        combined = hybrid_search._fusion_rank(vector_scores, keyword_scores)

        assert "doc1" in combined
        assert "doc2" in combined
        assert "doc3" in combined

        for _doc_id, (combined_score, v_score, k_score) in combined.items():
            assert 0 <= combined_score <= 1.0
            assert 0 <= v_score <= 1.0
            assert 0 <= k_score <= 1.0

        # Check weighted combination for doc1
        expected_doc1 = (0.7 * 0.9) + (0.3 * 0.3)
        actual_doc1 = combined["doc1"][0]
        assert abs(expected_doc1 - actual_doc1) < 0.01

    def test_search_empty_results(
        self, hybrid_search: HybridSearch, mock_vector_store: Mock
    ):
        """Test search with no results."""
        mock_vector_store.query_documents.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        results = hybrid_search.search("test query", library_id="/test-lib")

        assert results == []

    def test_search_returns_results(
        self,
        hybrid_search: HybridSearch,
        mock_vector_store: Mock,
        sample_documents: list[str],
    ):
        """Test search returns results."""
        # Mock vector store response
        mock_vector_store.query_documents.return_value = {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [sample_documents[:3]],
            "metadatas": [
                [
                    {"file_path": "/path/1.py", "line_start": 1, "line_end": 10},
                    {"file_path": "/path/2.py", "line_start": 1, "line_end": 10},
                    {"file_path": "/path/3.py", "line_start": 1, "line_end": 10},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }

        results = hybrid_search.search(
            "test query", library_id="/test-lib", n_results=3
        )

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].library_id == "/test-lib"

    def test_search_pure_vector(
        self,
        hybrid_search: HybridSearch,
        mock_vector_store: Mock,
        sample_documents: list[str],
    ):
        """Test pure vector search (no keyword matching)."""
        mock_vector_store.query_documents.return_value = {
            "ids": [["doc1"]],
            "documents": [sample_documents[:1]],
            "metadatas": [[{"file_path": "/path/1.py"}]],
            "distances": [[0.1]],
        }

        results = hybrid_search.search_pure_vector("test query", library_id="/test-lib")

        assert len(results) == 1
        assert results[0].vector_score > 0
        assert results[0].keyword_score == 0.0

    def test_search_pure_keyword(
        self,
        hybrid_search: HybridSearch,
        mock_vector_store: Mock,
        sample_documents: list[str],
    ):
        """Test pure keyword search (no vector similarity)."""
        mock_vector_store.get_documents.return_value = {
            "ids": ["doc1", "doc2"],
            "documents": sample_documents[:2],
            "metadatas": [
                {"file_path": "/path/1.py"},
                {"file_path": "/path/2.py"},
            ],
        }

        results = hybrid_search.search_pure_keyword(
            "test query", library_id="/test-lib"
        )

        assert len(results) == 2
        assert all(r.keyword_score >= 0 for r in results)
        assert all(r.vector_score == 0.0 for r in results)

    def test_get_search_config(self, hybrid_search: HybridSearch):
        """Test getting search configuration."""
        config = hybrid_search.get_search_config()

        assert isinstance(config, dict)
        assert config["vector_weight"] == 0.7
        assert config["keyword_weight"] == 0.3
        assert config["top_k"] == 10
        assert "embedding_model" in config
        assert "embedding_dimension" in config


class TestGlobalFunctions:
    """Test suite for global functions."""

    def test_get_hybrid_search_singleton(self):
        """Test that get_hybrid_search returns singleton."""
        reset_hybrid_search()

        search1 = get_hybrid_search()
        search2 = get_hybrid_search()

        assert search1 is search2

    def test_reset_hybrid_search(self):
        """Test resetting global hybrid search instance."""
        search1 = get_hybrid_search()
        reset_hybrid_search()
        search2 = get_hybrid_search()

        assert search1 is not search2


class TestErrorHandling:
    """Test suite for error handling."""

    def test_search_query_error(
        self,
        hybrid_search: HybridSearch,
        mock_vector_store: Mock,
    ):
        """Test query errors are properly raised."""
        mock_vector_store.query_documents.side_effect = Exception("Database error")

        with pytest.raises(HybridSearchQueryError):
            hybrid_search.search("test query", library_id="/test-lib")

    def test_vector_search_error(
        self,
        hybrid_search: HybridSearch,
        mock_vector_store: Mock,
    ):
        """Test vector search errors are properly raised."""
        mock_vector_store.query_documents.side_effect = Exception("Vector error")

        with pytest.raises(HybridSearchQueryError):
            hybrid_search.search_pure_vector("test query", library_id="/test-lib")

    def test_keyword_search_error(
        self,
        hybrid_search: HybridSearch,
        mock_vector_store: Mock,
    ):
        """Test keyword search errors are properly raised."""
        mock_vector_store.get_documents.side_effect = Exception("Keyword error")

        with pytest.raises(HybridSearchQueryError):
            hybrid_search.search_pure_keyword("test query", library_id="/test-lib")
