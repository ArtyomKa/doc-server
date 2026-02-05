"""
Hybrid search implementation combining keyword and semantic search.

Provides hybrid search capabilities combining vector similarity
with keyword matching, configurable result ranking, and
metadata enrichment.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from .embedding_service import EmbeddingService, get_embedding_service
from .vector_store import ChromaVectorStore, get_vector_store

logger = logging.getLogger(__name__)


class HybridSearchError(Exception):
    """Base exception for hybrid search errors."""

    pass


class HybridSearchValidationError(HybridSearchError):
    """Raised when search input validation fails."""

    pass


class HybridSearchQueryError(HybridSearchError):
    """Raised when search query execution fails."""

    pass


@dataclass
class SearchResult:
    """
    Search result with metadata and relevance scores.

    Attributes:
        content: Document content text
        file_path: Path to the source file
        library_id: Library identifier
        relevance_score: Overall relevance score (0-1)
        vector_score: Vector similarity score (0-1)
        keyword_score: Keyword match score (0-1)
        line_numbers: Optional tuple of (start_line, end_line)
        metadata: Additional metadata dictionary
    """

    content: str
    file_path: str
    library_id: str
    relevance_score: float
    vector_score: float
    keyword_score: float
    line_numbers: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "library_id": self.library_id,
            "relevance_score": self.relevance_score,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "line_numbers": self.line_numbers,
            "metadata": self.metadata,
        }


class BM25Scorer:
    """
    BM25 keyword scoring for document retrieval.

    Uses BM25 algorithm for ranking documents by keyword relevance.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25 scorer.

        Args:
            k1: Controls term saturation (higher = less saturation)
            b: Controls document length normalization
            epsilon: Minimum IDF value to prevent division by zero
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Document corpus statistics
        self.corpus_size = 0
        self.avg_doc_length: float = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_lengths: list[int] = []

    def build_index(self, documents: list[str]) -> None:
        """
        Build BM25 index from document corpus.

        Args:
            documents: List of document texts
        """
        if not documents:
            self.corpus_size = 0
            return

        self.corpus_size = len(documents)
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / self.corpus_size

        # Calculate document frequencies
        self.doc_freqs = {}
        for doc in documents:
            tokens = set(doc.lower().split())
            for token in tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # Calculate IDF scores
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
            self.idf[token] = max(idf, self.epsilon)

        logger.debug(
            f"BM25 index built with {self.corpus_size} documents, "
            f"{len(self.doc_freqs)} unique terms"
        )

    def score(self, query: str, document: str) -> float:
        """
        Score a document for a query using BM25.

        Args:
            query: Query text
            document: Document text

        Returns:
            BM25 relevance score
        """
        if self.corpus_size == 0 or self.avg_doc_length == 0:
            return 0.0

        query_tokens = query.lower().split()
        doc_tokens = document.lower().split()

        if not query_tokens or not doc_tokens:
            return 0.0

        # Calculate term frequencies for document
        term_freqs: dict[str, int] = {}
        for token in doc_tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1

        # Calculate BM25 score
        doc_length = len(doc_tokens)
        normalization = 1 - self.b + (self.b * doc_length / self.avg_doc_length)

        score = 0.0
        for token in query_tokens:
            tf = term_freqs.get(token, 0)
            idf = self.idf.get(token, self.epsilon)

            numerator = tf * (self.k1 + 1)
            denominator = tf + (self.k1 * normalization)
            score += idf * (numerator / denominator)

        return score

    def score_batch(self, query: str, documents: list[str]) -> list[float]:
        """
        Score multiple documents for a query.

        Args:
            query: Query text
            documents: List of document texts

        Returns:
            List of BM25 scores for each document
        """
        if not documents:
            return []

        return [self.score(query, doc) for doc in documents]

    def score_terms(self, query_terms: list[str], document: str) -> float:
        """
        Score a document for specific query terms with exact match boost.

        Args:
            query_terms: List of query terms
            document: Document text

        Returns:
            Keyword relevance score with exact term boost
        """
        if not query_terms or not document:
            return 0.0

        doc_lower = document.lower()
        score = 0.0
        exact_matches = 0

        for term in query_terms:
            term_lower = term.lower()
            term_length = len(term)

            # Check for exact match (boost)
            if term_lower in doc_lower:
                exact_matches += 1
                # Boost for exact matches
                score += 1.0 * (1 + term_length / 10)

            # Check for partial matches
            # Split on word boundaries
            words = re.findall(r"\b\w+\b", doc_lower)
            if term_lower in words:
                score += 0.5

        # Normalize by number of query terms
        if query_terms:
            score = score / len(query_terms)

        # Boost for multiple exact matches
        if exact_matches > 1:
            score *= 1.0 + (exact_matches - 1) * 0.2

        return min(score, 1.0)


class HybridSearch:
    """
    Hybrid search combining vector similarity and keyword matching.

    Features:
    - Vector similarity search using EmbeddingService
    - Keyword/term-based search using BM25
    - Weighted result fusion (configurable weights)
    - Metadata enrichment (scores, line numbers, file paths)
    - Error handling for invalid queries and parameters
    """

    def __init__(
        self,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 10,
        embedding_service: EmbeddingService | None = None,
        vector_store: ChromaVectorStore | None = None,
    ):
        """
        Initialize hybrid search service.

        Args:
            vector_weight: Weight for vector similarity scores (default 0.7)
            keyword_weight: Weight for keyword match scores (default 0.3)
            top_k: Number of results to return (default 10)
            embedding_service: Optional EmbeddingService instance
            vector_store: Optional ChromaVectorStore instance
        """
        # Validate weights sum to approximately 1.0
        total_weight = vector_weight + keyword_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Weights ({vector_weight} + {keyword_weight} = {total_weight}) "
                f"do not sum to 1.0, normalizing"
            )
            vector_weight = vector_weight / total_weight
            keyword_weight = keyword_weight / total_weight

        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k

        # Initialize services
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()

        # Initialize keyword scorer
        self.bm25 = BM25Scorer()

        logger.info(
            f"HybridSearch initialized: vector_weight={vector_weight}, "
            f"keyword_weight={keyword_weight}, top_k={top_k}"
        )

    def _validate_query(self, query: str) -> None:
        """
        Validate search query.

        Args:
            query: Query text to validate

        Raises:
            HybridSearchValidationError: If query is invalid
        """
        if not isinstance(query, str):
            raise HybridSearchValidationError(
                f"Query must be a string, got {type(query).__name__}"
            )

        if not query or not query.strip():
            raise HybridSearchValidationError("Query cannot be empty")

        # Check for extremely long queries
        if len(query) > 10000:
            raise HybridSearchValidationError(
                f"Query too long ({len(query)} characters, max 10000)"
            )

        # Check for potential injection patterns
        dangerous_patterns = [
            r"__import__",
            r"__subclasses__",
            r"__bases__",
            r"__getitem__",
            r"\$\([^)]*\)",  # Command substitution
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, query):
                logger.warning(
                    f"Query contains potentially dangerous pattern: {pattern}"
                )

    def _extract_query_terms(self, query: str) -> list[str]:
        """
        Extract meaningful terms from query.

        Args:
            query: Query text

        Returns:
            List of query terms
        """
        # Remove special characters and split on whitespace
        terms = re.findall(r"\b[a-zA-Z0-9_-]+\b", query)

        # Filter out very short terms (less than 2 chars)
        terms = [t for t in terms if len(t) >= 2]

        # Remove common stop words (simple list for efficiency)
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
            "this",
            "but",
            "they",
            "have",
            "had",
            "what",
            "when",
            "where",
            "who",
            "which",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "can",
            "should",
            "now",
        }
        terms = [t for t in terms if t.lower() not in stop_words]

        return list(set(terms))  # Deduplicate

    def _enrich_metadata(
        self,
        document: str,
        file_path: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Enrich metadata with additional information.

        Args:
            document: Document content
            file_path: File path
            metadata: Existing metadata

        Returns:
            Enriched metadata dictionary
        """
        enriched = metadata.copy()

        # Extract line numbers if available
        if "line_start" in metadata and "line_end" in metadata:
            line_numbers = (metadata["line_start"], metadata["line_end"])
            enriched["line_numbers"] = line_numbers
        elif "line_numbers" in metadata:
            enriched["line_numbers"] = metadata["line_numbers"]

        # Add document statistics
        enriched["document_length"] = len(document)
        enriched["word_count"] = len(document.split())

        # Add file path if not already present
        if "file_path" not in enriched:
            enriched["file_path"] = file_path

        return enriched

    def _fusion_rank(
        self,
        vector_scores: dict[str, float],
        keyword_scores: dict[str, float],
    ) -> dict[str, tuple[float, float, float]]:
        """
        Fuse vector and keyword scores using weighted combination.

        Args:
            vector_scores: Dictionary mapping doc_id to vector score
            keyword_scores: Dictionary mapping doc_id to keyword score

        Returns:
            Dictionary mapping doc_id to (combined, vector, keyword) scores
        """
        combined_scores = {}

        # Get all unique document IDs
        all_doc_ids = set(vector_scores.keys()) | set(keyword_scores.keys())

        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            k_score = keyword_scores.get(doc_id, 0.0)

            # Normalize scores to 0-1 range
            v_score = max(0.0, min(1.0, v_score))
            k_score = max(0.0, min(1.0, k_score))

            # Weighted combination
            combined = (self.vector_weight * v_score) + (self.keyword_weight * k_score)

            combined_scores[doc_id] = (combined, v_score, k_score)

        return combined_scores

    def search(
        self,
        query: str,
        library_id: str,
        n_results: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining vector and keyword matching.

        Args:
            query: Search query text
            library_id: Library identifier to search within
            n_results: Number of results to return (uses top_k if None)
            where: Optional metadata filter for ChromaDB

        Returns:
            List of SearchResult objects ranked by relevance

        Raises:
            HybridSearchValidationError: If query is invalid
            HybridSearchQueryError: If search execution fails
        """
        # Validate query
        self._validate_query(query)

        n_results = n_results or self.top_k

        logger.info(
            f"Hybrid search in library '{library_id}': "
            f"query='{query[:50]}...', n_results={n_results}"
        )

        try:
            # Step 1: Vector similarity search
            logger.debug("Performing vector similarity search")
            vector_results = self.vector_store.query_documents(
                library_id=library_id,
                query_texts=[query],
                n_results=n_results * 2,  # Get more for better fusion
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # Step 2: Extract documents and metadata
            documents = vector_results.get("documents", [[]])[0]
            metadatas = vector_results.get("metadatas", [[]])[0]
            distances = vector_results.get("distances", [[]])[0]
            doc_ids = vector_results.get("ids", [[]])[0]

            if not documents:
                logger.info("No documents found in vector search")
                return []

            # Step 3: Build BM25 index for keyword search
            logger.debug("Building BM25 index for keyword search")
            self.bm25.build_index(documents)

            # Step 4: Compute keyword scores
            logger.debug("Computing keyword match scores")
            query_terms = self._extract_query_terms(query)

            keyword_scores_raw = self.bm25.score_batch(query, documents)

            # Also compute exact term match scores for boosting
            exact_term_scores = [
                self.bm25.score_terms(query_terms, doc) for doc in documents
            ]

            # Combine BM25 and exact term scores
            keyword_scores = [
                (0.7 * ks) + (0.3 * es)
                for ks, es in zip(keyword_scores_raw, exact_term_scores, strict=True)
            ]

            # Step 5: Normalize vector scores (distance to similarity)
            vector_scores = []
            for dist in distances:
                # Convert distance to similarity (0-1)
                # Assuming cosine distance: similarity = 1 - distance
                similarity = max(0.0, min(1.0, 1.0 - dist))
                vector_scores.append(similarity)

            # Step 6: Fuse scores
            logger.debug("Fusing vector and keyword scores")
            vector_score_map = {
                doc_ids[i]: vector_scores[i] for i in range(len(doc_ids))
            }
            keyword_score_map = {
                doc_ids[i]: keyword_scores[i] for i in range(len(doc_ids))
            }

            combined_scores = self._fusion_rank(vector_score_map, keyword_score_map)

            # Step 7: Rank and create results
            logger.debug("Ranking results and enriching metadata")
            ranked_results = sorted(
                combined_scores.items(),
                key=lambda x: x[1][0],  # Sort by combined score
                reverse=True,
            )

            # Create SearchResult objects
            search_results = []
            for doc_id, (combined, v_score, k_score) in ranked_results[:n_results]:
                # Find the document in original lists
                idx = doc_ids.index(doc_id)
                document = documents[idx]
                metadata = metadatas[idx]

                # Extract file path from metadata
                file_path = metadata.get("file_path", "")

                # Enrich metadata
                enriched_metadata = self._enrich_metadata(document, file_path, metadata)

                # Get line numbers if available
                line_numbers = enriched_metadata.get("line_numbers")

                # Create SearchResult
                result = SearchResult(
                    content=document,
                    file_path=file_path,
                    library_id=library_id,
                    relevance_score=combined,
                    vector_score=v_score,
                    keyword_score=k_score,
                    line_numbers=line_numbers,
                    metadata=enriched_metadata,
                )

                search_results.append(result)

            logger.info(f"Found {len(search_results)} results for query")
            return search_results

        except HybridSearchValidationError:
            raise
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            raise HybridSearchQueryError(f"Search execution failed: {e}") from None

    def search_pure_vector(
        self,
        query: str,
        library_id: str,
        n_results: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Perform pure vector similarity search (no keyword matching).

        Args:
            query: Search query text
            library_id: Library identifier to search within
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of SearchResult objects
        """
        # Validate query
        self._validate_query(query)

        n_results = n_results or self.top_k

        logger.debug(f"Pure vector search in library '{library_id}'")

        try:
            results = self.vector_store.query_documents(
                library_id=library_id,
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            doc_ids = results.get("ids", [[]])[0]

            if not documents:
                return []

            search_results = []
            for doc, meta, dist, _doc_id in zip(
                documents, metadatas, distances, doc_ids, strict=True
            ):
                similarity = max(0.0, min(1.0, 1.0 - dist))
                file_path = meta.get("file_path", "")

                enriched_meta = self._enrich_metadata(doc, file_path, meta)

                result = SearchResult(
                    content=doc,
                    file_path=file_path,
                    library_id=library_id,
                    relevance_score=similarity,
                    vector_score=similarity,
                    keyword_score=0.0,
                    line_numbers=enriched_meta.get("line_numbers"),
                    metadata=enriched_meta,
                )

                search_results.append(result)

            return search_results

        except Exception as e:
            logger.error(f"Pure vector search failed: {e}", exc_info=True)
            raise HybridSearchQueryError(f"Vector search failed: {e}") from None

    def search_pure_keyword(
        self,
        query: str,
        library_id: str,
        n_results: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Perform pure keyword search (no vector similarity).

        Args:
            query: Search query text
            library_id: Library identifier to search within
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of SearchResult objects
        """
        # Validate query
        self._validate_query(query)

        n_results = n_results or self.top_k

        logger.debug(f"Pure keyword search in library '{library_id}'")

        try:
            # Get documents from vector store
            results = self.vector_store.get_documents(
                library_id=library_id,
                where=where,
                limit=n_results * 3,  # Get more to rank by keyword
                include=["documents", "metadatas"],
            )

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            doc_ids = results.get("ids", [])

            if not documents:
                return []

            # Build BM25 index
            self.bm25.build_index(documents)

            # Score documents
            scores = self.bm25.score_batch(query, documents)

            # Rank by score
            ranked = sorted(
                zip(doc_ids, documents, metadatas, scores, strict=True),
                key=lambda x: x[3],
                reverse=True,
            )[:n_results]

            search_results = []
            for _doc_id, doc, meta, score in ranked:
                file_path = meta.get("file_path", "")
                enriched_meta = self._enrich_metadata(doc, file_path, meta)

                result = SearchResult(
                    content=doc,
                    file_path=file_path,
                    library_id=library_id,
                    relevance_score=score,
                    vector_score=0.0,
                    keyword_score=score,
                    line_numbers=enriched_meta.get("line_numbers"),
                    metadata=enriched_meta,
                )

                search_results.append(result)

            return search_results

        except Exception as e:
            logger.error(f"Pure keyword search failed: {e}", exc_info=True)
            raise HybridSearchQueryError(f"Keyword search failed: {e}") from None

    def get_search_config(self) -> dict[str, Any]:
        """
        Get current search configuration.

        Returns:
            Dictionary containing search configuration
        """
        return {
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "top_k": self.top_k,
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimension": self.embedding_service.embedding_dimension,
            "bm25_k1": self.bm25.k1,
            "bm25_b": self.bm25.b,
            "bm25_epsilon": self.bm25.epsilon,
        }


# Global hybrid search instance
_hybrid_search: HybridSearch | None = None


def get_hybrid_search() -> HybridSearch:
    """Get the global hybrid search instance."""
    global _hybrid_search

    if _hybrid_search is None:
        _hybrid_search = HybridSearch(
            vector_weight=settings.search_vector_weight,
            keyword_weight=settings.search_keyword_weight,
            top_k=settings.default_top_k,
        )

    return _hybrid_search


def reset_hybrid_search() -> None:
    """Reset the global hybrid search instance."""
    global _hybrid_search
    _hybrid_search = None
