"""
Pytest configuration and fixtures for doc-server tests.

This module provides comprehensive fixtures for:
- Test data management (repositories, documents, temporary storage)
- Mock objects for external dependencies
- Performance benchmarking helpers
- Sample repositories for integration testing
"""

import os
import shutil
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from doc_server.config import Settings
from doc_server.logging_config import configure_structlog


@pytest.fixture(scope="session", autouse=True)
def configure_structlog_for_tests() -> None:
    """
    Configure structlog for test sessions.

    Uses console output to avoid JSON serialization issues in tests.
    """
    _original_stderr = sys.stderr

    def mock_isatty():
        return False

    original_log_format = os.environ.get("DOC_SERVER_LOG_FORMAT")
    os.environ["DOC_SERVER_LOG_FORMAT"] = "console"

    try:
        configure_structlog()
    finally:
        if original_log_format:
            os.environ["DOC_SERVER_LOG_FORMAT"] = original_log_format
        else:
            os.environ.pop("DOC_SERVER_LOG_FORMAT", None)


@pytest.fixture
def config() -> Settings:
    """Create a Settings instance for testing."""
    return Settings()


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Provide a temporary storage directory for tests.

    This fixture creates a temporary directory that mimics the
    doc-server storage structure (~/.doc-server/) for isolated testing.
    """
    storage_dir = tmp_path / "doc-server-test"
    storage_dir.mkdir(parents=True, exist_ok=True)

    (storage_dir / "chroma.db").mkdir(exist_ok=True)
    (storage_dir / "models").mkdir(exist_ok=True)
    (storage_dir / "libraries").mkdir(exist_ok=True)
    (storage_dir / "config.yaml").touch()

    yield storage_dir

    if storage_dir.exists():
        shutil.rmtree(storage_dir)


@pytest.fixture
def sample_repository(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Create a sample Python repository for testing.

    This fixture creates a minimal repository structure with:
    - Python source files
    - Documentation files
    - .gitignore file
    - setup.py/pyproject.toml
    """
    repo_dir = tmp_path / "sample-repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    src_dir = repo_dir / "sample_package"
    src_dir.mkdir(exist_ok=True)

    (src_dir / "__init__.py").write_text(
        '"""Sample package for testing."""\n\n__version__ = "1.0.0"\n'
    )
    (src_dir / "module_a.py").write_text(
        '''"""Module A with sample functions."""

def hello_world() -> str:
    """Return a greeting message."""
    return "Hello, World!"


def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


class Calculator:
    """A simple calculator class."""

    def __init__(self, initial_value: int = 0):
        self.value = initial_value

    def add(self, x: int) -> int:
        """Add a value to the calculator."""
        self.value += x
        return self.value

    def multiply(self, x: int) -> int:
        """Multiply the calculator value."""
        self.value *= x
        return self.value
'''
    )

    (src_dir / "module_b.py").write_text(
        '''"""Module B with data processing functions."""

from typing import List, Dict, Any
import json


def process_list(items: List[Any]) -> List[str]:
    """Process a list of items into strings."""
    return [str(item) for item in items]


def filter_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Filter a dictionary to only include specified keys."""
    return {k: d[k] for k in keys if k in d}


class DataProcessor:
    """Data processing utility class."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def get_fields(self, field_name: str) -> List[Any]:
        """Extract a specific field from all records."""
        return [record.get(field_name) for record in self.data]

    def count_by_field(self, field_name: str) -> Dict[Any, int]:
        """Count occurrences of each value in a field."""
        counts = {}
        for record in self.data:
            value = record.get(field_name)
            counts[value] = counts.get(value, 0) + 1
        return counts
'''
    )

    docs_dir = repo_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    (docs_dir / "README.md").write_text(
        """# Sample Repository

This is a sample repository for testing doc-server functionality.

## Installation

```bash
pip install sample-package
```

## Usage

```python
from sample_package import Calculator

calc = Calculator()
calc.add(10)
result = calc.multiply(2)  # Returns 20
```

## API Reference

See the `sample_package` module for detailed API documentation.
"""
    )

    (docs_dir / "api.md").write_text(
        """# API Reference

## Calculator

### Methods

- `__init__(initial_value: int = 0)` - Initialize with a value
- `add(x: int) -> int` - Add a value
- `multiply(x: int) -> int` - Multiply the value

## DataProcessor

### Methods

- `get_fields(field_name: str) -> List[Any]` - Extract field values
- `count_by_field(field_name: str) -> Dict[Any, int]` - Count field occurrences
"""
    )

    (repo_dir / ".gitignore").write_text(
        """__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
"""
    )

    (repo_dir / "setup.py").write_text(
        '''"""Setup configuration for sample package."""

from setuptools import setup, find_packages

setup(
    name="sample-package",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
'''
    )

    yield repo_dir


@pytest.fixture
def algorithms_repository(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Create a sample algorithms repository for search testing.

    This fixture creates a repository with algorithm implementations
    for testing search functionality and hybrid search ranking.
    """
    repo_dir = tmp_path / "algorithms-repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    sorting_dir = repo_dir / "sorting"
    sorting_dir.mkdir(exist_ok=True)

    (sorting_dir / "__init__.py").write_text(
        '''"""Sorting algorithms module."""

from typing import List, Callable

SortFunction = Callable[[List[int]], List[int]]


def bubble_sort(arr: List[int]) -> List[int]:
    """Sort a list using bubble sort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list of integers

    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def quick_sort(arr: List[int]) -> List[int]:
    """Sort a list using quick sort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list of integers

    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr: List[int]) -> List[int]:
    """Sort a list using merge sort algorithm.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list of integers

    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
'''
    )

    searching_dir = repo_dir / "searching"
    searching_dir.mkdir(exist_ok=True)

    (searching_dir / "__init__.py").write_text(
        '''"""Searching algorithms module."""

from typing import List, Optional


def linear_search(arr: List[int], target: int) -> Optional[int]:
    """Find target in array using linear search.

    Args:
        arr: List to search
        target: Value to find

    Returns:
        Index of target if found, None otherwise

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    for i, value in enumerate(arr):
        if value == target:
            return i
    return None


def binary_search(arr: List[int], target: int) -> Optional[int]:
    """Find target in sorted array using binary search.

    Args:
        arr: Sorted list to search
        target: Value to find

    Returns:
        Index of target if found, None otherwise

    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return None
'''
    )

    data_structures_dir = repo_dir / "data_structures"
    data_structures_dir.mkdir(exist_ok=True)

    (data_structures_dir / "__init__.py").write_text(
        '''"""Data structures module."""

from typing import Any, Optional, List


class Node:
    """Node for linked list and tree structures."""

    def __init__(self, value: Any, next: Optional['Node'] = None):
        self.value = value
        self.next = next


class LinkedList:
    """Singly linked list implementation."""

    def __init__(self):
        self.head: Optional[Node] = None
        self._size: int = 0

    def append(self, value: Any) -> None:
        """Add value to end of list."""
        new_node = Node(value)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self._size += 1

    def find(self, value: Any) -> Optional[Node]:
        """Find node with given value."""
        current = self.head
        while current:
            if current.value == value:
                return current
            current = current.next
        return None

    def __len__(self) -> int:
        return self._size


class Stack:
    """LIFO stack implementation."""

    def __init__(self):
        self._items: List[Any] = []

    def push(self, item: Any) -> None:
        """Push item onto stack."""
        self._items.append(item)

    def pop(self) -> Optional[Any]:
        """Pop item from stack."""
        return self._items.pop() if self._items else None

    def peek(self) -> Optional[Any]:
        """View top item without removing."""
        return self._items[-1] if self._items else None

    def is_empty(self) -> bool:
        return len(self._items) == 0


class Queue:
    """FIFO queue implementation."""

    def __init__(self):
        self._items: List[Any] = []

    def enqueue(self, item: Any) -> None:
        """Add item to queue."""
        self._items.append(item)

    def dequeue(self) -> Optional[Any]:
        """Remove and return front item."""
        return self._items.pop(0) if self._items else None

    def is_empty(self) -> bool:
        return len(self._items) == 0
'''
    )

    (repo_dir / "README.md").write_text(
        """# Algorithms Repository

A collection of common algorithms and data structures for testing.

## Contents

- **sorting/**: Sorting algorithms (bubble sort, quick sort, merge sort)
- **searching/**: Searching algorithms (linear search, binary search)
- **data_structures/**: Common data structures (linked list, stack, queue)

## Usage

```python
from sorting import bubble_sort

result = bubble_sort([3, 1, 4, 1, 5, 9, 2, 6])
# Returns: [1, 1, 2, 3, 4, 5, 6, 9]
```
"""
    )

    yield repo_dir


@pytest.fixture
def test_documents() -> list[dict[str, Any]]:
    """
    Provide test documents for search testing.

    Returns a list of documents with content and metadata for
    testing search functionality.
    """
    return [
        {
            "id": "doc1",
            "content": "The quick brown fox jumps over the lazy dog.",
            "file_path": "sample.txt",
            "library_id": "/test-lib",
            "line_numbers": (1, 1),
        },
        {
            "id": "doc2",
            "content": "Python is a high-level programming language with dynamic typing.",
            "file_path": "python/info.md",
            "library_id": "/test-lib",
            "line_numbers": (1, 1),
        },
        {
            "id": "doc3",
            "content": "FastAPI is a modern web framework for building APIs with Python.",
            "file_path": "fastapi/overview.md",
            "library_id": "/fastapi",
            "line_numbers": (1, 1),
        },
        {
            "id": "doc4",
            "content": "Machine learning algorithms can identify patterns in data.",
            "file_path": "ml/algorithms.md",
            "library_id": "/ml",
            "line_numbers": (1, 1),
        },
        {
            "id": "doc5",
            "content": "Binary search algorithm has O(log n) time complexity.",
            "file_path": "algorithms/search.md",
            "library_id": "/algorithms",
            "line_numbers": (5, 10),
        },
    ]


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model for fast tests."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3]] * 32
    mock_model.get_sentence_embedding_dimension.return_value = 384
    return mock_model


@pytest.fixture
def mock_chromadb_collection():
    """Create a mock ChromaDB collection."""
    collection = MagicMock()
    collection.add.return_value = None
    collection.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "documents": [["content1", "content2"]],
        "metadatas": [[{"file_path": "a.txt"}, {"file_path": "b.txt"}]],
        "distances": [[0.1, 0.2]],
    }
    collection.get.return_value = {
        "ids": ["doc1", "doc2"],
        "documents": ["content1", "content2"],
        "metadatas": [{"file_path": "a.txt"}, {"file_path": "b.txt"}],
    }
    collection.delete.return_value = None
    return collection


@pytest.fixture
def performance_monitor():
    """Helper class for performance benchmarking in tests."""
    import time

    class PerformanceMonitor:
        def __init__(self):
            self.start_times: dict[str, float] = {}

        def start(self, name: str) -> None:
            self.start_times[name] = time.perf_counter()

        def stop(self, name: str) -> float:
            if name not in self.start_times:
                raise ValueError(f"Timer '{name}' was not started")
            elapsed = time.perf_counter() - self.start_times[name]
            del self.start_times[name]
            return elapsed

        def measure(self, name: str, func, *args, **kwargs):
            self.start(name)
            result = func(*args, **kwargs)
            elapsed = self.stop(name)
            return result, elapsed

    return PerformanceMonitor()


@pytest.fixture
def test_config():
    """Provide pytest configuration options for tests."""
    return {
        "verbose": True,
        "tb": "short",
        "cov": True,
        "cov_fail_under": 90,
        "markers": ["unit", "integration", "performance", "security", "mcp"],
    }


@pytest.fixture
def isolated_environment(monkeypatch):
    """Create an isolated test environment with patched paths."""

    with tempfile.TemporaryDirectory() as tmpdir:
        test_home = Path(tmpdir) / "test_home"
        test_home.mkdir()

        monkeypatch.setenv("HOME", str(test_home))
        monkeypatch.setenv("DOC_SERVER_HOME", str(test_home / ".doc-server"))
        monkeypatch.setenv("DOC_SERVER_STORAGE", str(test_home / ".doc-server"))

        yield {
            "home": test_home,
            "config": test_home / ".doc-server",
            "storage": test_home / ".doc-server",
        }
