# Doc Server Makefile
# Provides standard targets for development, testing, and CI

.PHONY: help install dev format lint lint-fix typecheck test test-cov test-fast serve ci clean

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m



help: ## Show this help message
	@echo "$(CYAN)Doc Server Development Commands$(RESET)"
	@echo ""
	@echo "$(WHITE)Available targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(WHITE)Examples:$(RESET)"
	@echo "  make install     # Install dependencies"
	@echo "  make lint        # Run linting checks"
	@echo "  make test        # Run all tests"
	@echo "  make ci          # Run full CI pipeline"

venv: ## Create virtual environment only
	@echo "$(CYAN)Creating virtual environment...$(RESET)"
	@uv venv

install: ## Install dependencies
	@echo "$(CYAN)Installing dependencies...$(RESET)"
	@uv venv --allow-existing
	@uv pip install -e "."

dev: ## Install with dev dependencies
	@echo "$(CYAN)Installing development dependencies...$(RESET)"
	@uv venv --allow-existing
	@uv pip install -e ".[dev]"

format: ## Auto-format code with Black and isort
	@echo "$(CYAN)Formatting code...$(RESET)"
	@echo "$(YELLOW)Running Black...$(RESET)"
	@if uv run black doc_server/ tests/; then \
		echo "$(GREEN)✓ Black formatting complete$(RESET)"; \
	else \
		echo "$(RED)✗ Black formatting failed$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Running isort...$(RESET)"
	@if uv run isort doc_server/ tests/; then \
		echo "$(GREEN)✓ isort formatting complete$(RESET)"; \
	else \
		echo "$(RED)✗ isort formatting failed$(RESET)"; \
		exit 1; \
	fi

lint: ## Run all linting checks (Black, isort, Ruff)
	@echo "$(CYAN)Running linting checks...$(RESET)"
	@echo "$(YELLOW)Checking Black formatting...$(RESET)"
	@if uv run black --check doc_server/ tests/; then \
		echo "$(GREEN)✓ Black formatting OK$(RESET)"; \
	else \
		echo "$(RED)✗ Black formatting issues found$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Checking isort formatting...$(RESET)"
	@if uv run isort --check-only doc_server/ tests/; then \
		echo "$(GREEN)✓ isort formatting OK$(RESET)"; \
	else \
		echo "$(RED)✗ isort formatting issues found$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Running Ruff...$(RESET)"
	@if uv run ruff check doc_server/ tests/; then \
		echo "$(GREEN)✓ Ruff checks passed$(RESET)"; \
	else \
		echo "$(RED)✗ Ruff issues found$(RESET)"; \
		exit 1; \
	fi

lint-fix: ## Fix auto-fixable linting issues
	@echo "$(CYAN)Fixing linting issues...$(RESET)"
	@echo "$(YELLOW)Running Black...$(RESET)"
	@if uv run black doc_server/ tests/; then \
		echo "$(GREEN)✓ Black fixes applied$(RESET)"; \
	else \
		echo "$(RED)✗ Black fixes failed$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Running isort...$(RESET)"
	@if uv run isort doc_server/ tests/; then \
		echo "$(GREEN)✓ isort fixes applied$(RESET)"; \
	else \
		echo "$(RED)✗ isort fixes failed$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Running Ruff with --fix...$(RESET)"
	@if uv run ruff check --fix doc_server/ tests/; then \
		echo "$(GREEN)✓ Ruff fixes applied$(RESET)"; \
	else \
		echo "$(RED)✗ Ruff fixes failed$(RESET)"; \
		exit 1; \
	fi

typecheck: ## Run MyPy type checking
	@echo "$(CYAN)Running MyPy type checking...$(RESET)"
	@if uv run python -m mypy doc_server/; then \
		echo "$(GREEN)✓ Type checking passed$(RESET)"; \
	else \
		echo "$(RED)✗ Type checking failed$(RESET)"; \
		exit 1; \
	fi

test: ## Run all tests
	@echo "$(CYAN)Running tests...$(RESET)"
	@if uv run pytest tests/ --verbose --tb=short; then \
		echo "$(GREEN)✓ All tests passed$(RESET)"; \
	else \
		echo "$(RED)✗ Tests failed$(RESET)"; \
		exit 1; \
	fi

test-cov: ## Run tests with coverage (85% threshold)
	@echo "$(CYAN)Running tests with coverage...$(RESET)"
	@if uv run pytest tests/ \
		--verbose \
		--tb=short \
		--cov=doc_server \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-fail-under=85; then \
		echo "$(GREEN)✓ Tests and coverage passed$(RESET)"; \
	else \
		echo "$(RED)✗ Tests or coverage failed$(RESET)"; \
		exit 1; \
	fi

test-fast: ## Run quick tests only (exclude slow/performance/integration markers)
	@echo "$(CYAN)Running fast tests only...$(RESET)"
	@if uv run pytest tests/ \
		--verbose \
		--tb=short \
		-m "not slow and not performance and not integration"; then \
		echo "$(GREEN)✓ Fast tests passed$(RESET)"; \
	else \
		echo "$(RED)✗ Fast tests failed$(RESET)"; \
		exit 1; \
	fi

build: ## Build wheel and source distribution
	@echo "$(CYAN)Building distributions...$(RESET)"
	@echo "$(YELLOW)Building wheel and sdist with hatch...$(RESET)"
	@if uv tool run hatch build; then \
		echo "$(GREEN)✓ Build complete$(RESET)"; \
		echo "$(WHITE)Artifacts:$(RESET)"; \
		ls -lh dist/*.whl dist/*.tar.gz 2>/dev/null || echo "  No artifacts found"; \
	else \
		echo "$(RED)✗ Build failed$(RESET)"; \
		exit 1; \
	fi

serve: ## Run MCP server
	@echo "$(CYAN)Starting MCP server...$(RESET)"
	@if uv run python -m doc_server.mcp_server; then \
		echo "$(GREEN)✓ MCP server stopped$(RESET)"; \
	else \
		echo "$(RED)✗ MCP server failed$(RESET)"; \
		exit 1; \
	fi

ci: ## Run full CI pipeline (dev → lint → typecheck → test-cov)
	@echo "$(CYAN)Running CI pipeline...$(RESET)"
	@echo "$(YELLOW)Step 1/4: Installing dev dependencies...$(RESET)"
	@$(MAKE) dev
	@echo "$(YELLOW)Step 2/4: Running linting checks...$(RESET)"
	@$(MAKE) lint
	@echo "$(YELLOW)Step 3/4: Running type checking...$(RESET)"
	@$(MAKE) typecheck
	@echo "$(YELLOW)Step 4/4: Running tests with coverage...$(RESET)"
	@$(MAKE) test-cov
	@echo "$(GREEN)✓ CI pipeline completed successfully$(RESET)"

clean: ## Clean cache files and artifacts
	@echo "$(CYAN)Cleaning cache files and artifacts...$(RESET)"
	@echo "$(YELLOW)Removing Python cache files...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(YELLOW)Removing build artifacts...$(RESET)"
	@rm -rf build/ dist/ 2>/dev/null || true
	@rm -f .coverage coverage.xml 2>/dev/null || true
	@echo "$(YELLOW)UV cache can be cleaned manually with: uv cache clean$(RESET)"
	@echo "$(GREEN)✓ Clean completed$(RESET)"