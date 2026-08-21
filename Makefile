PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
CLANG_FORMAT ?= $(if $(wildcard .venv/bin/clang-format),.venv/bin/clang-format,clang-format)
BUILD_DIR ?= build
FRONTEND_DIR := src/frontend
PYTHON_PATHS := src tests
CPP_SOURCES := $(shell find dynamo_ src/backend/bindings -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \))

.PHONY: build check clean configure format format-check lint test test-cpp test-python test-frontend

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build $(BUILD_DIR) --parallel

format:
	$(PYTHON) -m ruff format $(PYTHON_PATHS)
	$(CLANG_FORMAT) -i $(CPP_SOURCES)

format-check:
	$(PYTHON) -m ruff format --check $(PYTHON_PATHS)
	$(CLANG_FORMAT) --dry-run --Werror $(CPP_SOURCES)

lint:
	$(PYTHON) -m ruff check $(PYTHON_PATHS)
	npm --prefix $(FRONTEND_DIR) run lint

test-python:
	$(PYTHON) -m pytest

test-cpp: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-frontend:
	npm --prefix $(FRONTEND_DIR) run build

test: test-python test-cpp test-frontend

check: lint format-check test

clean:
	cmake -E remove_directory $(BUILD_DIR)
	cmake -E remove_directory .pytest_cache
	cmake -E remove_directory .ruff_cache
	cmake -E remove_directory $(FRONTEND_DIR)/dist
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	find src tests -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
