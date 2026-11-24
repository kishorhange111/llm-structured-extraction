PYTHON ?= python

.PHONY: install install-dev lint format test ocr run clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,serve]"

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	$(PYTHON) -m pytest -v

ocr:
	receiptx-ocr --out data/ocr

run:
	receiptx-run --split test --out results --configs rules

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
