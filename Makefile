PYTHON ?= python

.PHONY: setup test lint reproduce reproduce-full clean

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

lint:
	ruff check . && ruff format --check .

test:
	pytest

reproduce:
	$(PYTHON) scripts/reproduce_all.py

reproduce-full:
	$(PYTHON) scripts/reproduce_all.py --full

clean:
	rm -rf data results/leak_models .pytest_cache .ruff_cache
