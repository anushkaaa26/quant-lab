.PHONY: setup lint test test-lookahead run clean

setup:
	pip install -r requirements.txt

lint:
	ruff check . --fix

test:
	pytest tests/ -v --disable-warnings

test-lookahead:
	pytest tests/test_lookahead.py -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage
