.PHONY: setup test

setup:
	@echo "Setup and install dependencies"
	@uv venv
	@uv pip install -r requirements.txt

test:
	@echo "Running tests"
	@pytest -vv --tb=line --cov-report=term-missing --cov=.