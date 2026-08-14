.PHONY: ci format lint typecheck test verify-model

ci: format lint typecheck test verify-model

format:
	uv run ruff format --check src tests

lint:
	uv run ruff check src tests

typecheck:
	uv run ty check src

test:
	uv run pytest --doctest-modules --cov=parlamonitor --cov-report=term-missing

# Fails if the saved model no longer matches the hand-authored topic names.
verify-model:
	uv run python scripts/verify_topic_model.py
