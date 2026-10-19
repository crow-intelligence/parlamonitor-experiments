.PHONY: ci format lint typecheck test verify-model check-site sync-site serve

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

# The published dashboard is a byte-for-byte mirror of dashboard/ in the site
# repo ($CROW_SITE, default ~/projects/crow-intelligence.github.io). Neither
# target commits or pushes; publishing stays a PR in the site repo.
check-site:
	uv run python scripts/sync_site.py --check

sync-site:
	uv run python scripts/build_dashboard_data.py
	uv run python scripts/sync_site.py

# Local preview. Sends Cache-Control: no-cache, so a reload always picks up a
# new app.js and new data (plain `python -m http.server` lets browsers keep stale copies).
serve:
	uv run python scripts/serve_dashboard.py
