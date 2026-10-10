.PHONY: install train serve test audit docker-build docker-up docker-down clean

# ---- install ---------------------------------------------------------------
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# ---- ML pipeline -----------------------------------------------------------
train:
	python -m src.ml.tabular_models

# ---- API server ------------------------------------------------------------
serve:
	uvicorn src.api.main:app --reload --port 8000

# ---- Tests -----------------------------------------------------------------
test:
	pytest -q

# ---- Governance ------------------------------------------------------------
audit:
	python -m src.ml.fairness_audit

# ---- Docker ----------------------------------------------------------------
docker-build:
	docker build -t ed-triage-api .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# ---- Clean -----------------------------------------------------------------
clean:
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
