.PHONY: install test lint format eval serve docker

install:        ## environnement de développement (uv)
	uv venv && uv pip install -e ".[data,service,tracking,dev]"

test:           ## tests unitaires (sans réseau)
	uv run pytest -q

lint:           ## PEP 8 / docstrings
	uv run ruff check src tests dags && uv run ruff format --check src tests dags

format:
	uv run ruff format src tests dags && uv run ruff check --fix src tests dags

eval:           ## métriques du consortium sur La Liberté (test), IC95 %
	uv run finlam-prod evaluate --dataset la-liberte --split test --models-dir models

serve:          ## API FastAPI sur http://127.0.0.1:8000/docs
	uv run finlam-prod serve

docker:
	docker build -t finlam-prod .
