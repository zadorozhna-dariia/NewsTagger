setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .

.PHONY: setup lint

# DVC remote is MinIO from docker-compose.yml; credentials come from .env (not committed)
DVC_ENV = set -a && . ./.env && set +a && AWS_ACCESS_KEY_ID=$$MINIO_ROOT_USER AWS_SECRET_ACCESS_KEY=$$MINIO_ROOT_PASSWORD

data-push:
	$(DVC_ENV) uv run dvc push

data-pull:
	$(DVC_ENV) uv run dvc pull

.PHONY: data-push data-pull

# Training reads the features table from Postgres and logs every run to MLflow (docker compose)
train:
	$(DVC_ENV) uv run python src/train.py

register:
	$(DVC_ENV) uv run python src/train.py --register

promote:
	$(DVC_ENV) uv run python src/promote.py $(VERSION)

reproduce:
	$(DVC_ENV) uv run python src/reproduce.py $(VERSION)

.PHONY: train register promote reproduce
