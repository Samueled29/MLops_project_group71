FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "uvicorn", "src.fruit_and_vegetable_disease.api:app", "--host", "0.0.0.0", "--port", "8000"]
