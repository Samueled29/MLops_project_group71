FROM python:3.12-slim

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md 

RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "src/fruit_and_vegetable_disease/train.py"]
