FROM python:3.12-slim

WORKDIR /app

# Copy project metadata first (for caching)
COPY pyproject.toml uv.lock README.md ./

# Copy source code before uv sync (uv installs the project itself)
COPY src ./src

RUN pip install --no-cache-dir uv && uv sync --frozen

COPY models ./models

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "fruit_and_vegetable_disease.api:app", "--host", "0.0.0.0", "--port", "8000"]
