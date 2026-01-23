FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY src/ ./src/
COPY data/processed/train_images.pt data/processed/train_target.pt ./data/processed/

RUN pip install uv && uv sync --frozen

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "fruit_and_vegetable_disease.drift_api:app", "--host", "0.0.0.0", "--port", "8080"]