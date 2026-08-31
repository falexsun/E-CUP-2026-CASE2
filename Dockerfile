# Development image for CPU audit/tests and the generic CLI.
# The official submission runtime is selected by metadata.json and run.py because
# competition-provided CUDA images already contain the shared model weights.
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[models]"

ENTRYPOINT ["ozon-quality"]
