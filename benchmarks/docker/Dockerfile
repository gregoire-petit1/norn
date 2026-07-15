FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN uv pip install --system -e ".[dev]"

WORKDIR /workspace
