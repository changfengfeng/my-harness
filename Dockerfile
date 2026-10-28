ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra web
RUN useradd --create-home agent && mkdir -p /data && chown agent /data
USER agent
ENV PATH="/app/.venv/bin:$PATH" MYH_WEB_ROOT=/data PORT=8000
EXPOSE 8000
CMD ["myh-web"]
