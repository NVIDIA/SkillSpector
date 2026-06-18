FROM python:3.12-slim-bookworm AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN python -m venv .venv
RUN .venv/bin/pip install --no-cache-dir .

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install --no-install-recommends -y git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /scan

ENTRYPOINT ["skillspector"]
