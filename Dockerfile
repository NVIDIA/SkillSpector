FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b AS builder

# uv matches UV_VERSION in release.yml; the digest pins the multi-arch index.
COPY --from=ghcr.io/astral-sh/uv:0.10.10@sha256:cbe0a44ba994e327b8fe7ed72beef1aaa7d2c4c795fd406d1dbf328bacb2f1c5 /uv /bin/uv

# Resolve exactly what uv.lock records, with the base image's interpreter.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

RUN apt-get update \
    && apt-get install --no-install-recommends -y git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

# HOME=/tmp keeps git and caches working under any uid, including the host
# uid passed with `docker run --user "$(id -u):$(id -g)"` to write into /scan.
ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin skillspector
WORKDIR /scan
USER 10001

ENTRYPOINT ["skillspector"]
