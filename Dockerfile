# syntax=docker/dockerfile:1
#
# Reproducible dev / CI image — ADR-0008.
#
# Build:   docker build -t bank-reason-trainer:dev .
# Test:    docker run --rm bank-reason-trainer:dev      # runs pytest
# Shell:   docker run --rm -it bank-reason-trainer:dev bash
#
# Reproducibility contract:
#   * Base image is pinned to an explicit Python point release + Debian
#     codename so a future `python:3.11` retag cannot silently change the
#     runtime. Releases SHOULD additionally pin the @sha256 digest — see
#     ADR-0008 §"Pinning the base image".
#   * Wheels are installed from uv.lock via `uv sync --frozen`. Adding or
#     bumping a dep requires re-running `uv lock` and committing the result —
#     drift between pyproject.toml and uv.lock fails the build.
#   * No `pip install <name>` shortcuts: every Python wheel that lands in the
#     image is pinned in uv.lock with a sha256.
FROM python:3.11.11-slim-bookworm

# uv is the only build-time tool we trust outside the lock — pin it too.
ENV UV_VERSION=0.8.17
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy
ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

WORKDIR /app

# System deps for tkinter (headless smoke), sklearn (libgomp), and matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-tk \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh

# Install Python deps from the lock file in a separate layer so source edits
# don't bust the dependency cache.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra dev

# Copy source after deps so app code changes hit a warm wheel cache.
COPY . .
RUN uv sync --frozen --extra dev

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Default: run the headless test suite (CI mode). Override with
# `python app.py` for GUI or `python -m bank_reason_trainer ...` for CLI.
CMD ["python", "-m", "pytest", "-q", "--tb=short", "tests/"]
