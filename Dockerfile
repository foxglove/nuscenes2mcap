# Modern Python 3.11 environment with uv pre-installed (3.11 is required to compile Shapely 1.8.5.post1 from source)
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install system dependencies (including libglib2.0-0 which is required by opencv-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up the workspace
WORKDIR /work
COPY pyproject.toml /work/

# Synchronize python dependencies onto the system interpreter via uv
RUN uv pip install --system -r pyproject.toml

COPY . /work
