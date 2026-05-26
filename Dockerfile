# Modern Python 3.10 environment
FROM python:3.10-slim

# Install system dependencies (including libglib2.0-0 which is required by opencv-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libgeos-dev

# Install pipenv and project dependencies via the Pipfile (without --deploy since lockfile needs generation)
RUN pip install --no-cache-dir pipenv
COPY Pipfile Pipfile.lock* /work/
WORKDIR /work
RUN pipenv install --system --skip-lock

COPY . /work
