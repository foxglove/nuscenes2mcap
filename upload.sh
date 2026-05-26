#!/usr/bin/env bash
set -euo pipefail
# This script runs the upload_mcap.py tool inside the Docker container to upload MCAP scenes to Foxglove.

if [ ! -d "output" ]; then
    echo "output dir does not exist: please run a conversion script first to generate MCAP files."
    exit 1
fi

docker build -t mcap_converter .
docker run -t --rm \
    --user $(id -u):$(id -g) \
    -e FOXGLOVE_API_KEY \
    -v $(pwd)/output:/output \
    mcap_converter python3 upload_mcap.py /output "$@"
