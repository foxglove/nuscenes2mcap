#!/usr/bin/env bash
set -euo pipefail
# This script is an example usage of `convert_to_mcap.py` to convert the nuScenes mini-v1.0 dataset to MCAP.

if [ ! -d "data" ]; then
    echo "data dir does not exist: please create and extract nuScenes data into it."
    exit 1
fi

docker build -t mcap_converter .
mkdir -p output
docker run -t --rm \
    --user $(id -u):$(id -g) \
    -v $(pwd)/data:/data -v $(pwd)/output:/output \
    mcap_converter python3 convert_to_mcap.py --data-dir /data --output-dir /output
