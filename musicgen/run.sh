#!/usr/bin/env bash
set -euo pipefail

docker run \
  --gpus all \
  --rm \
  -p 8080:8080 \
  ace-step-sagemaker:0.0.1
