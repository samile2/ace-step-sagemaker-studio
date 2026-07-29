#!/usr/bin/env bash
set -euo pipefail

docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --load \
  --progress=plain \
  -t ace-step-sagemaker:0.0.1 \
  -f dockerfile \
  .
