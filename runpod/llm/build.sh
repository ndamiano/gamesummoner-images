#!/usr/bin/env bash
# Build the RunPod llm image: build.sh <tag>. See runpod/README.md "Build and ship".
set -euo pipefail
tag=${1:?usage: build.sh <image tag>}
cd "$(dirname "$0")/../.."
scripts/stage_worker.sh runpod/llm
docker build -f runpod/llm/Dockerfile \
    --build-arg IMAGE_ID="$(cat /proc/sys/kernel/random/uuid)" \
    -t "$tag" .
