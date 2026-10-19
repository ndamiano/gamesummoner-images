#!/usr/bin/env bash
# Build the RunPod decide image: build.sh <tag>. See runpod/README.md.
set -euo pipefail
tag=${1:?usage: build.sh <image tag>}
cd "$(dirname "$0")/../.."
scripts/stage_worker.sh runpod/decide
docker build -f runpod/decide/Dockerfile \
    --build-arg IMAGE_ID="$(cat /proc/sys/kernel/random/uuid)" \
    -t "$tag" .
