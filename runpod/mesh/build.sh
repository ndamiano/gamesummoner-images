#!/usr/bin/env bash
# Build the RunPod mesh image: build.sh <tag>. See runpod/README.md.
set -euo pipefail
tag=${1:?usage: build.sh <image tag>}
cd "$(dirname "$0")/../.."
scripts/stage_worker.sh runpod/mesh
docker build -f runpod/mesh/Dockerfile \
    --build-arg IMAGE_ID="$(cat /proc/sys/kernel/random/uuid)" \
    -t "$tag" .
