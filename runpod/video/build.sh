#!/usr/bin/env bash
# Build the RunPod video image (ComfyUI, runpod/comfy/Dockerfile): build.sh <tag>. See runpod/README.md.
set -euo pipefail
tag=${1:?usage: build.sh <image tag>}
cd "$(dirname "$0")/../.."
scripts/stage_worker.sh runpod/video
docker build -f runpod/comfy/Dockerfile \
    --build-arg QUEUE=video \
    --build-arg IMAGE_ID="$(cat /proc/sys/kernel/random/uuid)" \
    -t "$tag" .
