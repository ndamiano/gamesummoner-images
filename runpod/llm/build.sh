#!/usr/bin/env bash
# Build the RunPod llm image: build.sh <tag> [--env]. See runpod/README.md "Build and ship".
set -euo pipefail
tag=${1:?usage: build.sh <image tag> [--env]}
cd "$(dirname "$0")/../.."
[ -f runpod/llm/penny_cache_v6.tgz ] || {
    echo "runpod/llm/penny_cache_v6.tgz is missing: fetch it from s3://<llm-volume>/fastboot/" >&2
    exit 1; }
scripts/stage_worker.sh runpod/llm
if [ "${2:-}" = "--env" ]; then
    docker build --target env --output type=local,dest=runpod/llm/out -f runpod/llm/Dockerfile .
    echo "engine tarball runpod/llm/out/env.tar, id $(cat runpod/llm/out/env.id)"
fi
docker build -f runpod/llm/Dockerfile \
    --build-arg IMAGE_ID="$(cat /proc/sys/kernel/random/uuid)" \
    -t "$tag" .
