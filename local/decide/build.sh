#!/usr/bin/env bash
# Build the local decide image; every build gets a fresh image id and the worker at local/decide/worker.ref.
set -euo pipefail
cd "$(dirname "$0")/../.."
scripts/stage_worker.sh local/decide
docker build -f local/decide/Dockerfile \
    --build-arg IMAGE_ID="$(cat /proc/sys/kernel/random/uuid)" \
    -t gamesummoner/decide-local:latest .
