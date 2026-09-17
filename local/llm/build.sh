#!/usr/bin/env bash
# Build the local llm image; every build gets a fresh image id.
set -euo pipefail
cd "$(dirname "$0")/../.."
docker build -f local/llm/Dockerfile \
    --build-arg IMAGE_ID="$(cat /proc/sys/kernel/random/uuid)" \
    -t gamesummoner/llm-local:latest .
