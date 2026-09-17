#!/usr/bin/env bash
# Kick off the local llm box by hand: host network so the worker reaches the control plane on
# localhost, the HF cache shared with the host, env data mounted read-only.
# The engine's JIT caches live on a docker volume: their file names outgrow what an encrypted
# home directory allows.
set -euo pipefail
env_json=${1:?usage: run.sh <env.json>}
exec docker run --rm --init --name gamesummoner-llm \
    --gpus all --ipc host --network host \
    --memory 40g --memory-swap 40g \
    --user "$(id -u):$(id -g)" -e USER=gamesummoner -e HOME=/cache/home -e HF_HOME=/cache/hf \
    -v "$HOME/.cache/huggingface:/cache/hf" \
    -v gamesummoner-llm-home:/cache/home \
    -v "$(realpath "$env_json"):/etc/gamesummoner/env.json:ro" \
    gamesummoner/llm-local:latest
