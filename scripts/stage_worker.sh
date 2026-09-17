#!/usr/bin/env bash
# Stage the worker at <image dir>/worker.ref into <image dir>/worker/ for the Dockerfile to COPY.
# From a local checkout (WORKER_REPO_DIR, default ~/Documents/gamesummoner-workers): the repo is
# private, and baking keeps tokens off the box.
set -euo pipefail
image_dir=${1:?usage: stage_worker.sh <image dir>}
repo=${WORKER_REPO_DIR:-$HOME/Documents/gamesummoner-workers}
ref=$(cat "$image_dir/worker.ref")
git -C "$repo" cat-file -e "$ref^{commit}" 2>/dev/null || {
    echo "worker ref $ref is not in $repo (fetch it, or set WORKER_REPO_DIR)" >&2; exit 1; }
rm -rf "$image_dir/worker"
mkdir -p "$image_dir/worker"
git -C "$repo" archive "$ref" src | tar -x -C "$image_dir/worker"
echo "$ref" > "$image_dir/worker/ref"
echo "staged worker $ref into $image_dir/worker"
