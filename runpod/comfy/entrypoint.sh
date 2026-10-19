#!/usr/bin/env bash
# RunPod ComfyUI box: stage this queue's weights, start ComfyUI, run the contract server (which
# warms the box) and the worker. Boot order and reasons: runpod/README.md.
set -euo pipefail

: "${CP_URL:?CP_URL is required}"
: "${WORKER_TOKEN:?WORKER_TOKEN is required}"
: "${QUEUE:?QUEUE is baked at build}"
WORKER_QUEUE="${WORKER_QUEUE:-$QUEUE}"
[ "$WORKER_QUEUE" = "$QUEUE" ] || {
    echo "WORKER_QUEUE=$WORKER_QUEUE but this image serves $QUEUE" >&2; exit 1; }

ENV_FILE=/etc/gamesummoner/env.json
IMAGE_PORT=8700
WORKER_DIR=/opt/gamesummoner/worker
WORKER_GRACE_SECONDS=60
VOL_MODELS=/workspace/comfy/models
STAGE_DIR="${COMFY_STAGE_DIR:-/stage/comfy-models}"
TREE=/opt/comfy-models

while [ $# -gt 0 ]; do
    case "$1" in
        --worker-id) WORKER_ID=$2; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 1 ;;
    esac
done
: "${WORKER_ID:?WORKER_ID is required — the scaler passes it as --worker-id}"
export WORKER_ID

boot_t0=$(date +%s)
mark() { echo "[boot +$(( $(date +%s) - boot_t0 ))s] $*"; }

test -e "$VOL_MODELS/checkpoints" || {
    echo "network volume not mounted: $VOL_MODELS is missing" >&2; exit 1; }

mkdir -p "$TREE"
for d in "$VOL_MODELS"/*/; do
    ln -sfn "${d%/}" "$TREE/$(basename "$d")"
done

mkdir -p "$(dirname "$ENV_FILE")"
python - "$ENV_FILE" <<'PY'
import json, os, sys
keys = ["CP_URL", "WORKER_TOKEN", "WORKER_QUEUE", "WORKER_ID", "WORKER_SLOTS",
        "IDLE_EXIT_SECONDS", "JOB_TIMEOUT_SECONDS"]
env = {k: os.environ[k] for k in keys if k in os.environ}
env.setdefault("WORKER_QUEUE", os.environ["QUEUE"])
json.dump(env, open(sys.argv[1], "w"))
PY

python /opt/gamesummoner/stage.py "$VOL_MODELS" "$STAGE_DIR" "$TREE" "$QUEUE" &
stage_pid=$!

python /opt/ComfyUI/main.py --listen 127.0.0.1 --port "$COMFY_PORT" --disable-auto-launch \
    --use-pytorch-cross-attention &
comfy_pid=$!

for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null && break
    kill -0 "$comfy_pid" 2>/dev/null || { echo "ComfyUI died during startup" >&2; exit 1; }
    sleep 2
done
curl -sf "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null || {
    echo "ComfyUI did not answer /system_stats within 180s" >&2; exit 1; }
mark "ComfyUI up"

wait "$stage_pid" || true
mark "staging settled"

PYTHONPATH=/opt/gamesummoner python -m "runpod.$QUEUE.server" \
    --engine-url "http://127.0.0.1:$COMFY_PORT" --engine-pid "$comfy_pid" \
    --image-id "$(cat /opt/gamesummoner/image-id)" --cli-params="ComfyUI --listen 127.0.0.1" \
    --entrypoint-pid $$ --port "$IMAGE_PORT" &
contract_pid=$!

PYTHONPATH="$WORKER_DIR/src" python -m gamesummoner_worker \
    --env-file "$ENV_FILE" --image "http://127.0.0.1:$IMAGE_PORT" &
worker_pid=$!

decommissioning=""
decommission() {
    [ -n "$decommissioning" ] && return
    decommissioning=1
    mark "decommissioning"
    kill -TERM "$worker_pid" 2>/dev/null || true
    for _ in $(seq 1 "$WORKER_GRACE_SECONDS"); do
        kill -0 "$worker_pid" 2>/dev/null || break
        sleep 1
    done
    kill -TERM "$worker_pid" "$comfy_pid" "$contract_pid" 2>/dev/null || true
    mark "box down"
    exit 0
}
trap decommission TERM INT

set +e
wait -n "$comfy_pid" "$contract_pid" "$worker_pid"
mark "a box process exited on its own"
kill -TERM "$worker_pid" "$comfy_pid" "$contract_pid" 2>/dev/null || true
exit 1
