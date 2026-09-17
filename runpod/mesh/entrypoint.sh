#!/usr/bin/env bash
# RunPod mesh box: start the TRELLIS and sprite servers, the contract server and the worker.
# Boot order and reasons: runpod/README.md.
set -euo pipefail

: "${CP_URL:?CP_URL is required}"
: "${WORKER_TOKEN:?WORKER_TOKEN is required}"

ENV_FILE=/etc/gamesummoner/env.json
IMAGE_PORT=8700
WORKER_DIR=/opt/gamesummoner/worker
WORKER_GRACE_SECONDS=60

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

test -d "$TRELLIS_WEIGHTS" || {
    echo "network volume not mounted: $TRELLIS_WEIGHTS is missing" >&2; exit 1; }
test -f "$SAFETY_MODEL_DIR/model.pt" || {
    echo "NSFW classifier missing: $SAFETY_MODEL_DIR/model.pt" >&2; exit 1; }

probe=$(ls -S "$TRELLIS_WEIGHTS"/ckpts/*.safetensors 2>/dev/null | head -1)
if [ -n "$probe" ]; then
    dd if="$probe" of=/dev/null bs=64M count=16 2>&1 | tail -1 | sed 's/^/[probe] volume seq read: /'
fi

mkdir -p "$(dirname "$ENV_FILE")"
python - "$ENV_FILE" <<'PY'
import json, os, sys
keys = ["CP_URL", "WORKER_TOKEN", "WORKER_QUEUE", "WORKER_ID", "WORKER_SLOTS",
        "IDLE_EXIT_SECONDS", "JOB_TIMEOUT_SECONDS"]
env = {k: os.environ[k] for k in keys if k in os.environ}
env.setdefault("WORKER_QUEUE", "mesh")
json.dump(env, open(sys.argv[1], "w"))
PY

"$SPRITE_PYTHON" /opt/gamesummoner/engines/sprites/sprite_server.py \
    --host 127.0.0.1 --port "$SPRITE_PORT" \
    ${KIMODO_TPOSE_BVH:+--ref-bvh "$KIMODO_TPOSE_BVH"} &
sprite_pid=$!

python /opt/gamesummoner/engines/trellis/trellis_server.py \
    --repo "$TRELLIS_REPO" --weights "$TRELLIS_WEIGHTS" \
    --host 127.0.0.1 --port "$TRELLIS_PORT" &
trellis_pid=$!

PYTHONPATH=/opt/gamesummoner python -m runpod.mesh.server \
    --engine-url "http://127.0.0.1:$TRELLIS_PORT" --engine-pid "$trellis_pid" \
    --sprite-url "http://127.0.0.1:$SPRITE_PORT" \
    --image-id "$(cat /opt/gamesummoner/image-id)" --cli-params="trellis_server.py --ptype 512 --texture 1024" \
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
    kill -TERM "$worker_pid" "$trellis_pid" "$sprite_pid" "$contract_pid" 2>/dev/null || true
    mark "box down"
    exit 0
}
trap decommission TERM INT

set +e
wait -n "$trellis_pid" "$sprite_pid" "$contract_pid" "$worker_pid"
mark "a box process exited on its own"
kill -TERM "$worker_pid" "$trellis_pid" "$sprite_pid" "$contract_pid" 2>/dev/null || true
exit 1
