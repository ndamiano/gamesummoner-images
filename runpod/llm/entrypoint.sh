#!/usr/bin/env bash
# RunPod llm box: restore the engine, boot Pennyroyal, run the contract server and the worker.
# Boot order and the reasons behind it: runpod/README.md.
set -euo pipefail

: "${CP_URL:?CP_URL is required}"
: "${WORKER_TOKEN:?WORKER_TOKEN is required}"
: "${LLM_MODEL:?LLM_MODEL is required}"
: "${LLM_N_CTX:?LLM_N_CTX is required}"

ENV_FILE=/etc/gamesummoner/env.json
ENGINE_PORT=${SGLANG_PORT:-8001}
IMAGE_PORT=8700
WORKER_DIR=/opt/gamesummoner/worker
WORKER_GRACE_SECONDS=60
SERVED_MODEL=pennyroyal
ENGINE_CONTEXT=524288

[ "$LLM_MODEL" = "$SERVED_MODEL" ] || {
    echo "LLM_MODEL=$LLM_MODEL but this engine serves $SERVED_MODEL" >&2; exit 1; }
[ "$LLM_N_CTX" -le "$ENGINE_CONTEXT" ] || {
    echo "LLM_N_CTX=$LLM_N_CTX exceeds the engine window $ENGINE_CONTEXT" >&2; exit 1; }

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

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
mark "card: ${gpu:-unknown}"

env_id=$(cat /opt/gamesummoner/env.id)
if [ ! -x /opt/venv/bin/python ]; then
    env_tar="$PENNY_ENV_DIR/llm-env-$env_id.tar"
    [ -f "$env_tar" ] || { echo "engine env $env_tar is not on the volume" >&2; exit 1; }
    mark "restoring engine env $env_id"
    tar -xf "$env_tar" -C /
    ldconfig
    mark "engine env restored"
fi

[ -f "$PENNY_MODEL_DIR/config.json" ] || { echo "no checkpoint at $PENNY_MODEL_DIR" >&2; exit 1; }

mkdir -p "$(dirname "$ENV_FILE")"
python3 - "$ENV_FILE" <<'PY'
import json, os, sys
keys = ["CP_URL", "WORKER_TOKEN", "WORKER_QUEUE", "WORKER_ID", "WORKER_SLOTS",
        "IDLE_EXIT_SECONDS", "JOB_TIMEOUT_SECONDS", "LLM_MODEL", "LLM_N_CTX", "SGLANG_ARGS_EXTRA"]
env = {k: os.environ[k] for k in keys if k in os.environ}
env.setdefault("WORKER_QUEUE", "llm")
json.dump(env, open(sys.argv[1], "w"))
PY

export TARGET_MODEL="$PENNY_MODEL_DIR" CACHE_BASE=/root/cache NIXL_STORAGE_BASE=/root/nixl
export SGLANG_EXE=/opt/venv/bin/sglang REPO_ROOT=/opt/penny NIXL_CONFIG=/root/nixl-posix.toml
export SGLANG_PREPACKED_DIR="$PENNY_MODEL_DIR/prepacked"
engine_args="--load-format prepacked ${SGLANG_ARGS_EXTRA:-} --cuda-graph-bs $(seq -s ' ' 1 "${WORKER_SLOTS:-1}")"
export SGLANG_ARGS_EXTRA="$engine_args"
mkdir -p /root/nixl
engine_url="http://127.0.0.1:$ENGINE_PORT"

healthy=""
for attempt in 1 2 3; do
    mark "engine launch attempt $attempt"
    bash /root/serve-nohicache.sh &
    engine_pid=$!
    for _ in $(seq 1 200); do
        curl -sf -o /dev/null "$engine_url/health_generate" && { healthy=1; break; }
        kill -0 "$engine_pid" 2>/dev/null || { mark "engine died during startup (attempt $attempt)"; break; }
        sleep 3
    done
    [ -n "$healthy" ] && break
    pkill -9 -f "sglang" 2>/dev/null || true
    sleep 5
done
[ -n "$healthy" ] || { echo "engine never became healthy after 3 attempts" >&2; exit 1; }
mark "pennyroyal up as $LLM_MODEL"

PYTHONPATH=/opt/gamesummoner python3 -m runpod.llm.server \
    --engine-url "$engine_url" --engine-pid "$engine_pid" \
    --image-id "$(cat /opt/gamesummoner/image-id)" --cli-params="serve-nohicache.sh $engine_args" \
    --entrypoint-pid $$ --port "$IMAGE_PORT" &
contract_pid=$!

PYTHONPATH="$WORKER_DIR/src" python3 -m gamesummoner_worker \
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
    kill -TERM "$worker_pid" "$engine_pid" "$contract_pid" 2>/dev/null || true
    mark "box down"
    exit 0
}
trap decommission TERM INT

set +e
wait -n "$engine_pid" "$contract_pid" "$worker_pid"
mark "a box process exited on its own"
kill -TERM "$worker_pid" "$engine_pid" "$contract_pid" 2>/dev/null || true
exit 1
