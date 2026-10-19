#!/usr/bin/env bash
# RunPod llm box: serve Basalt from the pack on the llm volume, run the contract server and the
# worker. Boot order and the reasons behind it: runpod/README.md.
set -euo pipefail

: "${CP_URL:?CP_URL is required}"
: "${WORKER_TOKEN:?WORKER_TOKEN is required}"
: "${LLM_MODEL:?LLM_MODEL is required}"
: "${LLM_N_CTX:?LLM_N_CTX is required}"

ENV_FILE=/etc/gamesummoner/env.json
ENGINE_PORT=8001
IMAGE_PORT=8700
WORKER_DIR=/opt/gamesummoner/worker
WORKER_GRACE_SECONDS=60
ENGINE_CONTEXT=262144
VENV=/opt/basalt/.venv

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

[ -f "$BASALT_PACK" ] || { echo "no pack at $BASALT_PACK" >&2; exit 1; }

mkdir -p "$(dirname "$ENV_FILE")"
python3 - "$ENV_FILE" <<'PY'
import json, os, sys
keys = ["CP_URL", "WORKER_TOKEN", "WORKER_QUEUE", "WORKER_ID", "WORKER_SLOTS",
        "IDLE_EXIT_SECONDS", "JOB_TIMEOUT_SECONDS", "LLM_MODEL", "LLM_N_CTX", "BASALT_ARGS_EXTRA"]
env = {k: os.environ[k] for k in keys if k in os.environ}
env.setdefault("WORKER_QUEUE", "llm")
json.dump(env, open(sys.argv[1], "w"))
PY

config=/root/basalt.json
python3 - "$config" "$BASALT_PACK" <<'CONFIG'
import json, sys
json.dump({"server": {"model": sys.argv[2], "vision": "dynamic", "fit-max-tokens": True},
           "engine": {"expert-cache": "auto", "prefill": "auto", "spec": 7, "spec-min-p": 0.5,
                      "kv": "int8", "vram-reserve-mib": 700, "resident-experts": True,
                      "max-context": 262144, "conversation-cache-mib": 32000,
                      "conversation-cache-slots": 16}}, open(sys.argv[1], "w"))
CONFIG
engine_args=(serve --config "$config" --alias "$LLM_MODEL" --host 127.0.0.1 --port "$ENGINE_PORT"
             --parallel "${WORKER_SLOTS:-1}")
read -r -a extra_args <<< "${BASALT_ARGS_EXTRA:-}"
engine_args+=("${extra_args[@]}")

mark "starting engine: basalt ${engine_args[*]}"
/opt/basalt/bin/basalt "${engine_args[@]}" &
engine_pid=$!

PYTHONPATH=/opt/gamesummoner "$VENV/bin/python" -m runpod.llm.server \
    --engine-url "http://127.0.0.1:$ENGINE_PORT" --engine-pid "$engine_pid" \
    --image-id "$(cat /opt/gamesummoner/image-id)" --cli-params="basalt ${engine_args[*]}" \
    --entrypoint-pid $$ --port "$IMAGE_PORT" &
contract_pid=$!

PYTHONPATH="$WORKER_DIR/src" "$VENV/bin/python" -m gamesummoner_worker \
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
