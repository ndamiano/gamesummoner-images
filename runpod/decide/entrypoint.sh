#!/usr/bin/env bash
# RunPod decide box: stage JEV and its NVFP4 base off the art volume, prep the decision head, then
# run SGLang, the contract server and the worker. Boot order and reasons: runpod/README.md.
set -euo pipefail

: "${CP_URL:?CP_URL is required}"
: "${WORKER_TOKEN:?WORKER_TOKEN is required}"

ENV_FILE=/etc/gamesummoner/env.json
JEV_REPO=autotrust/JEV-27B-VL
JEV_REVISION=9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a
BASE_REPO=unsloth/Qwen3.8-27B-NVFP4
BASE_REVISION=7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108
VOL_DECIDE=/workspace/decide
STAGE=/stage/decide
PREP_DIR=/stage/jev
ENGINE_PORT=30000
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

jev_vol="$VOL_DECIDE/${JEV_REPO/\//--}@$JEV_REVISION"
base_vol="$VOL_DECIDE/${BASE_REPO/\//--}@$BASE_REVISION"
for d in "$jev_vol" "$base_vol"; do
    [ -f "$d/complete" ] || { echo "weights not on the volume: $d/complete is missing" >&2; exit 1; }
done

mkdir -p "$(dirname "$ENV_FILE")"
python3 - "$ENV_FILE" <<'PY'
import json, os, sys
keys = ["CP_URL", "WORKER_TOKEN", "WORKER_QUEUE", "WORKER_ID", "WORKER_SLOTS",
        "IDLE_EXIT_SECONDS", "JOB_TIMEOUT_SECONDS", "SGLANG_ARGS_EXTRA"]
env = {k: os.environ[k] for k in keys if k in os.environ}
env.setdefault("WORKER_QUEUE", "decide")
json.dump(env, open(sys.argv[1], "w"))
PY

mkdir -p "$STAGE"
cp -r "$jev_vol" "$STAGE/jev" & cp -r "$base_vol" "$STAGE/base" & wait
mark "weights staged ($(du -sh "$STAGE" | cut -f1))"

PYTHONPATH=/opt/gamesummoner python3 -m engines.jev.prep \
    --jev "$STAGE/jev" --base "$STAGE/base" --out "$PREP_DIR"
mark "decision head prepped in $PREP_DIR"

engine_args=(
    --model-path "$PREP_DIR/model" --served-model-name jev-27b-vl
    --enable-lora --max-lora-rank 32 --lora-paths "jev-decision=$PREP_DIR/adapter_sglang"
    --lora-strict-loading --max-loras-per-batch 2
    --enable-return-hidden-states
    --context-length 131072 --mem-fraction-static 0.91 --chunked-prefill-size 2048
    --cuda-graph-max-bs-prefill 512 --max-running-requests 1
    --host 127.0.0.1 --port "$ENGINE_PORT"
)
read -r -a extra_args <<< "${SGLANG_ARGS_EXTRA:-}"
engine_args+=("${extra_args[@]}")

mark "starting engine: ${engine_args[*]}"
python3 -m sglang.launch_server "${engine_args[@]}" &
engine_pid=$!

PYTHONPATH=/opt/gamesummoner python3 -m runpod.decide.server \
    --engine-url "http://127.0.0.1:$ENGINE_PORT" --jev-dir "$PREP_DIR" --engine-pid "$engine_pid" \
    --image-id "$(cat /opt/gamesummoner/image-id)" --cli-params="${engine_args[*]}" \
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
