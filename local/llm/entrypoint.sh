#!/usr/bin/env bash
# Local llm box: pull the weights, then run SGLang, the contract server and the baked worker side
# by side. Decommissioning the box is ending this script: the contract
# server's /shutdown signals it, and so does `docker stop`.
set -euo pipefail

ENV_FILE=/etc/gamesummoner/env.json
MODEL_REPO=gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090
MODEL_REVISION=5b7a687fc8211a5d631c8ca6a593dd37eb26ce33
ENGINE_CONTEXT=262144
ENGINE_PORT=30000
IMAGE_PORT=8700
WORKER_DIR=/opt/gamesummoner/worker
WORKER_GRACE_SECONDS=60

boot_t0=$(date +%s)
mark() { echo "[boot +$(( $(date +%s) - boot_t0 ))s] $*"; }

[ -f "$ENV_FILE" ] || { echo "no env data at $ENV_FILE" >&2; exit 1; }
env_get() {
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))' \
        "$ENV_FILE" "$1" "${2:-}"
}
LLM_MODEL=$(env_get LLM_MODEL)
LLM_N_CTX=$(env_get LLM_N_CTX)
WORKER_SLOTS=$(env_get WORKER_SLOTS 1)
SGLANG_ARGS_EXTRA=$(env_get SGLANG_ARGS_EXTRA)
[ -n "$LLM_MODEL" ] && [ -n "$LLM_N_CTX" ] || { echo "env data needs LLM_MODEL and LLM_N_CTX" >&2; exit 1; }
[ "$LLM_N_CTX" -le "$ENGINE_CONTEXT" ] || {
    echo "LLM_N_CTX=$LLM_N_CTX exceeds the engine window $ENGINE_CONTEXT" >&2; exit 1; }

mark "pulling $MODEL_REPO@$MODEL_REVISION"
model_dir=$(python3 -c '
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=["assets/*"]))
' "$MODEL_REPO" "$MODEL_REVISION")
mark "weights at $model_dir"

engine_args=(
    --model-path "$model_dir" --trust-remote-code
    --served-model-name "$LLM_MODEL" --context-length "$ENGINE_CONTEXT"
    --max-running-requests "$WORKER_SLOTS"
    --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --chunked-prefill-size 2048
    --mamba-radix-cache-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16
    --max-mamba-cache-size 12 --mem-fraction-static 0.85
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder
    --host 127.0.0.1 --port "$ENGINE_PORT"
)
read -r -a extra_args <<< "$SGLANG_ARGS_EXTRA"
engine_args+=("${extra_args[@]}")

mark "starting engine: ${engine_args[*]}"
python3 -m sglang.launch_server "${engine_args[@]}" &
engine_pid=$!

PYTHONPATH=/opt/gamesummoner python3 -m local.llm.server \
    --engine-url "http://127.0.0.1:$ENGINE_PORT" --engine-pid "$engine_pid" \
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
decommission
