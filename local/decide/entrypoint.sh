#!/usr/bin/env bash
# Local decide box: pull JEV and its NVFP4 base, then run vLLM with JEV's decision LoRA, the
# contract server and the baked worker side by side. Decommissioning the box is ending this
# script: the contract server's /shutdown signals it, and so does `docker stop`.
set -euo pipefail

ENV_FILE=/etc/gamesummoner/env.json
JEV_REPO=autotrust/JEV-27B-VL
JEV_REVISION=9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a
BASE_REPO=unsloth/Qwen3.8-27B-NVFP4
BASE_REVISION=7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108
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
VLLM_ARGS_EXTRA=$(env_get VLLM_ARGS_EXTRA)

pull() {
    python3 -c '
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=sys.argv[3:] or None))
' "$@"
}
mark "pulling $JEV_REPO@$JEV_REVISION and $BASE_REPO@$BASE_REVISION"
jev_dir=$(pull "$JEV_REPO" "$JEV_REVISION" "adapter_vllm/*" "*.json" "*.jinja" "merges.txt")
base_dir=$(pull "$BASE_REPO" "$BASE_REVISION")
mark "weights at $jev_dir, $base_dir"

engine_args=(
    serve "$base_dir" --tokenizer "$jev_dir" --served-model-name jev-base
    --enable-lora --max-lora-rank 32 --lora-modules "jev-decision=$jev_dir/adapter_vllm"
    --logprobs-mode processed_logprobs --max-logprobs 32 --max-model-len 32768 --max-num-seqs 8
    --gpu-memory-utilization 0.90 --enforce-eager --limit-mm-per-prompt '{"image":0,"video":0}'
    --load-format runai_streamer --host 127.0.0.1 --port "$ENGINE_PORT"
)
read -r -a extra_args <<< "$VLLM_ARGS_EXTRA"
engine_args+=("${extra_args[@]}")

mark "starting engine: ${engine_args[*]}"
vllm "${engine_args[@]}" &
engine_pid=$!

PYTHONPATH=/opt/gamesummoner python3 -m local.decide.server \
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
