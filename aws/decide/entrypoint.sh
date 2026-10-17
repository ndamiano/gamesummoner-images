#!/usr/bin/env bash
# EC2 decide box: pull JEV and its NVFP4 base onto the instance store, prep the decision head,
# then run SGLang, the contract server and the baked worker side by side. Decommissioning the box
# is ending this script: /shutdown signals it, and the instance powers off once it ends.
set -euo pipefail

ENV_FILE=/etc/gamesummoner/env.json
GS=/opt/gamesummoner
VENV=/opt/venv
JEV_REPO=autotrust/JEV-27B-VL
JEV_REVISION=9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a
BASE_REPO=unsloth/Qwen3.8-27B-NVFP4
BASE_REVISION=7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108
NVME=/opt/dlami/nvme
WEIGHTS=$NVME/weights
PREP_DIR=$NVME/jev
ENGINE_PORT=30000
IMAGE_PORT=8700
WORKER_GRACE_SECONDS=60
CACHE_BASE=$GS/cache

boot_t0=$(date +%s)
mark() { echo "[boot +$(( $(date +%s) - boot_t0 ))s] $*"; }

token=""
for _ in $(seq 1 60); do
    token=$(curl -sf -m 2 -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' \
        http://169.254.169.254/latest/api/token || true)
    [ -n "$token" ] && break
    sleep 1
done
if [ -z "$token" ]; then
    echo "instance metadata unreachable after 60 s: no network, nothing can work" >&2
    systemctl poweroff
    exit 1
fi
if [ ! -f "$ENV_FILE" ]; then
    user_data=$(curl -sf -m 5 -H "X-aws-ec2-metadata-token: $token" \
        http://169.254.169.254/latest/user-data || true)
    if printf '%s' "$user_data" | python3 -c \
            'import json, sys; assert "WORKER_TOKEN" in json.load(sys.stdin)' 2>/dev/null; then
        mkdir -p "$(dirname "$ENV_FILE")"
        printf '%s' "$user_data" > "$ENV_FILE"
        chmod 600 "$ENV_FILE"
    elif [ "${user_data:0:2}" = "#!" ]; then
        echo "instance user-data is a script, not env data: running it"
        printf '%s' "$user_data" > /root/user-data.sh
        exec bash /root/user-data.sh
    else
        echo "no env data at $ENV_FILE and none in the instance user-data: not a worker launch"
        exit 0
    fi
fi
finish() {
    if [ -n "${bucket:-}" ] && command -v s5cmd >/dev/null; then
        instance=$(curl -sf -H "X-aws-ec2-metadata-token: $token" \
            http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo unknown)
        s5cmd --log error cp /var/log/gamesummoner-decide.log \
            "s3://$bucket/bootlogs/$(date -u +%Y%m%dT%H%M%S)-decide-$instance.log" 2>/dev/null || true
    fi
    [ -n "${GS_NO_POWEROFF:-}" ] || systemctl poweroff
}
env_get() {
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))' \
        "$ENV_FILE" "$1" "${2:-}"
}
SGLANG_ARGS_EXTRA=$(env_get SGLANG_ARGS_EXTRA)

mark "host mem $(awk '/MemTotal/{printf "%d MiB", $2/1024}' /proc/meminfo), $(nproc) cpus"

for _ in $(seq 1 60); do mountpoint -q "$NVME" && break; sleep 1; done
mountpoint -q "$NVME" || { echo "instance store is not mounted at $NVME" >&2; exit 1; }

region=$(curl -sf -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/placement/region || echo eu-north-1)
account=$(cat "$GS/account" 2>/dev/null || true)
bucket=${account:+gamesummoner-models-$region-$account}
S5=(s5cmd --log error --endpoint-url "http://s3.$region.amazonaws.com")
( t=$(date +%s); nvidia-smi -pm 1 > /dev/null 2>&1
  mark "gpu persistence on, init took $(( $(date +%s) - t ))s" ) &

ENV_DIR=$NVME/env
if [ -f "$GS/env-tar" ] && [ ! -f "$ENV_DIR/.ready" ]; then
    mkdir -p "$ENV_DIR"
    read -r env_key env_n < "$GS/env-tar"
    rm -rf "$NVME/env.parts" && mkdir -p "$NVME/env.parts"
    for i in $(seq 0 $((env_n - 1))); do
        ( until "${S5[@]}" cp --concurrency 4 --part-size 64 "s3://$bucket/$env_key/$(printf '%04d' "$i")" "$NVME/env.parts/"; do
            sleep 1
        done ) &
    done
    for i in $(seq 0 $((env_n - 1))); do
        f="$NVME/env.parts/$(printf '%04d' "$i")"
        until [ -f "$f" ]; do sleep 0.05; done
        cat "$f"
        rm -f "$f"
    done | tar -xf - -C "$ENV_DIR"
    wait
    rm -rf "$NVME/env.parts"
    mark "engine tarball down and unpacked"
    touch "$ENV_DIR/.ready"
fi
if [ -f "$ENV_DIR/.ready" ]; then
    for d in /opt/venv /usr/local/cuda-13.0 /root/.local/share/uv/python "$CACHE_BASE"; do
        mountpoint -q "$d" || mount --bind "$ENV_DIR$d" "$d"
    done
    mark "engine files on the instance store"
fi

fetch() {
    local repo=$1 revision=$2; shift 2
    local key="decide/${repo/\//--}@$revision" dir=$WEIGHTS/${repo/\//--}
    if [ -n "$bucket" ] && "${S5[@]}" ls "s3://$bucket/$key/complete" >/dev/null 2>&1; then
        "${S5[@]}" cp --concurrency 16 --part-size 64 --exclude complete "s3://$bucket/$key/*" "$dir/"
        mark "$repo from s3://$bucket/$key ($(du -sh "$dir" | cut -f1))"
    else
        HF_HOME=$NVME/hf HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV/bin/python" -c '
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3],
                  allow_patterns=sys.argv[4:] or None, max_workers=16)
' "$repo" "$revision" "$dir" "$@"
        mark "$repo from Hugging Face, not in the bucket ($(du -sh "$dir" | cut -f1))"
    fi
}
mkdir -p "$WEIGHTS"
fetch "$JEV_REPO" "$JEV_REVISION" "adapter_vllm/*" "*.json" "*.jinja" "merges.txt" &
jev_fetch=$!
fetch "$BASE_REPO" "$BASE_REVISION"
wait "$jev_fetch"

export CUDA_HOME=/usr/local/cuda-13.0 PATH=$VENV/bin:/usr/local/cuda-13.0/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export HOME=$CACHE_BASE/home XDG_CACHE_HOME=$CACHE_BASE TORCH_HOME=$CACHE_BASE/torch
export TORCHINDUCTOR_CACHE_DIR=$CACHE_BASE/torchinductor TRITON_CACHE_DIR=$CACHE_BASE/triton
export CUDA_CACHE_PATH=$CACHE_BASE/cuda FLASHINFER_WORKSPACE_BASE=$CACHE_BASE/flashinfer
export SGLANG_CACHE_DIR=$CACHE_BASE/sglang HF_HUB_OFFLINE=1
mkdir -p "$CACHE_BASE"/{home,torch,torchinductor,triton,cuda,flashinfer,sglang}

PYTHONPATH=$GS "$VENV/bin/python" -m engines.jev.prep \
    --jev "$WEIGHTS/${JEV_REPO/\//--}" --base "$WEIGHTS/${BASE_REPO/\//--}" --out "$PREP_DIR"
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
read -r -a extra_args <<< "$SGLANG_ARGS_EXTRA"
engine_args+=("${extra_args[@]}")

mark "starting engine: ${engine_args[*]}"
setsid "$VENV/bin/python" -m sglang.launch_server "${engine_args[@]}" &
engine_pid=$!

PYTHONPATH=$GS "$VENV/bin/python" -m aws.decide.server \
    --engine-url "http://127.0.0.1:$ENGINE_PORT" --jev-dir "$PREP_DIR" --engine-pid "$engine_pid" \
    --image-id "$(cat $GS/image-id)" --cli-params="${engine_args[*]}" \
    --entrypoint-pid $$ --port "$IMAGE_PORT" &
contract_pid=$!

PYTHONPATH=$GS/worker/src "$VENV/bin/python" -m gamesummoner_worker \
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
    kill -TERM "$worker_pid" "$contract_pid" 2>/dev/null || true
    kill -TERM -- "-$engine_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 -- "-$engine_pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$engine_pid" 2>/dev/null || true
    mark "box down"
    finish
    exit 0
}
trap decommission TERM INT

set +e
wait -n "$engine_pid" "$contract_pid" "$worker_pid"
mark "a box process exited on its own"
kill -TERM "$worker_pid" "$contract_pid" 2>/dev/null || true
kill -TERM -- "-$engine_pid" 2>/dev/null || true
finish
exit 1
