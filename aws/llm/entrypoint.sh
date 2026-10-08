#!/usr/bin/env bash
# EC2 llm box: pull the checkpoint onto the instance store, then run Pennyroyal, the contract
# server and the baked worker side by side. Decommissioning the box is ending this script:
# /shutdown signals it, and the systemd unit powers the instance off once it ends.
set -euo pipefail

ENV_FILE=/etc/gamesummoner/env.json
GS=/opt/gamesummoner
VENV=/opt/venv
PENNY=/opt/penny
MODEL_REPO=RadixArk/Qwen3.8-Flash-Next-NVFP4
MODEL_REVISION=7b719225242aacd3dbd3f9407468c2ee9a9d2594
NVME=/opt/dlami/nvme
SERVED_MODEL=pennyroyal
ENGINE_CONTEXT=524288
ENGINE_PORT=8001
IMAGE_PORT=8700
WORKER_GRACE_SECONDS=60
CACHE_BASE=$GS/cache
PACK_DIR=$NVME/prepacked
SWAP_GB=64
JIT_JOBS=4

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
        s5cmd --log error cp /var/log/gamesummoner-llm.log \
            "s3://$bucket/bootlogs/$(date -u +%Y%m%dT%H%M%S)-$instance.log" 2>/dev/null || true
    fi
    [ -n "${GS_NO_POWEROFF:-}" ] || systemctl poweroff
}
env_get() {
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))' \
        "$ENV_FILE" "$1" "${2:-}"
}
LLM_MODEL=$(env_get LLM_MODEL)
LLM_N_CTX=$(env_get LLM_N_CTX)
WORKER_SLOTS=$(env_get WORKER_SLOTS 1)
SGLANG_ARGS_EXTRA=$(env_get SGLANG_ARGS_EXTRA)
[ -n "$LLM_MODEL" ] && [ -n "$LLM_N_CTX" ] || { echo "env data needs LLM_MODEL and LLM_N_CTX" >&2; exit 1; }
[ "$LLM_MODEL" = "$SERVED_MODEL" ] || {
    echo "LLM_MODEL=$LLM_MODEL but this engine serves $SERVED_MODEL" >&2; exit 1; }
[ "$LLM_N_CTX" -le "$ENGINE_CONTEXT" ] || {
    echo "LLM_N_CTX=$LLM_N_CTX exceeds the engine window $ENGINE_CONTEXT" >&2; exit 1; }

mark "host mem $(awk '/MemTotal/{printf "%d MiB", $2/1024}' /proc/meminfo), $(nproc) cpus"

for _ in $(seq 1 60); do mountpoint -q "$NVME" && break; sleep 1; done
mountpoint -q "$NVME" || { echo "instance store is not mounted at $NVME" >&2; exit 1; }
avail=$(df -BG --output=avail "$NVME" | tail -1 | tr -dc 0-9)
[ "${avail:-0}" -ge 200 ] || { echo "$NVME has ${avail}G free, the checkpoint needs 130G" >&2; exit 1; }

if [ "$(awk '/SwapTotal/{print $2}' /proc/meminfo)" -eq 0 ]; then
    fallocate -l "$SWAP_GB"G "$NVME/swap" && chmod 600 "$NVME/swap" && mkswap -q "$NVME/swap" && swapon "$NVME/swap"
fi
mark "swap: $(awk '/SwapTotal/{printf "%d MiB", $2/1024}' /proc/meminfo)"

region=$(curl -sf -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/placement/region || echo eu-north-1)
account=$(cat "$GS/account" 2>/dev/null || true)
bucket=${account:+gamesummoner-models-$region-$account}
S5=(s5cmd --log error --endpoint-url "http://s3.$region.amazonaws.com")
sysctl -q vm.dirty_background_bytes=268435456 vm.dirty_bytes=1073741824
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
    for d in /opt/venv /opt/penny /usr/local/cuda-13.3 /root/.local/share/uv/python "$CACHE_BASE"; do
        mountpoint -q "$d" || mount --bind "$ENV_DIR$d" "$d"
    done
    mark "engine files on the instance store"
fi

PART_BYTES=$((1024 * 1024 * 1024))
PART_WINDOW=$(env_get PACK_WINDOW 6)
PART_CONCURRENCY=$(env_get PACK_CONCURRENCY 16)
PART_RANGE_MB=$(env_get PACK_RANGE_MB 32)
PACK_THREADS=$(env_get PACK_THREADS 8)
pack_stream_parts() {
    for index in $(ls -S "$PACK_DIR"/*.parts/index.json 2>/dev/null | sort -r); do
        dir=$(dirname "$index")
        spec=$(cat "$index")
        read -r n part_bytes <<< "$(python3 -c 'import json, sys; s = json.loads(sys.argv[1]); print(-(-s["total_bytes"] // s["part_bytes"]), s["part_bytes"])' "$spec")"
        mountpoint -q "$dir" || mount -t tmpfs -o size=$(( (PART_WINDOW + 1) * part_bytes / 1024 / 1024 ))m tmpfs "$dir"
        printf '%s' "$spec" > "$dir/index.json"
        for i in $(seq 0 $((n - 1))); do
            while [ $(( $(ls "$dir" | grep -cE '^[0-9]{4}$') + $(jobs -rp | wc -l) )) -ge "$PART_WINDOW" ]; do
                sleep 0.1
            done
            part=$(printf '%04d' "$i")
            ( until "${S5[@]}" cp --concurrency "$PART_CONCURRENCY" --part-size "$PART_RANGE_MB" \
                    "s3://$bucket/prepacked/$(basename "$dir")/$part" "$dir/"; do
                rm -f "$dir/$part"[0-9]*
                sleep 1
            done ) &
        done
        wait
    done
    mark "pack parts streamed"
}
pack_complete=""
if [ -n "$bucket" ] && command -v s5cmd >/dev/null; then
    mkdir -p "$PACK_DIR"
    "${S5[@]}" sync --exclude '*.parts/0*' "s3://$bucket/prepacked/*" "$PACK_DIR/" 2>/dev/null || true
    [ -f "$PACK_DIR/complete" ] && pack_complete=1
    mark "pack: $(ls "$PACK_DIR" 2>/dev/null | tr '\n' ' ')${pack_complete:+(complete)}"
    [ -n "$pack_complete" ] && pack_stream_parts &
fi

export HF_HOME=$NVME/hf HF_XET_HIGH_PERFORMANCE=1
mark "pulling $MODEL_REPO@$MODEL_REVISION${pack_complete:+ without shards}"
model_dir=$("$VENV/bin/python" -c '
import sys
from huggingface_hub import snapshot_download
ignore = ["*.safetensors"] if sys.argv[3] else []
print(snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=16, ignore_patterns=ignore))
' "$MODEL_REPO" "$MODEL_REVISION" "$pack_complete")
mark "weights at $model_dir ($(du -shL "$model_dir" | cut -f1))"

[ -z "$pack_complete" ] && echo 3 > /proc/sys/vm/drop_caches
mark "$(awk '/MemFree/{printf "%d MiB free", $2/1024}' /proc/meminfo)"

export CUDA_HOME=/usr/local/cuda-13.3 CUDACXX=/usr/local/cuda-13.3/bin/nvcc
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 TORCH_CUDA_ARCH_LIST=12.0
export PATH=$VENV/bin:$CUDA_HOME/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=$JIT_JOBS CMAKE_BUILD_PARALLEL_LEVEL=$JIT_JOBS FLASHINFER_NINJA_JOBS=$JIT_JOBS
export FLASHINFER_NVCC_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=$JIT_JOBS
export XDG_CACHE_HOME=$CACHE_BASE TORCH_HOME=$CACHE_BASE/torch
export TORCHINDUCTOR_CACHE_DIR=$CACHE_BASE/torchinductor TRITON_CACHE_DIR=$CACHE_BASE/triton
export CUDA_CACHE_PATH=$CACHE_BASE/cuda FLASHINFER_WORKSPACE_BASE=$CACHE_BASE/flashinfer
export SGLANG_CACHE_DIR=$CACHE_BASE/sglang SGLANG_JIT_CACHE_DIR=$CACHE_BASE/sglang/jit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,pinned_max_round_threshold_mb:1024
export SGLANG_NUMA_BIND_V2=false SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_PREPACKED_DIR=$PACK_DIR SGLANG_PREPACKED_THREADS="$PACK_THREADS" SGLANG_PREPACKED_CONSUME=1
export SGLANG_MAMBA_CONV_DTYPE=bfloat16
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$CACHE_BASE"/{torch,torchinductor,triton,cuda,flashinfer,sglang/jit}

target_overrides='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'
engine_args=(
    --model-path "$model_dir" --load-format prepacked
    --served-model-name "$SERVED_MODEL" --host 127.0.0.1 --port "$ENGINE_PORT" --tp 1
    --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3
    --mem-fraction-static 0.981
    --context-length "$ENGINE_CONTEXT" --json-model-override-args "$target_overrides"
    --page-size 64 --max-running-requests "$WORKER_SLOTS" --sleep-on-idle
    --chunked-prefill-size 4096
    --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16
    --max-mamba-cache-size 24 --gdn-mtp-cache-mode none
    --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer
    --mamba-track-interval 64
    --ple-offload-embedding --trust-remote-code
    --chat-template "$model_dir/chat_template.jinja"
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder
    --enable-request-time-stats-logging --enable-metrics
    --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}'
    --speculative-algorithm NEXTN --speculative-num-steps 3
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
    --speculative-draft-model-quantization unquant --watchdog-timeout 1800
)
read -r -a extra_args <<< "$SGLANG_ARGS_EXTRA"
engine_args+=("${extra_args[@]}")
read -r -a graph_bs <<< "$(seq -s ' ' 1 "$WORKER_SLOTS")"
engine_args+=(--cuda-graph-bs "${graph_bs[@]}")

mark "starting engine: ${engine_args[*]}"
setsid "$VENV/bin/sglang" serve "${engine_args[@]}" &
engine_pid=$!

PYTHONPATH=$GS "$VENV/bin/python" -m aws.llm.server \
    --engine-url "http://127.0.0.1:$ENGINE_PORT" --engine-pid "$engine_pid" \
    --image-id "$(cat $GS/image-id)" --cli-params="${engine_args[*]}" \
    --entrypoint-pid $$ --port "$IMAGE_PORT" &
contract_pid=$!

PYTHONPATH=$GS/worker/src "$VENV/bin/python" -m gamesummoner_worker \
    --env-file "$ENV_FILE" --image "http://127.0.0.1:$IMAGE_PORT" &
worker_pid=$!

pack_seed() {
    [ -n "$bucket" ] && [ -z "$pack_complete" ] || return 0
    until curl -sf -o /dev/null "http://127.0.0.1:$IMAGE_PORT/health"; do
        kill -0 "$engine_pid" 2>/dev/null || return 0
        sleep 5
    done
    ls "$PACK_DIR"/*.bin >/dev/null 2>&1 || return 0
    command -v s5cmd >/dev/null || return 0
    for bin in "$PACK_DIR"/*.bin; do
        dir="${bin%.bin}.parts"
        [ -f "$dir/index.json" ] && continue
        mkdir -p "$dir"
        nice -n 19 split -b "$PART_BYTES" -d -a 4 "$bin" "$dir/"
        printf '{"part_bytes": %d, "total_bytes": %d}' "$PART_BYTES" "$(stat -c %s "$bin")" > "$dir/index.json"
    done
    touch "$PACK_DIR/complete"
    mark "seeding s3://$bucket/prepacked/ with $(du -sh "$PACK_DIR" | cut -f1)"
    nice -n 19 s5cmd --log error sync --exclude '*.bin' --exclude 'complete' "$PACK_DIR/" "s3://$bucket/prepacked/" \
        && s5cmd --log error cp "$PACK_DIR/complete" "s3://$bucket/prepacked/complete"
    mark "pack seeded"
}
pack_seed &

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
