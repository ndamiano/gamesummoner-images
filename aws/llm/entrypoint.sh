#!/usr/bin/env bash
# EC2 llm box: pull the Basalt pack onto the instance store, then run Basalt, the contract server
# and the baked worker side by side. Decommissioning the box is ending this script: /shutdown
# signals it, and the systemd unit powers the instance off once it ends.
set -euo pipefail

ENV_FILE=/etc/gamesummoner/env.json
GS=/opt/gamesummoner
VENV=/opt/basalt/.venv
BASALT=/opt/basalt
PACK=Qwen3.8-FN-UD-Q4_K_XL-bf16key-mtptuned.basalt
NVME=/opt/dlami/nvme
ENGINE_CONTEXT=262144
ENGINE_PORT=8001
IMAGE_PORT=8700
WORKER_GRACE_SECONDS=60

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
BASALT_ARGS_EXTRA=$(env_get BASALT_ARGS_EXTRA)
[ -n "$LLM_MODEL" ] && [ -n "$LLM_N_CTX" ] || { echo "env data needs LLM_MODEL and LLM_N_CTX" >&2; exit 1; }
[ "$LLM_N_CTX" -le "$ENGINE_CONTEXT" ] || {
    echo "LLM_N_CTX=$LLM_N_CTX exceeds the engine window $ENGINE_CONTEXT" >&2; exit 1; }

mark "host mem $(awk '/MemTotal/{printf "%d MiB", $2/1024}' /proc/meminfo), $(nproc) cpus"

for _ in $(seq 1 60); do mountpoint -q "$NVME" && break; sleep 1; done
mountpoint -q "$NVME" || { echo "instance store is not mounted at $NVME" >&2; exit 1; }
avail=$(df -BG --output=avail "$NVME" | tail -1 | tr -dc 0-9)
[ "${avail:-0}" -ge 120 ] || { echo "$NVME has ${avail}G free, the pack needs 107G" >&2; exit 1; }

region=$(curl -sf -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/placement/region || echo eu-north-1)
account=$(cat "$GS/account" 2>/dev/null || true)
bucket=${account:+gamesummoner-models-$region-$account}
[ -n "$bucket" ] || { echo "no account baked at $GS/account: the bucket is unknown" >&2; exit 1; }
( t=$(date +%s); nvidia-smi -pm 1 > /dev/null 2>&1
  mark "gpu persistence on, init took $(( $(date +%s) - t ))s" ) &

if [ ! -f "$NVME/$PACK" ]; then
    s5cmd --log error --endpoint-url "http://s3.$region.amazonaws.com" cp --concurrency 64 --part-size 64 \
        "s3://$bucket/basalt/$PACK" "$NVME/$PACK.part"
    mv "$NVME/$PACK.part" "$NVME/$PACK"
fi
mark "pack on the instance store ($(du -sh "$NVME/$PACK" | cut -f1))"

config=$NVME/basalt.json
python3 - "$config" "$NVME/$PACK" <<'CONFIG'
import json, sys
json.dump({"server": {"model": sys.argv[2], "vision": "dynamic", "fit-max-tokens": True},
           "engine": {"expert-cache": "auto", "prefill": "auto", "spec": 7, "spec-min-p": 0.5,
                      "kv": "int8", "vram-reserve-mib": 700, "resident-experts": True,
                      "max-context": 262144, "conversation-cache-mib": 32000,
                      "conversation-cache-slots": 16}}, open(sys.argv[1], "w"))
CONFIG
engine_args=(serve --config "$config" --alias "$LLM_MODEL" --host 127.0.0.1 --port "$ENGINE_PORT"
             --parallel "$WORKER_SLOTS")
read -r -a extra_args <<< "$BASALT_ARGS_EXTRA"
engine_args+=("${extra_args[@]}")

mark "starting engine: basalt ${engine_args[*]}"
setsid "$BASALT/bin/basalt" "${engine_args[@]}" &
engine_pid=$!

PYTHONPATH=$GS "$VENV/bin/python" -m aws.llm.server \
    --engine-url "http://127.0.0.1:$ENGINE_PORT" --engine-pid "$engine_pid" \
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
