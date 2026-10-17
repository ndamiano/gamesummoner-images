#!/usr/bin/env bash
# Runs as root on the builder instance (from build.sh's user-data): the CUDA toolkit, stock SGLang
# in a venv pinned to the local decide image's packages, the contract server and the worker, the
# systemd unit, and JEV's weights seeded into the bucket. Reports to s3://$BUCKET/build/$BUILD_ID/.
set -euo pipefail

: "${BUCKET:?}" "${BUILD_ID:?}"
SGLANG_VERSION=0.5.21
PYTHON_VERSION=3.12.3
CUDA_VERSION=13.0
JEV_REPO=autotrust/JEV-27B-VL
JEV_REVISION=9e53b8a809d2ab98e3e6a55b2b2dbf94b4c1e84a
BASE_REPO=unsloth/Qwen3.8-27B-NVFP4
BASE_REVISION=7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108
GS=/opt/gamesummoner
NVME=/opt/dlami/nvme
LOG=/var/log/gamesummoner-provision.log
REPORT="s3://$BUCKET/build/$BUILD_ID"

exec > >(tee -a "$LOG") 2>&1
t0=$(date +%s)
mark() { echo "[provision +$(( $(date +%s) - t0 ))s] $*"; }
report() {
    aws s3 cp "$LOG" "$REPORT/provision.log" --only-show-errors || true
    echo "$1" | aws s3 cp - "$REPORT/status" --only-show-errors || true
}
trap 'report failed' ERR

export DEBIAN_FRONTEND=noninteractive
mark "ubuntu $(. /etc/os-release && echo "$VERSION_ID"), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'no gpu')"

apt-get update -q
apt-get install -y -q --no-install-recommends ca-certificates curl gcc g++ ninja-build python3-pip
if [ ! -x "/usr/local/cuda-$CUDA_VERSION/bin/nvcc" ]; then
    curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -q
    apt-get install -y -q --no-install-recommends "cuda-toolkit-${CUDA_VERSION/./-}"
fi
mark "cuda $("/usr/local/cuda-$CUDA_VERSION/bin/nvcc" --version | grep -o 'release [0-9.]*')"

systemctl disable --now docker.service docker.socket containerd.service soci-snapshotter.service \
    nvidia-fabricmanager.service nvidia-cdi-refresh.service nvidia-cdi-refresh.path \
    nvidia_gpu_settings_service.service snapd.service snapd.socket snapd.seeded.service \
    unattended-upgrades.service hibinit-agent.service apport.service apport-autoreport.path \
    apport-autoreport.timer apport-forward.socket ModemManager.service udisks2.service \
    ec2-instance-connect-harvest-hostkeys.service efa-config.service networkd-dispatcher.service \
    lxd-installer.socket snapd.autoimport.service snapd.core-fixup.service \
    snapd.recovery-chooser-trigger.service snapd.system-shutdown.service snapd.snap-repair.timer \
    lvm2-monitor.service multipathd.service multipathd.socket open-iscsi.service iscsid.socket \
    open-vm-tools.service vgauth.service nvidia-dcgm.service pollinate.service ubuntu-advantage.service \
    ua-reboot-cmds.service ua-timer.timer ufw.service sysstat.service sysstat-collect.timer \
    sysstat-summary.timer apply_instance_tag_service.service e2scrub_reap.service e2scrub_all.timer \
    fwupd-refresh.timer motd-news.timer update-notifier-download.timer update-notifier-motd.timer \
    apt-daily.timer apt-daily-upgrade.timer man-db.timer mdcheck_start.timer mdcheck_continue.timer \
    mdmonitor-oneshot.timer dpkg-db-backup.timer 2>/dev/null || true
grep -l 'efa\|ib_core' /etc/modules-load.d/*.conf 2>/dev/null | xargs -r rm -f || true
systemctl enable snapd.apparmor.service 2>/dev/null || true
curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz \
    | tar -xz -C /usr/local/bin s5cmd
mark "boot services trimmed, s5cmd $(s5cmd version)"

mkdir -p "$GS"
find "$GS" -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} +
aws s3 cp "$REPORT/tree.tar" - --only-show-errors | tar -x -C "$GS"
chmod +x "$GS/entrypoint.sh"
echo "$ACCOUNT" > "$GS/account"

engine_stamp="$SGLANG_VERSION $PYTHON_VERSION $(sha256sum "$GS/aws/decide/pins.txt" | cut -c1-12)"
if [ "$(cat /opt/venv/.gamesummoner-engine 2>/dev/null)" = "$engine_stamp" ]; then
    mark "engine kept: $engine_stamp"
else
    pip install --break-system-packages -q uv
    uv python install "$PYTHON_VERSION"
    rm -rf /opt/venv "$GS/cache"
    uv venv --python "$PYTHON_VERSION" /opt/venv
    VIRTUAL_ENV=/opt/venv UV_CACHE_DIR=$NVME/uv uv pip install --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://docs.sglang.ai/whl/cu130/ \
        --extra-index-url https://flashinfer.ai/whl/cu130/ --extra-index-url https://flashinfer.ai/whl/ \
        --overrides "$GS/aws/decide/pins.txt" -r "$GS/aws/decide/pins.txt" \
        "sglang==$SGLANG_VERSION" hf_transfer
    rm -rf "$NVME/uv"
    echo "$engine_stamp" > /opt/venv/.gamesummoner-engine
    versions=$(/opt/venv/bin/python -c 'import sglang, torch; print(sglang.__version__, torch.__version__)')
    mark "engine installed: sglang and torch $versions"
fi

seed() {
    local repo=$1 revision=$2; shift 2
    local key="decide/${repo/\//--}@$revision"
    if aws s3 ls "s3://$BUCKET/$key/complete" >/dev/null 2>&1; then
        mark "$key already in the bucket"
        return
    fi
    local dir=$NVME/seed/${repo/\//--}
    HF_HUB_ENABLE_HF_TRANSFER=1 /opt/venv/bin/python -c '
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3],
                  allow_patterns=sys.argv[4:] or None, max_workers=16)
' "$repo" "$revision" "$dir" "$@"
    rm -rf "$dir/.cache"
    s5cmd --log error cp --concurrency 16 --part-size 64 "$dir/" "s3://$BUCKET/$key/"
    echo ok | aws s3 cp - "s3://$BUCKET/$key/complete" --only-show-errors
    rm -rf "$dir"
    mark "seeded s3://$BUCKET/$key"
}
seed "$JEV_REPO" "$JEV_REVISION" "adapter_vllm/*" "*.json" "*.jinja" "merges.txt"
seed "$BASE_REPO" "$BASE_REVISION"

cp "$GS/aws/decide/gamesummoner-decide.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable gamesummoner-decide
cat > /etc/netplan/60-gamesummoner.yaml <<'NETPLAN'
network:
  version: 2
  ethernets:
    primary:
      match:
        name: "en*"
      dhcp4: true
      dhcp6: false
NETPLAN
chmod 600 /etc/netplan/60-gamesummoner.yaml
netplan generate
touch /etc/cloud/cloud-init.disabled
mark "box code at $GS, image id $(cat "$GS/image-id"), worker $(cat "$GS/worker/ref")"

apt-get clean
rm -rf /root/.cache/pip /root/.cache/uv /tmp/cuda-keyring.deb
mark "provisioned"
report ok
