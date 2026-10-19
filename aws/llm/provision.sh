#!/usr/bin/env bash
# Runs as root on the builder instance (from build.sh's user-data): the toolchain, Basalt from
# the pinned fork tag, the contract server and the worker, the systemd unit. Reports to
# s3://$BUCKET/build/$BUILD_ID/. Why each step: aws/README.md.
set -euo pipefail

: "${BUCKET:?}" "${BUILD_ID:?}"
BASALT_REPO=https://github.com/ndamiano/basalt
BASALT_REF=gamesummoner-v1
CUDA_VERSION=13.3
GS=/opt/gamesummoner
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
apt-get install -y -q --no-install-recommends \
    ca-certificates curl git build-essential cmake python3-venv
mark "toolchain installed"

if [ ! -x "/usr/local/cuda-$CUDA_VERSION/bin/nvcc" ]; then
    curl -fsSL -o /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -q
    apt-get install -y -q --no-install-recommends "cuda-toolkit-${CUDA_VERSION/./-}"
fi
mark "cuda $("/usr/local/cuda-$CUDA_VERSION/bin/nvcc" --version | grep -o 'release [0-9.]*')"
export CUDA_HOME=/usr/local/cuda-$CUDA_VERSION PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH

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
curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz \
    | tar -xz -C /usr/local/bin s5cmd
mark "boot services trimmed, s5cmd $(s5cmd version)"

mkdir -p "$GS"
find "$GS" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
aws s3 cp "$REPORT/tree.tar" - --only-show-errors | tar -x -C "$GS"
chmod +x "$GS/entrypoint.sh"
echo "$ACCOUNT" > "$GS/account"

engine_stamp="$BASALT_REF"
if [ "$(cat /opt/basalt/.gamesummoner-engine 2>/dev/null)" = "$engine_stamp" ] && [ -x /opt/basalt/bin/basalt ]; then
    mark "engine kept: $engine_stamp"
else
    rm -rf /opt/basalt
    git clone --depth 1 -b "$BASALT_REF" "$BASALT_REPO" /opt/basalt
    cmake -S /opt/basalt -B /opt/basalt/build -DCMAKE_BUILD_TYPE=Release
    cmake --build /opt/basalt/build -j"$(nproc)"
    python3 -m venv /opt/basalt/.venv
    /opt/basalt/.venv/bin/pip install -q -r /opt/basalt/requirements.txt "requests>=2.31"
    echo "$engine_stamp" > /opt/basalt/.gamesummoner-engine
    mark "basalt $(git -C /opt/basalt rev-parse --short HEAD) built"
fi

cp "$GS/aws/llm/gamesummoner-llm.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable gamesummoner-llm
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
rm -rf /root/.cache/pip /tmp/cuda-keyring.deb
mark "provisioned"
report ok
