#!/usr/bin/env bash
# Runs on a box before it is baked (bake.sh calls it): tars the engine files the entrypoint
# bind-mounts from the instance store at boot and uploads them to the bucket under this image id.
set -euo pipefail
GS=/opt/gamesummoner
NVME=/opt/dlami/nvme
DIRS=(/opt/venv /usr/local/cuda-13.0 /root/.local/share/uv/python "$GS/cache")
imds_token=$(curl -sf -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 600' http://169.254.169.254/latest/api/token)
region=$(curl -sf -H "X-aws-ec2-metadata-token: $imds_token" http://169.254.169.254/latest/meta-data/placement/region)
bucket=gamesummoner-models-$region-$(cat "$GS/account")
key="env/decide-env-$(cat "$GS/image-id").parts"
t0=$(date +%s)
mark() { echo "[pack_env +$(( $(date +%s) - t0 ))s] $*"; }

for d in "${DIRS[@]}"; do
    mountpoint -q "$d" && umount "$d"
done

stage=$NVME/env-stage
rm -rf "$stage" && mkdir -p "$stage"
for d in "${DIRS[@]}"; do
    mkdir -p "$stage$d"
    rsync -a --delete "$d/" "$stage$d/"
done
find "$stage/usr/local/cuda-13.0" -name '*.a' ! -name 'libcudadevrt.a' ! -name 'libcudart_static.a' -delete
rm -rf "$stage/usr/local/cuda-13.0"/{compute-sanitizer,gds,extras,nsight*,doc,src}
mark "staged $(du -sh "$stage" | cut -f1)"
rm -rf "$NVME/env.parts" && mkdir -p "$NVME/env.parts"
tar -cf - -C "$stage" opt root usr | split -b 1G -d -a 4 - "$NVME/env.parts/"
rm -rf "$stage"
s5cmd --log error cp --concurrency 8 --part-size 64 "$NVME/env.parts/0*" "s3://$bucket/$key/"
echo "$key $(ls "$NVME/env.parts" | wc -l)" > "$GS/env-tar"
rm -rf "$NVME/env.parts"
mark "uploaded s3://$bucket/$key"
