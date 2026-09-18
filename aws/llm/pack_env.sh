#!/usr/bin/env bash
# Runs on a box before it is baked (bake.sh calls it): promotes the JIT kernels to FlashInfer's
# AOT tree, tars the engine files the entrypoint bind-mounts from the instance store at boot,
# and uploads the tarball to the bucket under this image id. The previous image's tarball, if
# provisioning left its key at /root/env-tar.previous, is the base: it is restored to the NVMe
# and only changed files are read off the (lazily restored, slow) root volume. Why:
# aws/README.md, "engine files".
set -euo pipefail
GS=/opt/gamesummoner
NVME=/opt/dlami/nvme
DIRS=(/opt/venv /opt/penny /usr/local/cuda-13.3 /root/.local/share/uv/python "$GS/cache")
bucket=$(cat "$GS/bucket")
image_id=$(cat "$GS/image-id")
key="env/llm-env-$image_id.parts"
t0=$(date +%s)
mark() { echo "[pack_env +$(( $(date +%s) - t0 ))s] $*"; }

for d in "${DIRS[@]}"; do
    mountpoint -q "$d" && umount "$d"
done

aot=/opt/venv/lib/python3.12/site-packages/flashinfer/data/aot
for so in "$GS"/cache/flashinfer/.cache/flashinfer/*/*/cached_ops/*/*.so; do
    [ -f "$so" ] || continue
    op=$(basename "$so" .so)
    mkdir -p "$aot/$op" && cp "$so" "$aot/$op/$op.so"
done
mark "aot: $(ls "$aot" 2>/dev/null | tr '\n' ' ')"

stage=$NVME/env-stage
rm -rf "$stage" && mkdir -p "$stage"
if [ -f /root/env-tar.previous ]; then
    read -r prev_key prev_n < /root/env-tar.previous
    if [ -n "$prev_n" ]; then
        mkdir -p "$NVME/env-prev"
        s5cmd --log error cp --concurrency 8 --part-size 64 "s3://$bucket/$prev_key/*" "$NVME/env-prev/"
        cat "$NVME/env-prev"/0* | tar -xf - -C "$stage"
        rm -rf "$NVME/env-prev"
    else
        s5cmd --log error cp --concurrency 32 --part-size 128 "s3://$bucket/$prev_key" "$NVME/env-prev.tar"
        tar -xf "$NVME/env-prev.tar" -C "$stage"
        rm -f "$NVME/env-prev.tar"
    fi
    mark "previous tarball staged"
fi
for d in "${DIRS[@]}"; do
    mkdir -p "$stage$d"
    rsync -a --delete "$d/" "$stage$d/"
done
find "$stage/usr/local/cuda-13.3" -name '*.a' ! -name 'libcudadevrt.a' ! -name 'libcudart_static.a' -delete
rm -rf "$stage/usr/local/cuda-13.3"/{compute-sanitizer,gds,extras,nsight*,doc,src}
mark "staged $(du -sh "$stage" | cut -f1)"
tar -cf "$NVME/env.tar" -C "$stage" opt root usr
rm -rf "$stage"
mark "tar $(du -sh "$NVME/env.tar" | cut -f1)"
rm -rf "$NVME/env.parts" && mkdir -p "$NVME/env.parts"
split -b 1G -d -a 4 "$NVME/env.tar" "$NVME/env.parts/"
rm -f "$NVME/env.tar"
s5cmd --log error cp --concurrency 8 --part-size 64 "$NVME/env.parts/0*" "s3://$bucket/$key/"
echo "$key $(ls "$NVME/env.parts" | wc -l)" > "$GS/env-tar"
rm -rf "$NVME/env.parts"
mark "uploaded s3://$bucket/$key"
