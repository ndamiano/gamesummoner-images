#!/usr/bin/env bash
# Provision a builder instance from the DLAMI: stage the worker, ship the box code through the
# bucket, launch, and wait for provision.sh to report. Leaves the builder running for the
# end-to-end test; bake.sh turns it into an AMI. build.sh [--on-demand] [--base <ami>]
# --base starts from one of our own AMIs instead of the DLAMI, keeping its tuned caches.
set -euo pipefail
cd "$(dirname "$0")/../.."
. aws/llm/common.sh

market=spot
base=""
while [ $# -gt 0 ]; do
    case "$1" in
        --on-demand) market=on-demand; shift ;;
        --base) base=$2; shift 2 ;;
        *) echo "usage: build.sh [--on-demand] [--base <ami>]" >&2; exit 1 ;;
    esac
done

scripts/stage_worker.sh aws/llm
build_id=$(date -u +%Y%m%dT%H%M%S)
image_id=$(cat /proc/sys/kernel/random/uuid)
out=aws/llm/out/$build_id
rm -rf "$out" && mkdir -p "$out/tree/aws/llm"
cp -r contract "$out/tree/contract"
cp aws/server.py "$out/tree/aws/server.py"
cp aws/llm/server.py aws/llm/gamesummoner-llm.service aws/llm/prepacked-parts.patch aws/llm/pack_env.sh "$out/tree/aws/llm/"
cp -r aws/llm/worker "$out/tree/worker"
cp aws/llm/entrypoint.sh "$out/tree/entrypoint.sh"
echo "$image_id" > "$out/tree/image-id"
find "$out/tree" -name __pycache__ -type d -exec rm -rf {} +
tar -cf "$out/tree.tar" -C "$out/tree" .
aws s3 cp "$out/tree.tar" "s3://$BUCKET/build/$build_id/tree.tar" --only-show-errors
aws s3 cp aws/llm/provision.sh "s3://$BUCKET/build/$build_id/provision.sh" --only-show-errors
echo "build $build_id, image id $image_id, shipped to s3://$BUCKET/build/$build_id/"

cat > "$out/user-data.sh" <<USERDATA
#!/bin/bash
export BUCKET=$BUCKET BUILD_ID=$build_id AWS_DEFAULT_REGION=$REGION
command -v aws >/dev/null || snap install aws-cli --classic
aws s3 cp s3://$BUCKET/build/$build_id/provision.sh /root/provision.sh
bash /root/provision.sh
USERDATA

instance=$(run_instance "gs-llm-builder-$build_id" "${base:-$(dlami_id)}" "$out/user-data.sh" "$market")
echo "$instance" > "$out/instance"
echo "builder $instance; waiting for provision.sh (log: s3://$BUCKET/build/$build_id/provision.log)"
for _ in $(seq 1 180); do
    status=$(aws s3 cp "s3://$BUCKET/build/$build_id/status" - 2>/dev/null || true)
    if [ -n "$status" ]; then
        aws s3 cp "s3://$BUCKET/build/$build_id/provision.log" "$out/provision.log" --only-show-errors
        grep '^\[provision' "$out/provision.log"
        echo "provision $status on $instance (build $build_id)"
        [ "$status" = ok ]
        exit
    fi
    state=$(ec2 describe-instances --instance-ids "$instance" \
        --query 'Reservations[0].Instances[0].State.Name' --output text)
    [ "$state" = running ] || [ "$state" = pending ] || { echo "builder is $state" >&2; exit 1; }
    sleep 30
done
echo "provision did not report in 90 minutes" >&2
exit 1
