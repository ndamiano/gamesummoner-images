#!/usr/bin/env bash
# Turn a provisioned (and ideally end-to-end tested) builder into an AMI:
# bake.sh <queue> <instance id>
set -euo pipefail
. "$(dirname "$0")/common.sh"
queue=${1:?usage: bake.sh <queue> <instance id>}
instance=${2:?usage: bake.sh <queue> <instance id>}
image_id=$(ssm_run "$instance" "cat /opt/gamesummoner/image-id")
SSM_WAIT_SECONDS=1800 ssm_run "$instance" "systemctl stop gamesummoner-$queue 2>/dev/null; pkill -TERM -fx 'bash /opt/gamesummoner/entrypoint.sh'; sleep 5; bash /opt/gamesummoner/aws/$queue/pack_env.sh"
ssm_run "$instance" "rm -f /etc/gamesummoner/env.json /var/log/gamesummoner-$queue.log; rm -rf /opt/dlami/nvme/hf; cloud-init clean --logs; sync"
name="gamesummoner-$queue-$(date -u +%Y%m%dT%H%M%S)"
ami=$(ec2 create-image --instance-id "$instance" --name "$name" \
    --description "GameSummoner $queue box, image id $image_id" \
    --tag-specifications "ResourceType=image,Tags=[{Key=gamesummoner-image-id,Value=$image_id}]" \
    --query ImageId --output text)
echo "creating $ami ($name)"
for _ in $(seq 1 120); do
    state=$(ec2 describe-images --image-ids "$ami" --query 'Images[0].State' --output text)
    [ "$state" = available ] && { echo "ami $ami available, image id $image_id"; exit 0; }
    [ "$state" = pending ] || { echo "ami $ami is $state" >&2; exit 1; }
    sleep 30
done
echo "ami $ami still pending after an hour" >&2
exit 1
