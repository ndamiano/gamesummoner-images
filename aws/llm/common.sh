# Sourced by the aws/llm scripts: the region, the standing infrastructure, the pool walk.
REGION=${AWS_REGION:-eu-north-1}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=gamesummoner-models-$REGION-$ACCOUNT
INSTANCE_TYPE=${INSTANCE_TYPE:-g7e.2xlarge}
SECURITY_GROUP_NAME=gs-gpu-worker
INSTANCE_PROFILE=gs-gpu-worker
DLAMI_PARAM=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id

ec2() { aws ec2 --region "$REGION" "$@"; }

security_group_id() {
    ec2 describe-security-groups --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
        --query 'SecurityGroups[0].GroupId' --output text
}

dlami_id() {
    aws ssm get-parameter --region "$REGION" --name "$DLAMI_PARAM" --query Parameter.Value --output text
}

# Subnets of the zones that offer the instance type, one per line: "<zone> <subnet>".
pools() {
    local zones
    zones=$(ec2 describe-instance-type-offerings --location-type availability-zone \
        --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
        --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' '\n' | sort)
    for z in $zones; do
        echo "$z $(ec2 describe-subnets --filters "Name=availability-zone,Values=$z" \
            "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text)"
    done
}

# run_instance <name> <ami> <user-data file> <market: spot|on-demand> [root GB]
# Walks the pools; prints the instance id of the first that takes it.
run_instance() {
    local name=$1 ami=$2 user_data=$3 market=$4 root_gb=${5:-100}
    local market_args=()
    [ "$market" = spot ] && market_args=(--instance-market-options \
        'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
    while read -r zone subnet; do
        local out
        if out=$(ec2 run-instances --image-id "$ami" --instance-type "$INSTANCE_TYPE" \
                --subnet-id "$subnet" --security-group-ids "$(security_group_id)" \
                --iam-instance-profile "Name=$INSTANCE_PROFILE" \
                --instance-initiated-shutdown-behavior terminate \
                --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$root_gb,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
                --metadata-options 'HttpTokens=required,HttpPutResponseHopLimit=1' \
                --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name}]" \
                --user-data "file://$user_data" "${market_args[@]}" \
                --query 'Instances[0].InstanceId' --output text 2>&1); then
            echo "launched $out in $zone ($market)" >&2
            echo "$out"
            return 0
        fi
        echo "$zone refused: $out" >&2
    done < <(pools)
    return 1
}

# ssm_run <instance id> <command...>: run a shell command on the box, print its output.
# SSM_WAIT_SECONDS bounds the wait (default 240).
ssm_run() {
    local instance=$1; shift
    local cmd_id
    cmd_id=$(aws ssm send-command --region "$REGION" --instance-ids "$instance" \
        --document-name AWS-RunShellScript --timeout-seconds 7200 \
        --parameters "$(python3 -c 'import json, sys; print(json.dumps({"commands": [sys.argv[1]], "executionTimeout": ["7200"]}))' "$*")" \
        --query Command.CommandId --output text)
    for _ in $(seq 1 $(( ${SSM_WAIT_SECONDS:-240} / 2 ))); do
        local status
        status=$(aws ssm get-command-invocation --region "$REGION" --command-id "$cmd_id" \
            --instance-id "$instance" --query Status --output text 2>/dev/null || echo Pending)
        case "$status" in
            Success|Failed|Cancelled|TimedOut)
                aws ssm get-command-invocation --region "$REGION" --command-id "$cmd_id" \
                    --instance-id "$instance" --query '[StandardOutputContent,StandardErrorContent]' \
                    --output text
                [ "$status" = Success ]
                return ;;
        esac
        sleep 2
    done
    echo "ssm command $cmd_id still $status" >&2
    return 1
}
