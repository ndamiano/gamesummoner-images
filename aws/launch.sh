#!/usr/bin/env bash
# Boot one box from an AMI by hand, the way the scaler will: the env json is the instance
# user-data, which the entrypoint reads from IMDS. launch.sh <queue> <ami> <env.json> [--on-demand]
set -euo pipefail
. "$(dirname "$0")/common.sh"
queue=${1:?usage: launch.sh <queue> <ami> <env.json> [--on-demand]}
ami=${2:?usage: launch.sh <queue> <ami> <env.json> [--on-demand]}
env_json=${3:?usage: launch.sh <queue> <ami> <env.json> [--on-demand]}
market=spot
[ "${4:-}" = --on-demand ] && market=on-demand
python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$env_json"

instance=$(run_instance "gs-$queue-$(date -u +%Y%m%dT%H%M%S)" "$ami" "$env_json" "$market")
echo "$instance"
