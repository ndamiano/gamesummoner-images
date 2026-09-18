#!/usr/bin/env bash
# Boot one llm box from an AMI by hand, the way the scaler will: the env json is the instance
# user-data, which the entrypoint reads from IMDS. launch.sh <ami> <env.json> [--on-demand]
set -euo pipefail
. "$(dirname "$0")/common.sh"
ami=${1:?usage: launch.sh <ami> <env.json> [--on-demand]}
env_json=${2:?usage: launch.sh <ami> <env.json> [--on-demand]}
market=spot
[ "${3:-}" = --on-demand ] && market=on-demand
python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$env_json"

instance=$(run_instance "gs-llm-$(date -u +%Y%m%dT%H%M%S)" "$ami" "$env_json" "$market")
echo "$instance"
