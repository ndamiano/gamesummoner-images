#!/usr/bin/env bash
# Run one shell command on a box and print what it said: ssm.sh <instance id> <command...>
set -euo pipefail
. "$(dirname "$0")/common.sh"
instance=${1:?usage: ssm.sh <instance id> <command...>}; shift
ssm_run "$instance" "$@"
