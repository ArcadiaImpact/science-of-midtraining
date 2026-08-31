#!/usr/bin/env bash
# Compatibility shim for a one-arm work unit under the v2 launch schema.
set -euo pipefail
OPS=$(cd "$(dirname "$0")" && pwd)
PROFILE=${1:?usage: launch_arm.sh <profile> <arm> <ssh-alias>}
ARM=${2:?usage: launch_arm.sh <profile> <arm> <ssh-alias>}
ALIAS=${3:?usage: launch_arm.sh <profile> <arm> <ssh-alias>}
exec "$OPS/launch_unit.sh" "$PROFILE" "$ARM" "$ALIAS"
