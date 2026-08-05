#!/usr/bin/env bash
# Fail closed unless a chosen PR currently targets this task's exact ARCH branch.
set -euo pipefail

pr_number=${1:?usage: verify-pr-base.sh PR_NUMBER [CONFIG]}
config=${2:-.arch/config.toml}
[[ "$pr_number" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid PR number: $pr_number" >&2
  exit 2
}
[ -f "$config" ] || { echo "config not found: $config" >&2; exit 2; }

task_name=$(python3 - "$config" <<'PY'
import pathlib
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with pathlib.Path(sys.argv[1]).open("rb") as handle:
    print(tomllib.load(handle)["task_name"])
PY
)
[[ "$task_name" =~ ^[a-z0-9][a-z0-9-]{0,47}$ ]] || {
  echo "invalid task_name in config" >&2
  exit 2
}

expected_base="arch/$task_name"
actual_base=$(gh pr view "$pr_number" --json baseRefName --jq .baseRefName)
[ "$actual_base" = "$expected_base" ] || {
  echo "refusing merge: PR $pr_number base is '$actual_base', expected exact '$expected_base'" >&2
  exit 3
}
echo "verified PR $pr_number baseRefName=$actual_base"
