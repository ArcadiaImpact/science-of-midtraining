#!/usr/bin/env bash
set -euo pipefail

stage=${1:?usage: run_stage.sh STAGE}
attempt_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$attempt_dir" rev-parse --show-toplevel)
python_bin=/workspace/arch-work-midtraining-monitor-evasion/.venv-research/bin/python

date -u +start_utc=%Y-%m-%dT%H:%M:%SZ
git -C "$repo_root" rev-parse HEAD | sed 's/^/execution_commit=/'
printf 'stage=%s\n' "$stage"
printf 'command=%s %s %s\n' "$python_bin" "$attempt_dir/experiment.py" "$stage"
printf 'output_directory=%s\n' "$attempt_dir/run"
printf 'non_secret_config_begin\n'
sed 's/^/config: /' "$attempt_dir/config.json"
printf 'non_secret_config_end\n'

set +e
"$python_bin" "$attempt_dir/experiment.py" "$stage"
status=$?
set -e
printf 'exit_status=%s\n' "$status"
date -u +end_utc=%Y-%m-%dT%H:%M:%SZ
exit "$status"
