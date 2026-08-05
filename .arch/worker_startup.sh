#!/bin/bash
# Worker pod startup script. Rendered by `arch init` and injected as the
# pod's dockerStartCmd via base64. Runs as PID 1 in a RunPod container.
#
# Lessons baked in (each was an outage during live testing):
#   * set -e off — one transient failure shouldn't bootloop PID 1
#   * stream stdout/stderr to PID 1 AND to a file, line-buffered, so
#     `runpodctl logs <pod>` actually shows what the worker is doing
#   * start sshd in background so the researcher can SSH in to debug
#   * clone is idempotent — a valid checkout is refreshed; a partial checkout
#     is quarantined under a boot-stamped path for later diagnosis
#   * `codex exec` is one-shot; the loop below restarts it until the
#     deadline. PID 1 never falls off the end of the script.
#   * `codex exec` pins both model and reasoning effort for comparable runs.
#   * `--json` streams the native Codex event log directly to PID 1 and to a
#     timestamped file; no provider-specific transcript scraper is needed.

set -uo pipefail

# Split credentials at PID 1 before running any repository-controlled code.
# Operational RunPod and transcript credentials remain non-exported variables
# owned by the root supervisor. The research agent receives explicitly scoped
# GitHub/Hugging Face/task credentials. Its dedicated Codex key is present in
# the Codex parent process and is therefore also assumed compromised: Linux
# does not provide a reliable secret boundary between Codex and its same-UID
# shell children without a credential broker.
CODEX_AGENT_KEY="${CODEX_API_KEY:-}"
WORKER_GH_KEY="${WORKER_GH_TOKEN:-}"
WORKER_HF_KEY="${HF_TOKEN:-}"
TRANSCRIPT_ACCESS_KEY="${AWS_ACCESS_KEY_ID:-}"
TRANSCRIPT_SECRET_KEY="${AWS_SECRET_ACCESS_KEY:-}"
TRANSCRIPT_SESSION_TOKEN="${AWS_SESSION_TOKEN:-}"
TRANSCRIPT_BUCKET="${S3_BUCKET:-}"
TRANSCRIPT_REGION="${AWS_REGION:-eu-north-1}"
POD_ID="${RUNPOD_POD_ID:-}"
declare -A TASK_CREDENTIALS=()
TASK_CREDENTIAL_NAMES=(
  "TINKER_API_KEY"
)
for credential_name in "${TASK_CREDENTIAL_NAMES[@]}"; do
  TASK_CREDENTIALS["$credential_name"]="${!credential_name:-}"
  unset "$credential_name"
done
unset CODEX_API_KEY WORKER_GH_TOKEN HF_TOKEN RUNPOD_API_KEY RUNPOD_POD_ID
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_REGION S3_BUCKET

# Root-only redaction input. Durable transcripts are scrubbed for the literal
# and base64 forms of every credential known to the supervisor before upload.
install -m 0600 /dev/null /run/arch-redaction-values
for secret_value in \
  "$CODEX_AGENT_KEY" "$WORKER_GH_KEY" "$WORKER_HF_KEY" \
  "$TRANSCRIPT_ACCESS_KEY" "$TRANSCRIPT_SECRET_KEY" "$TRANSCRIPT_SESSION_TOKEN"; do
  [ -z "$secret_value" ] || printf '%s\n' "$secret_value" >> /run/arch-redaction-values
done
for credential_name in "${TASK_CREDENTIAL_NAMES[@]}"; do
  [ -z "${TASK_CREDENTIALS[$credential_name]}" ] \
    || printf '%s\n' "${TASK_CREDENTIALS[$credential_name]}" >> /run/arch-redaction-values
done

# ---- Visibility: tee everything line-buffered to a log file. ----
mkdir -p /workspace
chown root:root /workspace
chmod 0755 /workspace
exec > >(stdbuf -oL tee -a /workspace/arch-worker.log) 2>&1
export PYTHONUNBUFFERED=1
echo "=== arch-worker boot $(date -u) — task=midtraining-monitor-evasion ==="
BOOT_TS="$(date -u +%Y%m%dT%H%M%SZ)"
BOOT_RFC3339="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SOURCE_COMMIT=unknown

# ---- sshd for live debugging ----
mkdir -p /root/.ssh /var/run/sshd
if [ -n "${PUBLIC_KEY:-}" ]; then
  echo "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi
command -v sshd >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq openssh-server; }
ssh-keygen -A 2>/dev/null || true
(/usr/sbin/sshd -D &) && echo "sshd started in background"

# ---- Self-shutdown safety nets ----
# The deadline is measured from this worker's FIRST boot, not baked at init
# time. Init + fleet-debug time before the worker is healthy must NOT eat into
# iteration time — every worker gets the full budget from when IT starts.
# We persist the computed deadline to a file so a container/pod restart
# honors the original boot deadline instead of resetting the clock (which
# would let a bootlooping worker run forever). `at`/sleep fallbacks below call
# the RunPod REST DELETE because `shutdown -h now` is a no-op in a container.
WALL_CLOCK_BUDGET_S=86400
DEADLINE_FILE=/workspace/.arch_deadline_epoch
if [ -f "$DEADLINE_FILE" ] && [ -s "$DEADLINE_FILE" ]; then
  DEADLINE_EPOCH=$(cat "$DEADLINE_FILE")
  echo "Resuming deadline from $DEADLINE_FILE (set on first boot; survives restart)"
else
  DEADLINE_EPOCH=$(( $(date -u +%s) + WALL_CLOCK_BUDGET_S ))
  echo "$DEADLINE_EPOCH" > "$DEADLINE_FILE"
  echo "Deadline set from worker boot: ${WALL_CLOCK_BUDGET_S}s budget"
fi
NOW=$(date -u +%s)
SECS_TO_DEADLINE=$(( DEADLINE_EPOCH - NOW ))
if [ "$SECS_TO_DEADLINE" -lt 60 ]; then SECS_TO_DEADLINE=60; fi
export ARCH_DEADLINE_EPOCH="$DEADLINE_EPOCH"  # workers read this for accurate time-left
echo "Deadline: $(date -u -d @$DEADLINE_EPOCH) ($SECS_TO_DEADLINE s from now)"

# ---- Transcript persistence: mirror to S3 so it survives pod destruction ----
# The JSONL transcripts (/workspace/codex-transcripts/*.jsonl) and boot
# log live on the ephemeral container disk — they vanish the moment the pod is
# reaped or dies. If the researcher set S3_BUCKET, `arch init` puts it + creds
# into the pod env and we mirror the transcripts to a task-specific prefix,
# keyed by worker number, pod id, AND boot timestamp — the boot-timestamp leaf
# means a pod that's stopped and restarted (same RUNPOD_POD_ID) writes to a
# fresh prefix instead of overwriting the previous boot's arch-worker.log,
# which is exactly the forensic evidence a restart-worthy failure needs.
ARCH_WORKER_IDX="${ARCH_WORKER_INDEX:-unknown}"
TRANSCRIPT_PREFIX="arch2/midtraining-monitor-evasion/worker-${ARCH_WORKER_IDX}/${POD_ID:-nopod}/${BOOT_TS}"
aws_arch() {
  AWS_ACCESS_KEY_ID="$TRANSCRIPT_ACCESS_KEY" \
  AWS_SECRET_ACCESS_KEY="$TRANSCRIPT_SECRET_KEY" \
  AWS_SESSION_TOKEN="$TRANSCRIPT_SESSION_TOKEN" \
  AWS_DEFAULT_REGION="$TRANSCRIPT_REGION" \
    aws "$@"
}
_upload_transcripts_unlocked() {
  if [ -z "$TRANSCRIPT_BUCKET" ] || [ -z "$TRANSCRIPT_ACCESS_KEY" ] || [ -z "$TRANSCRIPT_SECRET_KEY" ]; then
    echo "ERROR: [transcript-s3] durable staging is not configured"
    return 1
  fi
  command -v aws >/dev/null 2>&1 || { echo "ERROR: [transcript-s3] aws CLI absent"; return 1; }
  local stage=/workspace/.arch-transcript-stage
  [ ! -L /workspace/arch-worker.log ] && [ -f /workspace/arch-worker.log ] \
    || { echo "ERROR: [transcript-s3] supervisor log is not a regular file"; return 1; }
  [ "$(stat -c %u /workspace/arch-worker.log)" -eq 0 ] \
    || { echo "ERROR: [transcript-s3] supervisor log is not root-owned"; return 1; }
  [ ! -L /workspace/codex-transcripts ] && [ -d /workspace/codex-transcripts ] \
    || { echo "ERROR: [transcript-s3] transcript root is unsafe"; return 1; }
  [ "$(stat -c %u /workspace/codex-transcripts)" -eq 0 ] \
    || { echo "ERROR: [transcript-s3] transcript root is not root-owned"; return 1; }
  if find /workspace/codex-transcripts -mindepth 1 \
      \( -type l -o \( ! -type f -a ! -type d \) \) -print -quit | grep -q .; then
    echo "ERROR: [transcript-s3] symlink or special object in transcript tree"
    return 1
  fi
  install -d -o root -g root -m 0700 "$stage"
  find "$stage" -mindepth 1 -depth -delete
  mkdir -p "$stage/codex-transcripts"
  cp -a /workspace/codex-transcripts/. "$stage/codex-transcripts/" \
    || { echo "ERROR: [transcript-s3] transcript snapshot copy failed"; return 1; }
  cp /workspace/arch-worker.log "$stage/arch-worker.log"
  env -i HOME=/root PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8 PYTHONNOUSERSITE=1 python3 -I - \
    "$stage" "$POD_ID" "$SOURCE_COMMIT" "$BOOT_RFC3339" /run/arch-redaction-values <<'PY'
import base64
import hashlib
import json
import os
import pathlib
import stat
import sys
from datetime import datetime, timezone

root = pathlib.Path(sys.argv[1])
secrets = {
    value
    for value in pathlib.Path(sys.argv[5]).read_bytes().splitlines()
    if len(value) >= 4
}
secrets |= {base64.b64encode(value) for value in secrets}
files = []
for path in sorted(root.rglob("*")):
    metadata = path.lstat()
    if stat.S_ISDIR(metadata.st_mode):
        continue
    if not stat.S_ISREG(metadata.st_mode) or path.name == "MANIFEST.json":
        raise SystemExit(f"unsafe staged transcript object: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise SystemExit(f"staged transcript changed while opening: {path}")
        content = handle.read()
    for secret in secrets:
        content = content.replace(secret, b"<redacted:credential>")
    fd = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW)
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)
    hasher = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    files.append({
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(content),
        "sha256": digest,
    })
manifest = {
    "schema_version": 1,
    "task_name": "midtraining-monitor-evasion",
    "session_id": "472e6a816d07ad3315d93d4b5c6b95d5",
    "pod_id": sys.argv[2],
    "pod_kind": "worker",
    "source_commit": sys.argv[3],
    "started_at": sys.argv[4],
    "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "files": files,
}
(root / "MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  if ! aws_arch s3 sync "$stage" "s3://${TRANSCRIPT_BUCKET}/${TRANSCRIPT_PREFIX}" \
      --no-follow-symlinks --only-show-errors; then
    echo "ERROR: [transcript-s3] upload FAILED"
    return 1
  fi
  echo "[transcript-s3] staged a manifest-backed snapshot"
}
upload_transcripts() (
  exec 9>/run/arch-transcript-upload.lock
  flock -w 120 9 || {
    echo "ERROR: [transcript-s3] timed out waiting for upload lock"
    return 1
  }
  _upload_transcripts_unlocked
)

# Verify every staged object's remote byte size before allowing host cleanup.
verify_transcript_upload() (
  exec 9>/run/arch-transcript-upload.lock
  flock -w 120 9 || return 1
  local stage=/workspace/.arch-transcript-stage rel expected actual manifest_size
  [ -s "$stage/MANIFEST.json" ] || return 1
  while IFS=$'\t' read -r rel expected; do
    actual=$(aws_arch s3api head-object \
      --bucket "$TRANSCRIPT_BUCKET" \
      --key "${TRANSCRIPT_PREFIX}/${rel}" \
      --query ContentLength --output text 2>/dev/null) || return 1
    [ "$actual" = "$expected" ] || return 1
  done < <(jq -r '.files[] | [.path, (.size_bytes | tostring)] | @tsv' "$stage/MANIFEST.json")
  manifest_size=$(stat -c %s "$stage/MANIFEST.json")
  actual=$(aws_arch s3api head-object \
    --bucket "$TRANSCRIPT_BUCKET" \
    --key "${TRANSCRIPT_PREFIX}/MANIFEST.json" \
    --query ContentLength --output text 2>/dev/null) || return 1
  [ "$actual" = "$manifest_size" ] || return 1
  return 0
)

prepare_for_host_cleanup() {
  local reason="$1"
  if ! mkdir /run/arch-cleanup-preparing 2>/dev/null; then
    echo "cleanup preparation is already running"
    exec sleep infinity
  fi
  echo "=== deadline $(date -u): reason=$reason pod=${POD_ID:-<unset>} ==="

  # Final transcript flush before the host is allowed to delete this pod.
  # A worker does not receive the account-level RunPod key.  It marks itself
  # safe for exact-ID host cleanup only after every staged object size verifies.
  if ! upload_transcripts; then
    echo "ERROR: [transcript-s3] upload failed — retrying once before teardown"
    sleep 5
    upload_transcripts || {
      echo "CRITICAL: durable transcript staging failed; refusing cleanup"
      exec sleep infinity
    }
  fi
  if ! verify_transcript_upload; then
    echo "CRITICAL: remote object-size verification failed; refusing cleanup"
    exec sleep infinity
  fi
  touch /workspace/.arch-ready-for-host-cleanup
  chmod 644 /workspace/.arch-ready-for-host-cleanup
  echo "READY_FOR_HOST_CLEANUP pod=${POD_ID:-<unset>} manifest=s3://${TRANSCRIPT_BUCKET}/${TRANSCRIPT_PREFIX}/MANIFEST.json"
  exec sleep infinity
}

# ---- Auth + tooling install ----
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl jq ca-certificates gnupg build-essential at python3 procps util-linux || true
mkdir -p -m 755 /etc/apt/keyrings
if ! command -v gh >/dev/null 2>&1; then
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
  chmod 644 /etc/apt/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    > /etc/apt/sources.list.d/github-cli.list
  apt-get update -qq && apt-get install -y -qq gh
fi

# ---- uv for Python tooling and project dependencies ----
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh \
    || echo "WARN: uv install failed"
  export PATH="/root/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "CRITICAL: uv is unavailable — the vendored ARCH runtime cannot run."
  exec sleep infinity
fi
if [ "$(command -v uv)" != /usr/local/bin/uv ]; then
  install -m 755 "$(command -v uv)" /usr/local/bin/uv
fi

# ---- awscli for transcript persistence (only if a bucket is configured) ----
# Installed lazily so workers without a transcript bucket don't pay for it.
# Used by upload_transcripts() above to mirror transcripts to S3.
if [ -n "$TRANSCRIPT_BUCKET" ] && ! command -v aws >/dev/null 2>&1; then
  uv tool install awscli 2>/dev/null \
    || echo "WARN: awscli install failed — transcript upload disabled"
fi
if [ -z "$TRANSCRIPT_BUCKET" ] || [ -z "$TRANSCRIPT_ACCESS_KEY" ] || [ -z "$TRANSCRIPT_SECRET_KEY" ]; then
  echo "CRITICAL: unattended GPU workers require private S3 transcript staging."
  exec sleep infinity
fi

# ---- Codex CLI ----
# Install the exact CLI recorded by init so reruns use the same client behavior.
if command -v codex >/dev/null 2>&1; then
  current_codex_version=$(codex --version | awk '{print $NF}')
else
  current_codex_version=""
fi
if [ "$current_codex_version" != "0.144.6" ]; then
  if ! command -v npm >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null 2>&1 || true
    apt-get install -y -qq nodejs || true
  fi
  npm install -g "@openai/codex@0.144.6" || echo "WARN: Codex CLI install failed"
fi
if ! command -v codex >/dev/null 2>&1; then
  echo "CRITICAL: Codex CLI not installed — the worker loop cannot run. Keeping container alive for debug."
  exec sleep infinity
fi
if [ -z "$CODEX_AGENT_KEY" ]; then
  echo "CRITICAL: CODEX_API_KEY is not set — Codex automation cannot authenticate."
  exec sleep infinity
fi
installed_codex_version=$(codex --version | awk '{print $NF}')
if [ "$installed_codex_version" != "0.144.6" ]; then
  echo "CRITICAL: Codex CLI version $installed_codex_version != pinned 0.144.6"
  exec sleep infinity
fi
echo "Codex CLI: $installed_codex_version; model=gpt-5.6-sol; reasoning=xhigh; native_multi_agent=disabled"

# hf_transfer is fast but the env flag without the package hard-fails every
# model download. Default it off; the task's setup.sh can `pip install
# hf_transfer` and re-enable it if it wants the speedup.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# ---- Idempotent clone of the task branch ----
# Use Basic-auth via http.extraheader — this is the only form that works
# reliably across `ghs_*`, `gho_*`, and PAT tokens. URL-embedded tokens hit
# git's "could not read Username" path with ephemeral GHA tokens.
mkdir -p /workspace && cd /workspace
WORKDIR="/workspace/arch-work-midtraining-monitor-evasion"
if [ -z "$WORKER_GH_KEY" ]; then
  echo "CRITICAL: WORKER_GH_TOKEN is not set. Keeping container alive for debug."
  exec sleep infinity
fi
GH_AUTH_B64=$(printf 'x-access-token:%s' "$WORKER_GH_KEY" | base64 -w0)
GIT_AUTH_HEADER="Authorization: Basic ${GH_AUTH_B64}"
export GIT_TERMINAL_PROMPT=0
if [ -d "$WORKDIR/.git" ]; then
  cd "$WORKDIR"
  git -c "http.extraheader=${GIT_AUTH_HEADER}" fetch origin "arch/midtraining-monitor-evasion" \
    || { echo "ERROR: git fetch failed. Keeping container alive for debug."; exec sleep infinity; }
  current_branch=$(git branch --show-current)
  if [ "$current_branch" != "arch/midtraining-monitor-evasion" ]; then
    echo "CRITICAL: existing worker checkout is on '$current_branch'; refusing to discard work."
    exec sleep infinity
  fi
else
  if [ -e "$WORKDIR" ]; then
    mv -- "$WORKDIR" "${WORKDIR}.partial-${BOOT_TS}"
  fi
  if ! git -c "http.extraheader=${GIT_AUTH_HEADER}" clone \
      --branch "arch/midtraining-monitor-evasion" \
      "https://github.com/ArcadiaImpact/science-of-midtraining.git" "$WORKDIR"; then
    echo "ERROR: git clone failed. Keeping container alive for debug."
    exec sleep infinity
  fi
  cd "$WORKDIR"
fi
unset GH_AUTH_B64 GIT_AUTH_HEADER
# ---- Monorepo: tasks live in a subfolder named by task_name, not repo root. ----
# Without this, .arch/, pyproject.toml and the eval shim aren't found and the
# worker runs against the wrong dir. Mirrors the held-out eval startup.
if [ -d "midtraining-monitor-evasion" ]; then cd "midtraining-monitor-evasion"; echo "cd into task subdir: $(pwd)"; fi

# The agent is unprivileged. Operational credentials above remain only in the
# root supervisor. A credential helper stores a variable reference, never the
# token value, in git config.
id archagent >/dev/null 2>&1 || useradd --create-home --home-dir /home/archagent --shell /bin/bash archagent
AGENT_UID=$(id -u archagent)
AGENT_GID=$(id -g archagent)
chown -R archagent:archagent "$WORKDIR"
install -d -o archagent -g archagent -m 700 /home/archagent/.codex
install -d -o root -g root -m 0700 /workspace/codex-transcripts
install -d -m 755 /usr/local/libexec
install -m 755 /dev/stdin /usr/local/libexec/arch-git-credential <<'CREDENTIAL_HELPER'
#!/bin/sh
if [ "${1:-}" = get ]; then
  printf 'username=x-access-token\npassword=%s\n' "$WORKER_GH_TOKEN"
fi
CREDENTIAL_HELPER
setpriv --reuid="$AGENT_UID" --regid="$AGENT_GID" --init-groups \
  git config --local credential.helper /usr/local/libexec/arch-git-credential

AGENT_ENV=(
  env -i
  "HOME=/home/archagent"
  "USER=archagent"
  "LOGNAME=archagent"
  "SHELL=/bin/bash"
  "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  "LANG=C.UTF-8"
  "GIT_TERMINAL_PROMPT=0"
  "WORKER_GH_TOKEN=$WORKER_GH_KEY"
  "GH_TOKEN=$WORKER_GH_KEY"
  "GITHUB_TOKEN=$WORKER_GH_KEY"
  "HF_TOKEN=$WORKER_HF_KEY"
  "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  "ARCH_DEADLINE_EPOCH=$ARCH_DEADLINE_EPOCH"
)
for credential_name in "${TASK_CREDENTIAL_NAMES[@]}"; do
  AGENT_ENV+=("$credential_name=${TASK_CREDENTIALS[$credential_name]}")
done
# Serialize the exact agent environment to a root-only NUL-delimited file. The
# root launcher starts under a fixed environment, reads this file, drops UID/GID,
# and only then installs these potentially interpreter-controlling values.
install -m 0600 /dev/null /run/arch-agent-env
for env_entry in "${AGENT_ENV[@]:2}"; do
  printf '%s\0%s\0' "${env_entry%%=*}" "${env_entry#*=}" >> /run/arch-agent-env
done
run_as_agent() {
  setpriv --reuid="$AGENT_UID" --regid="$AGENT_GID" --init-groups "${AGENT_ENV[@]}" "$@"
}

# Enforce the wall-clock deadline even if setup or a Codex turn hangs. Only
# processes owned by the dedicated agent UID are targeted; PID 1 and root
# supervisor processes are never part of this set.
deadline_guard() {
  local now wait_s pid
  now=$(date -u +%s)
  wait_s=$(( DEADLINE_EPOCH - now ))
  [ "$wait_s" -gt 0 ] && sleep "$wait_s"
  while read -r pid; do
    case "$pid" in
      ''|*[!0-9]*|1) continue ;;
      *) kill -TERM "$pid" 2>/dev/null || true ;;
    esac
  done < <(ps -u "$AGENT_UID" -o pid=)
  sleep 20
  while read -r pid; do
    case "$pid" in
      ''|*[!0-9]*|1) continue ;;
      *) kill -KILL "$pid" 2>/dev/null || true ;;
    esac
  done < <(ps -u "$AGENT_UID" -o pid=)
  prepare_for_host_cleanup "wall-clock-deadline"
}
deadline_guard &

# ---- Project deps ----
# pyproject may live in repro/ (eval shim expects repro/.venv); handle that first.
if   [ -f .arch/setup.sh ];        then run_as_agent bash .arch/setup.sh   || echo "WARN: .arch/setup.sh failed"
elif [ -f repro/pyproject.toml ];  then ( cd repro && run_as_agent uv sync ) || echo "WARN: repro uv sync failed"
elif [ -f pyproject.toml ];        then run_as_agent uv sync               || echo "WARN: uv sync failed"
elif [ -f requirements.txt ];      then run_as_agent uv venv .venv && run_as_agent uv pip install --python .venv/bin/python -r requirements.txt || echo "WARN: dependency setup failed"
fi
if [ ! -x scripts/arch2 ]; then
  echo "CRITICAL: scripts/arch2 is missing or not executable. Keeping container alive for debug."
  exec sleep infinity
fi
run_as_agent scripts/arch2 --help >/dev/null || {
  echo "CRITICAL: vendored ARCH runtime is not runnable. Keeping container alive for debug."
  exec sleep infinity
}
echo "Task commit: $(git rev-parse HEAD)"
SOURCE_COMMIT=$(git rev-parse HEAD)
echo "Worker prompt sha256: $(sha256sum .arch/worker_README.md | awk '{print $1}')"

# ---- Periodic transcript upload (bounds loss from hard death / OOM) ----
if [ -n "$TRANSCRIPT_BUCKET" ]; then
  ( while true; do sleep "${ARCH_TRANSCRIPT_SYNC_INTERVAL_S:-180}"; upload_transcripts; done ) &
fi

# The launcher limits the Codex key to the Codex process rather than every
# setup/eval command, then permanently drops to archagent. Same-UID Codex shell
# children can still inspect their parent on Linux, so init must provision this
# as a dedicated, revocable, budget/rate-limited key and record that it is
# treated as compromised.
install -m 600 /dev/null /run/arch-codex-api-key
printf '%s' "$CODEX_AGENT_KEY" > /run/arch-codex-api-key
install -m 700 /dev/stdin /usr/local/libexec/arch-codex-launch <<'CODEX_LAUNCHER'
#!/usr/bin/python3 -I
import os
import pwd
import sys

account = pwd.getpwnam("archagent")
with open("/run/arch-codex-api-key", encoding="utf-8") as handle:
    key = handle.read()
with open("/run/arch-agent-env", "rb") as handle:
    fields = handle.read().split(b"\0")
if fields[-1] != b"" or len(fields) % 2 != 1:
    raise RuntimeError("invalid agent environment file")
child_env = {
    fields[index].decode(): fields[index + 1].decode()
    for index in range(0, len(fields) - 1, 2)
}
os.setgroups([])
os.setgid(account.pw_gid)
os.setuid(account.pw_uid)
child_env["CODEX_API_KEY"] = key
os.chdir(sys.argv[1])
os.execvpe(sys.argv[2], sys.argv[2:], child_env)
CODEX_LAUNCHER

# ---- Main loop: keep `codex exec` alive until the deadline ----
# `codex exec` exits cleanly after one task; without this loop, the pod
# would idle until self-shutdown after the first PR. We restart it with
# the worker README as the prompt, until 60s before the deadline.
RESTART_GUARD_S=60
ITER=0
while :; do
  NOW=$(date -u +%s)
  REMAINING=$(( DEADLINE_EPOCH - NOW - RESTART_GUARD_S ))
  if [ "$REMAINING" -le 0 ]; then
    echo "Deadline reached; not restarting Codex."
    break
  fi
  ITER=$(( ITER + 1 ))
  echo "=== codex exec iteration $ITER, $REMAINING s remaining ==="
  transcript="/workspace/codex-transcripts/iter-${ITER}-${BOOT_TS}.jsonl"
  mkdir -p /workspace/codex-transcripts
  timeout --foreground --signal=TERM --kill-after=30 "$REMAINING" \
    env -i HOME=/root USER=root LOGNAME=root \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
    /usr/local/libexec/arch-codex-launch "$PWD" codex exec \
    --json \
    --color never \
    --ephemeral \
    --model "gpt-5.6-sol" \
    --sandbox danger-full-access \
    --ignore-user-config \
    --strict-config \
    --disable multi_agent \
    -c 'approval_policy="never"' \
    -c 'model_reasoning_effort="xhigh"' \
    -c 'shell_environment_policy.include_only=["HOME","USER","LOGNAME","SHELL","PATH","LANG","GIT_TERMINAL_PROMPT","WORKER_GH_TOKEN","GH_TOKEN","GITHUB_TOKEN","HF_TOKEN","HF_HUB_ENABLE_HF_TRANSFER","ARCH_DEADLINE_EPOCH","TINKER_API_KEY"]' \
    - < .arch/worker_README.md | tee -a "$transcript"
  codex_rc=${PIPESTATUS[0]}
  if [ "$codex_rc" -ne 0 ]; then
    echo "WARN: Codex exited non-zero ($codex_rc) on iteration $ITER"
  fi
  # Avoid hot-looping on rapid failure.
  sleep 10
done

# Hand off to the deadline timer; do NOT exit (RunPod would restart us).
prepare_for_host_cleanup "codex-loop-exhausted"
