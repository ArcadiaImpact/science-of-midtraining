#!/bin/bash
# Fresh, fail-closed held-out evaluator for one exact PR commit.
#
# STATIC ARTIFACT CONTRACT
# ------------------------
# PR-controlled code is never imported or executed. Only allow-listed inert
# files are copied into a read-only submission tree. The evaluator and any
# setup/dependencies must already exist in the trusted base branch/image.
# The unprivileged evaluator can read only that inert tree, the trusted scorer,
# and a read-only data root; it has an empty/minimal environment and no network.

set -uo pipefail

fatal_hold() {
  echo "CRITICAL: $*" >&2
  echo "CRITICAL: refusing to score or self-delete; host watcher must recover the exact pod" >&2
  exec sleep infinity
}

# Capture orchestration capabilities as unexported root-shell variables, then
# remove the inherited names before any repository-controlled bytes are read.
[ -n "${GH_TOKEN:-}" ] || fatal_hold "GH_TOKEN is required"
[ -n "${RUNPOD_API_KEY:-}" ] || fatal_hold "RUNPOD_API_KEY is required"
[ -n "${PRIVATE_LOG_UPLOAD_URL:-}" ] || fatal_hold "PRIVATE_LOG_UPLOAD_URL is required"
[ -n "${PRIVATE_LOG_VERIFY_URL:-}" ] || fatal_hold "PRIVATE_LOG_VERIFY_URL is required"
[ -n "${TRUSTED_GRADE_B64:-}" ] || fatal_hold "TRUSTED_GRADE_B64 is required"
[ -n "${PR_NUMBER:-}" ] || fatal_hold "PR_NUMBER is required"
[ -n "${PR_HEAD_SHA:-}" ] || fatal_hold "PR_HEAD_SHA is required"
[ -n "${REPO_OWNER:-}" ] || fatal_hold "REPO_OWNER is required"
[ -n "${REPO_NAME:-}" ] || fatal_hold "REPO_NAME is required"
[ -n "${TRUSTED_BASE_SHA:-}" ] || fatal_hold "TRUSTED_BASE_SHA is required"
ORCH_GH_TOKEN="$GH_TOKEN"
ORCH_RUNPOD_API_KEY="$RUNPOD_API_KEY"
ORCH_PRIVATE_LOG_UPLOAD_URL="$PRIVATE_LOG_UPLOAD_URL"
ORCH_PRIVATE_LOG_VERIFY_URL="$PRIVATE_LOG_VERIFY_URL"
ORCH_TRUSTED_GRADE_B64="$TRUSTED_GRADE_B64"
EXPECTED_TRUSTED_BASE_SHA="$TRUSTED_BASE_SHA"
unset GH_TOKEN RUNPOD_API_KEY PRIVATE_LOG_UPLOAD_URL PRIVATE_LOG_VERIFY_URL
unset TRUSTED_GRADE_B64 TRUSTED_BASE_SHA

case "$PR_HEAD_SHA" in
  *[!0-9a-f]*|'') fatal_hold "PR_HEAD_SHA must be a lowercase hexadecimal commit id" ;;
esac
[ "$(printf %s "$PR_HEAD_SHA" | wc -c)" -eq 40 ] || fatal_hold "PR_HEAD_SHA must be the exact 40-character GitHub commit id"
case "$EXPECTED_TRUSTED_BASE_SHA" in
  *[!0-9a-f]*|'') fatal_hold "TRUSTED_BASE_SHA must be a lowercase hexadecimal commit id" ;;
esac
[ "$(printf %s "$EXPECTED_TRUSTED_BASE_SHA" | wc -c)" -eq 40 ] \
  || fatal_hold "TRUSTED_BASE_SHA must be the exact trusted checkout commit id"

mkdir -p /workspace
LOG=/workspace/heldout-eval.log
exec > >(stdbuf -oL tee -a "$LOG") 2>&1
export PYTHONUNBUFFERED=1
echo "=== arch heldout-eval boot $(date -u) task=midtraining-monitor-evasion pr=$PR_NUMBER sha=$PR_HEAD_SHA ==="

# Root-only operational tooling. Unexported orchestration capabilities do not
# enter apt, sshd, git setup, or the eventual evaluator environment.
export DEBIAN_FRONTEND=noninteractive
timeout 600 apt-get -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update -qq \
  || fatal_hold "apt metadata refresh failed or timed out"
timeout 600 apt-get -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 \
  install -y -qq git curl jq ca-certificates openssh-server iptables python3 util-linux \
  || fatal_hold "tool install failed or timed out"
unset DEBIAN_FRONTEND

mkdir -p -m 0700 /root/.ssh /var/run/sshd
if [ -n "${PUBLIC_KEY:-}" ]; then
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 0600 /root/.ssh/authorized_keys
fi
ssh-keygen -A >/dev/null 2>&1 || fatal_hold "sshd host-key generation failed"
/usr/sbin/sshd || fatal_hold "sshd failed to start"

WORKDIR="/workspace/arch-heldout-midtraining-monitor-evasion-pr-$PR_NUMBER"
TRUSTED_ROOT="/opt/arch-trusted"
SUBMISSION_ROOT="/srv/arch-submission"
ARCH_DATA_ROOT="/srv/arch-grades"
OUT="/workspace/arch-heldout-result-$PR_NUMBER.json"

rm -f "$OUT"
if [ -e "$WORKDIR" ] || [ -e "$TRUSTED_ROOT" ] || [ -e "$SUBMISSION_ROOT" ] || [ -e "$ARCH_DATA_ROOT" ]; then
  fatal_hold "fresh-pod invariant violated: evaluator paths already exist"
fi

# Decode the trusted GitHub-side grader result before reading repository bytes,
# then erase its inherited representation. The OpenAI credential never enters
# this pod. The scorer later verifies that these grades bind to the exact inert
# artifact hashes copied from PR_HEAD_SHA.
mkdir -p -m 0700 "$ARCH_DATA_ROOT"
case "$ORCH_TRUSTED_GRADE_B64" in
  ''|*[!A-Za-z0-9+/=]*) fatal_hold "trusted grade payload is not canonical base64" ;;
esac
printf '%s' "$ORCH_TRUSTED_GRADE_B64" | base64 -d > "$ARCH_DATA_ROOT/grades.json" \
  || fatal_hold "trusted grade payload could not be decoded"
unset ORCH_TRUSTED_GRADE_B64
[ "$(stat -c %s "$ARCH_DATA_ROOT/grades.json" 2>/dev/null || echo 999999999)" -le 1048576 ] \
  || fatal_hold "trusted grade payload exceeds 1 MiB"
python3 - "$ARCH_DATA_ROOT/grades.json" "$PR_HEAD_SHA" <<'PY' \
  || fatal_hold "trusted grade payload is invalid or targets another commit"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
if not isinstance(value, dict) or value.get("schema_version") != 1:
    raise SystemExit(1)
if value.get("pr_head_sha") != sys.argv[2]:
    raise SystemExit(1)
if value.get("model") != "gpt-5.6-terra":
    raise SystemExit(1)
PY
chmod 0400 "$ARCH_DATA_ROOT/grades.json"

# Clone/fetch with an in-memory, per-process authorization header. No token is
# written to git config, credential stores, shell profiles, or disk.
GH_AUTH_B64=$(printf 'x-access-token:%s' "$ORCH_GH_TOKEN" | base64 -w0)
git_with_auth() {
  GIT_CONFIG_COUNT=1 \
  GIT_CONFIG_KEY_0=http.extraheader \
  GIT_CONFIG_VALUE_0="Authorization: Basic $GH_AUTH_B64" \
    timeout 300 git "$@"
}
git_with_auth clone --filter=blob:none --no-checkout \
  "https://github.com/${REPO_OWNER}/${REPO_NAME}.git" "$WORKDIR" \
  || fatal_hold "trusted clone failed"
cd "$WORKDIR" || fatal_hold "clone directory unavailable"
git_with_auth fetch --no-tags origin \
  "$PR_HEAD_SHA:refs/arch/pr-head" \
  "refs/heads/arch/midtraining-monitor-evasion:refs/remotes/origin/arch/midtraining-monitor-evasion" \
  || fatal_hold "PR SHA or trusted base fetch failed"
timeout 60 git checkout --detach refs/arch/pr-head || fatal_hold "exact PR SHA checkout failed"
test "$(git rev-parse HEAD)" = "$PR_HEAD_SHA" || fatal_hold "checked-out commit does not match requested PR SHA"

TRUSTED_BASE_SHA=$(git rev-parse "refs/remotes/origin/arch/midtraining-monitor-evasion^{commit}") \
  || fatal_hold "trusted base did not resolve to a commit"
test "$TRUSTED_BASE_SHA" = "$EXPECTED_TRUSTED_BASE_SHA" \
  || fatal_hold "trusted base moved or does not match workflow-pinned SHA"
timeout 60 git worktree add --detach "$TRUSTED_ROOT" "$TRUSTED_BASE_SHA" \
  || fatal_hold "trusted base worktree creation failed"
test "$(git -C "$TRUSTED_ROOT" rev-parse HEAD)" = "$TRUSTED_BASE_SHA" \
  || fatal_hold "trusted worktree SHA verification failed"

# Config paths are task-dir-relative. Support the established monorepo layout
# only when the named task subdirectory exists in both exact commits.
PR_TASK_ROOT="$WORKDIR"
TRUSTED_TASK_ROOT="$TRUSTED_ROOT"
TASK_PREFIX=""
if [ -d "$WORKDIR/midtraining-monitor-evasion" ] || [ -d "$TRUSTED_ROOT/midtraining-monitor-evasion" ]; then
  [ -d "$WORKDIR/midtraining-monitor-evasion" ] && [ -d "$TRUSTED_ROOT/midtraining-monitor-evasion" ] \
    || fatal_hold "task subdirectory exists in only one of PR/base commits"
  PR_TASK_ROOT="$WORKDIR/midtraining-monitor-evasion"
  TRUSTED_TASK_ROOT="$TRUSTED_ROOT/midtraining-monitor-evasion"
  TASK_PREFIX="midtraining-monitor-evasion/"
fi

TRUSTED_PATHS=(
  ".arch"
)
for _tp in "${TRUSTED_PATHS[@]}"; do
  case "$_tp" in
    ''|/*|../*|*/../*|*/..) fatal_hold "invalid trusted path: $_tp" ;;
  esac
  EXPECTED_OBJECT=$(git rev-parse "$TRUSTED_BASE_SHA:${TASK_PREFIX}${_tp}") \
    || fatal_hold "trusted path is absent from base: $_tp"
  RESTORED_OBJECT=$(git -C "$TRUSTED_ROOT" rev-parse "HEAD:${TASK_PREFIX}${_tp}") \
    || fatal_hold "trusted path restore failed: $_tp"
  test "$RESTORED_OBJECT" = "$EXPECTED_OBJECT" \
    || fatal_hold "trusted path hash mismatch: $_tp"
  echo "verified trusted path $_tp object=$EXPECTED_OBJECT"
done

# Copy only inert, allow-listed artifacts. The PR checkout becomes root-only
# before the evaluator account exists, so submitted Python/shell/project files
# cannot be read or imported by the scorer unless explicitly copied here.
SUBMISSION_ARTIFACTS=(
  "submission/results.json"
  "submission/curves.json"
  "submission/report.md"
  "submission/figures"
)
mkdir -p "$SUBMISSION_ROOT"
copy_static_artifact() {
  local rel="$1" src resolved dest
  case "$rel" in
    ''|/*|../*|*/../*|*/..) fatal_hold "invalid submission artifact path: $rel" ;;
  esac
  src="$PR_TASK_ROOT/$rel"
  [ -e "$src" ] || { echo "submitted artifact absent (allowed): $rel"; return 0; }
  resolved=$(realpath "$src") || fatal_hold "cannot resolve submitted artifact: $rel"
  case "$resolved" in
    "$PR_TASK_ROOT"/*) ;;
    *) fatal_hold "submitted artifact escapes PR checkout: $rel" ;;
  esac
  if find "$src" -type l -print -quit | grep -q .; then
    fatal_hold "symlink in submitted artifact tree: $rel"
  fi
  if find "$src" \( ! -type f -a ! -type d \) -print -quit | grep -q .; then
    fatal_hold "non-regular object in submitted artifact tree: $rel"
  fi
  dest="$SUBMISSION_ROOT/$rel"
  mkdir -p "$(dirname "$dest")"
  cp -a --no-preserve=ownership "$src" "$dest" \
    || fatal_hold "copy of submitted artifact failed: $rel"
}
for _artifact in "${SUBMISSION_ARTIFACTS[@]}"; do
  copy_static_artifact "$_artifact"
done
if find "$SUBMISSION_ROOT" -type l -print -quit | grep -q .; then
  fatal_hold "symlink reached static submission tree"
fi
chmod 0700 "$WORKDIR"
unset GH_AUTH_B64
unset -f git_with_auth

# Trusted scorer is immutable to the evaluator. No repo-controlled setup,
# pyproject, requirements file, module, or executable is ever invoked.
TRUSTED_EVAL="$TRUSTED_TASK_ROOT/.arch/eval.sh"
[ -f "$TRUSTED_EVAL" ] && [ ! -L "$TRUSTED_EVAL" ] \
  || fatal_hold "trusted evaluator is missing or a symlink"
[ ! -w "$TRUSTED_EVAL" ] || chmod a-w "$TRUSTED_EVAL"
chmod -R a-w "$TRUSTED_ROOT" "$SUBMISSION_ROOT" "$ARCH_DATA_ROOT"

if ! id arch-eval >/dev/null 2>&1; then
  useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin arch-eval \
    || fatal_hold "could not create evaluator account"
fi
EVAL_UID=$(id -u arch-eval)
chgrp -R arch-eval "$TRUSTED_ROOT" "$SUBMISSION_ROOT" "$ARCH_DATA_ROOT"
find "$TRUSTED_ROOT" "$SUBMISSION_ROOT" "$ARCH_DATA_ROOT" -type d -exec chmod 0550 {} +
# Preserve execute permission only on base-branch-trusted programs. Submitted
# artifacts are always inert data even if their Git mode claimed executable.
find "$TRUSTED_ROOT" -type f -perm /111 -exec chmod 0550 {} +
find "$TRUSTED_ROOT" -type f ! -perm /111 -exec chmod 0440 {} +
find "$SUBMISSION_ROOT" -type f -exec chmod 0440 {} +
find "$ARCH_DATA_ROOT" -type f -exec chmod 0440 {} +

# The only held-out data is the trusted Terra result produced in GitHub Actions.
[ -d "$ARCH_DATA_ROOT" ] && [ -n "$(find "$ARCH_DATA_ROOT" -mindepth 1 -print -quit)" ] \
  || fatal_hold "trusted grader result is missing or empty"
runuser --user arch-eval -- test -r "$ARCH_DATA_ROOT" \
  || fatal_hold "evaluator account cannot read the configured data root"

# Block both IPv4 and IPv6 for the evaluator UID. If the image cannot enforce
# both rules, refuse to evaluate rather than relying on a partial sandbox.
iptables -I OUTPUT 1 -m owner --uid-owner "$EVAL_UID" -j REJECT \
  || fatal_hold "network isolation unavailable; refusing to evaluate"
ip6tables -I OUTPUT 1 -m owner --uid-owner "$EVAL_UID" -j REJECT \
  || fatal_hold "network isolation unavailable; refusing to evaluate"
iptables -C OUTPUT -m owner --uid-owner "$EVAL_UID" -j REJECT \
  || fatal_hold "IPv4 evaluator network rule verification failed"
ip6tables -C OUTPUT -m owner --uid-owner "$EVAL_UID" -j REJECT \
  || fatal_hold "IPv6 evaluator network rule verification failed"

install -o arch-eval -g arch-eval -m 0600 /dev/null "$OUT"
echo "=== running trusted evaluator as uid=$EVAL_UID with env-i and no network ==="
cd "$TRUSTED_TASK_ROOT" || fatal_hold "trusted evaluator working directory unavailable"
set +e
EVAL_TIMEOUT_SECONDS=900
timeout --signal=TERM --kill-after=30 "${EVAL_TIMEOUT_SECONDS}s" \
  runuser --user arch-eval -- \
  env -i \
    HOME=/nonexistent \
    PATH=/usr/local/bin:/usr/bin:/bin \
    PYTHONNOUSERSITE=1 \
    ARCH_SUBMISSION_ROOT="$SUBMISSION_ROOT" \
    ARCH_DATA_ROOT="$ARCH_DATA_ROOT" \
    ARCH_EVAL_OUTPUT="$OUT" \
    bash "$TRUSTED_EVAL"
EVAL_EXIT=$?
set -e
echo "=== trusted evaluator exited $EVAL_EXIT ==="
if [ "$EVAL_EXIT" -eq 124 ]; then
  echo "ERROR: trusted evaluator exceeded ${EVAL_TIMEOUT_SECONDS}s wall-clock limit" >&2
fi
if [ "$EVAL_EXIT" -ne 0 ]; then
  printf '%s\n' '{"score":null,"metrics":null,"notes":"trusted evaluator failed or timed out"}' > "$OUT"
fi

# Parse evaluator output as inert JSON and constrain the public schema. The
# full file is size-bounded. Public metrics are reduced to finite scalar/null
# values; strings lose control characters and are truncated before markdown.
PUBLIC_OUT="/workspace/arch-heldout-public-$PR_NUMBER.json"
if [ "$(stat -c %s "$OUT" 2>/dev/null || echo 999999999)" -gt 1048576 ]; then
  echo "invalid evaluator output: file exceeds 1 MiB" >&2
  printf '%s\n' '{"score":null,"metrics":null,"notes":"trusted evaluator output exceeded size limit"}' > "$OUT"
  EVAL_EXIT=1
fi
if ! python3 - "$OUT" "$PUBLIC_OUT" <<'PY'
import json
import math
import sys

path = sys.argv[1]
public_path = sys.argv[2]
public_keys = []

def public_scalar(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        clean = "".join(ch if ch >= " " and ch != "\x7f" else " " for ch in value)
        return clean[:120]
    return None

try:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("top-level output must be an object")
    score = value.get("score")
    metrics = value.get("metrics")
    valid_score = score is None or (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(score)
    )
    if not valid_score or metrics is not None and not isinstance(metrics, dict):
        raise ValueError("invalid score/metrics schema")
    public = {"score": score}
    metrics = metrics or {}
    for key in public_keys:
        public[key] = public_scalar(metrics.get(key))
    encoded = json.dumps(public, ensure_ascii=True, separators=(",", ":"))
    if len(encoded.encode()) > 4096:
        raise ValueError("public projection exceeds 4 KiB")
    with open(public_path, "w", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
except Exception as exc:
    print(f"invalid evaluator output: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
then
  printf '%s\n' '{"score":null,"metrics":null,"notes":"trusted evaluator output was invalid"}' > "$OUT"
  printf '%s\n' '{"score":null}' > "$PUBLIC_OUT"
  EVAL_EXIT=1
fi
chown root:root "$OUT" "$PUBLIC_OUT"
chmod 0400 "$OUT" "$PUBLIC_OUT"

SCORE=$(jq -r '.score' "$OUT")
PUBLIC_JSON=$(jq -c . "$PUBLIC_OUT")
PUBLIC_BULLETS=$(python3 - "$PUBLIC_OUT" <<'PY'
import json
import sys

public_keys = []
with open(sys.argv[1], encoding="utf-8") as handle:
    public = json.load(handle)
for key in public_keys:
    rendered = json.dumps(public.get(key), ensure_ascii=True)
    rendered = rendered.replace("`", "'")[:160]
    print(f"- {key}: `{rendered}`")
PY
)
[ -z "$PUBLIC_BULLETS" ] || PUBLIC_BULLETS=$'\n'"$PUBLIC_BULLETS"

# Only the allow-listed public projection leaves through GitHub. The dedicated
# token enters a root-owned Python REST subprocess only; it is never persisted
# or placed in argv.
github_post() {
  local api_path="$1" body="$2"
  GITHUB_ORCH_TOKEN="$ORCH_GH_TOKEN" \
  GITHUB_API_PATH="$api_path" \
  GITHUB_API_BODY="$body" \
  python3 - <<'PY'
import os
import sys
import urllib.error
import urllib.request

request = urllib.request.Request(
    "https://api.github.com" + os.environ["GITHUB_API_PATH"],
    data=os.environ["GITHUB_API_BODY"].encode(),
    headers={
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer " + os.environ["GITHUB_ORCH_TOKEN"],
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status // 100 != 2:
            raise RuntimeError(f"unexpected HTTP {response.status}")
except urllib.error.HTTPError as error:
    print(f"GitHub publication failed: HTTP {error.code}", file=sys.stderr)
    raise SystemExit(1)
PY
}

COMMENT_BODY="$(cat <<EOF
## Held-out eval — automated

**Score:** \`${SCORE}\`${PUBLIC_BULLETS}

_Only the score and allow-listed public metrics are shown._
EOF
)"
COMMENT_JSON=$(jq -nc --arg body "$COMMENT_BODY" '{body: $body}')
github_post "/repos/${REPO_OWNER}/${REPO_NAME}/issues/${PR_NUMBER}/comments" "$COMMENT_JSON" \
  || fatal_hold "sanitized PR comment failed"

STATUS_DESCRIPTION="$PUBLIC_JSON"
if [ "$(printf %s "$STATUS_DESCRIPTION" | wc -c)" -gt 140 ]; then
  STATUS_DESCRIPTION=$(jq -c '{score: .score}' "$OUT")
fi
STATUS_STATE=success
[ "$EVAL_EXIT" -eq 0 ] && [ "$SCORE" != null ] || STATUS_STATE=failure
STATUS_JSON=$(jq -nc \
  --arg context arch-eval \
  --arg state "$STATUS_STATE" \
  --arg description "$STATUS_DESCRIPTION" \
  '{context: $context, state: $state, description: $description}')
github_post "/repos/${REPO_OWNER}/${REPO_NAME}/statuses/${PR_HEAD_SHA}" "$STATUS_JSON" \
  || fatal_hold "commit status publication failed"

# Keep every pod observable long enough for exact-ID publication and the host
# registrar/watcher handshake, even when evaluation finishes immediately.
NOW=$(date +%s)
BORN=${POD_BORN_EPOCH:-$NOW}
MIN_OBSERVATION=${MIN_OBSERVATION_SECONDS:-120}
REMAINING=$(( BORN + MIN_OBSERVATION - NOW ))
if [ "$REMAINING" -gt 0 ]; then
  echo "=== minimum observation window: sleeping ${REMAINING}s ==="
  sleep "$REMAINING"
fi

# Freeze a stable final snapshot after evaluation, publication, and the host
# observation window. Closing the tee input before copying prevents a tar/read
# race with the live log. Upload and a separate HEAD capability must agree on
# byte size before exact-pod deletion is permitted.
echo "=== publication complete; freezing final private evidence snapshot ==="
FINALIZE_LOG=/workspace/heldout-finalize.log
exec > >(stdbuf -oL tee -a "$FINALIZE_LOG") 2>&1
sleep 1
sync
LOG_SNAPSHOT=/workspace/heldout-eval.final.log
cp "$LOG" "$LOG_SNAPSHOT" || fatal_hold "could not freeze final log snapshot"
chown root:root "$LOG_SNAPSHOT"
chmod 0400 "$LOG_SNAPSHOT"
GRADE_SNAPSHOT=/workspace/trusted-grades.final.json
cp "$ARCH_DATA_ROOT/grades.json" "$GRADE_SNAPSHOT" \
  || fatal_hold "could not freeze trusted grader evidence"
chown root:root "$GRADE_SNAPSHOT"
chmod 0400 "$GRADE_SNAPSHOT"
PRIVATE_BUNDLE="/workspace/heldout-private-$PR_NUMBER.tgz"
tar -C /workspace -czf "$PRIVATE_BUNDLE" \
  "$(basename "$LOG_SNAPSHOT")" "$(basename "$OUT")" "$(basename "$PUBLIC_OUT")" \
  "$(basename "$GRADE_SNAPSHOT")" \
  || fatal_hold "could not create stable private log bundle"
LOCAL_BUNDLE_BYTES=$(stat -c %s "$PRIVATE_BUNDLE") \
  || fatal_hold "could not measure private log bundle"
case "$ORCH_PRIVATE_LOG_UPLOAD_URL $ORCH_PRIVATE_LOG_VERIFY_URL" in
  *$'\n'*|*'"'*) fatal_hold "private log capability URL contains unsafe characters" ;;
esac
for _private_url in "$ORCH_PRIVATE_LOG_UPLOAD_URL" "$ORCH_PRIVATE_LOG_VERIFY_URL"; do
  case "$_private_url" in
    https://*) ;;
    *) fatal_hold "private log capabilities must use HTTPS" ;;
  esac
done
if ! printf 'url = "%s"\nfail\nsilent\nshow-error\nconnect-timeout = 30\nmax-time = 900\n' "$ORCH_PRIVATE_LOG_UPLOAD_URL" \
  | curl --config - --upload-file "$PRIVATE_BUNDLE"; then
  fatal_hold "durable private log upload failed; refusing to self-delete"
fi
VERIFY_HEADERS=/workspace/private-log-verify.headers
if ! printf 'url = "%s"\nfail\nsilent\nshow-error\nconnect-timeout = 30\nmax-time = 60\n' "$ORCH_PRIVATE_LOG_VERIFY_URL" \
  | curl --config - --head --output /dev/null --dump-header "$VERIFY_HEADERS"; then
  fatal_hold "durable private log size verification request failed; refusing to self-delete"
fi
REMOTE_BUNDLE_BYTES=$(awk '
  BEGIN { IGNORECASE=1 }
  /^Content-Length:/ { gsub("\\r", "", $2); size=$2 }
  END { print size }
' "$VERIFY_HEADERS")
case "$REMOTE_BUNDLE_BYTES" in
  ''|*[!0-9]*) fatal_hold "durable private log verification omitted a valid Content-Length" ;;
esac
[ "$REMOTE_BUNDLE_BYTES" -eq "$LOCAL_BUNDLE_BYTES" ] \
  || fatal_hold "durable private log size mismatch local=$LOCAL_BUNDLE_BYTES remote=$REMOTE_BUNDLE_BYTES"
echo "=== durable private log verified: $REMOTE_BUNDLE_BYTES bytes ==="

[ -n "${RUNPOD_POD_ID:-}" ] || fatal_hold "RUNPOD_POD_ID missing; cannot self-delete exact pod"
if ! RUNPOD_API_KEY="$ORCH_RUNPOD_API_KEY" python3 - "$RUNPOD_POD_ID" <<'PY'
import os
import sys
import urllib.error
import urllib.request

pod_id = sys.argv[1]
request = urllib.request.Request(
    f"https://rest.runpod.io/v1/pods/{pod_id}",
    headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]},
    method="DELETE",
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status // 100 != 2:
            raise RuntimeError(f"unexpected HTTP {response.status}")
except urllib.error.HTTPError as error:
    if error.code != 404:
        print(f"exact pod deletion failed: HTTP {error.code}", file=sys.stderr)
        raise SystemExit(1)
PY
then
  fatal_hold "exact pod deletion failed after durable staging"
fi
echo "=== exact pod deletion accepted after durable staging ==="
exec sleep infinity
