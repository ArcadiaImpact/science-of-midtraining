#!/bin/bash
# Held-out eval pod startup. Spawned by .github/workflows/arch-eval.yml
# when a labeled PR is opened / ready_for_review / synchronized / reopened.
#
# What this is:
#   * the authoritative-eval side of the iteration/eval split
#   * the only place the held-out data is mounted or visible
#   * the only place that posts the held-out score back to the PR
#
# Leak prevention:
#   * The PR comment and the check_run we post are restricted to:
#       - `score`  (always public — workers need it to iterate)
#       - the metric keys the researcher whitelisted as `public_metrics`
#         in `.arch/config.toml` (rendered into this script at init time)
#   * Nothing else is published. No model name, no dataset shape, no
#     intermediate metrics — anything a worker could use to overfit.
#   * The full eval JSON stays on the pod for debug; only the filtered
#     view goes back to GitHub. Pod filesystem is destroyed on self-delete.
#
# Inputs (env vars injected by the workflow):
#   HF_TOKEN, GH_TOKEN, ANTHROPIC_API_KEY, OPENROUTER_API_KEY  — credentials
#   PR_NUMBER, PR_HEAD_SHA, REPO_OWNER, REPO_NAME
#   RUNPOD_API_KEY, RUNPOD_POD_ID         — for self-delete

set -uo pipefail   # NOT -e — we want failures to fall through to debug-sleep

# ---- Visibility ----
mkdir -p /workspace
exec > >(stdbuf -oL tee -a /workspace/heldout-eval.log) 2>&1
export PYTHONUNBUFFERED=1
echo "=== arch heldout-eval boot $(date -u) — task=midtrain-sft-interaction-1b pr=${PR_NUMBER:-?} ==="

# ---- sshd for live debugging ----
mkdir -p /root/.ssh /var/run/sshd
if [ -n "${PUBLIC_KEY:-}" ]; then
  echo "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
fi
command -v sshd >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq openssh-server; }
ssh-keygen -A 2>/dev/null || true
(/usr/sbin/sshd -D &) && echo "sshd started in background"

# ---- env presence (warn, don't fatal) ----
for v in HF_TOKEN GH_TOKEN ANTHROPIC_API_KEY OPENROUTER_API_KEY PR_NUMBER PR_HEAD_SHA REPO_OWNER REPO_NAME; do
  if [ -z "${!v:-}" ]; then echo "WARN: $v is missing"; fi
done

# ---- Robust self-termination ----
# `shutdown -h now` is a no-op without systemd inside a RunPod container,
# and PID 1 falling off triggers a container restart → bootloop. The only
# reliable way out is the RunPod API DELETE, and even that occasionally
# returns 403 for opaque edge reasons — so we layer fallbacks.
self_terminate() {
  local reason="$1"
  echo "=== self-terminate $(date -u): reason=$reason pod=${RUNPOD_POD_ID:-<unset>} ==="

  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "CRITICAL: cannot self-delete — RUNPOD_API_KEY or RUNPOD_POD_ID unset."
    echo "CRITICAL: this container will keep billing until manual cleanup."
    exec sleep infinity
  fi

  local attempt code body
  body=""

  # 1) REST DELETE with exponential backoff (~4.5 min worst case).
  for attempt in 1 2 3 4 5; do
    code=$(curl -sS -o /tmp/rp_delete_body -w '%{http_code}' \
      -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      --max-time 30 || echo 000)
    body=$(cat /tmp/rp_delete_body 2>/dev/null || echo '')
    echo "[self-terminate] REST DELETE attempt $attempt → HTTP $code: ${body:0:200}"
    case "$code" in
      2*)    echo "[self-terminate] REST DELETE succeeded"; exec sleep infinity ;;
      404)   echo "[self-terminate] pod already gone (404)"; exec sleep infinity ;;
    esac
    sleep $((5 * attempt * attempt))
  done

  # 2) GraphQL podTerminate fallback.
  echo "[self-terminate] REST exhausted; trying GraphQL podTerminate"
  local gql_payload
  gql_payload=$(printf '{"query":"mutation { podTerminate(input: { podId: \\"%s\\" }) }"}' "$RUNPOD_POD_ID")
  for attempt in 1 2 3; do
    code=$(curl -sS -o /tmp/rp_gql_body -w '%{http_code}' \
      -X POST "https://api.runpod.io/graphql" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -d "$gql_payload" \
      --max-time 30 || echo 000)
    body=$(cat /tmp/rp_gql_body 2>/dev/null || echo '')
    echo "[self-terminate] GraphQL attempt $attempt → HTTP $code: ${body:0:200}"
    if [ "${code:0:1}" = "2" ] && ! echo "$body" | grep -q '"errors"'; then
      echo "[self-terminate] GraphQL podTerminate succeeded"
      exec sleep infinity
    fi
    sleep $((10 * attempt))
  done

  # 3) runpodctl fallback.
  if command -v runpodctl >/dev/null 2>&1; then
    echo "[self-terminate] trying runpodctl remove pod"
    if RUNPOD_API_KEY="$RUNPOD_API_KEY" runpodctl remove pod "$RUNPOD_POD_ID"; then
      echo "[self-terminate] runpodctl removed pod"
      exec sleep infinity
    fi
  fi

  # 4) Loud failure. Sleep so PID 1 stays alive (exit would bootloop).
  echo "CRITICAL: ALL self-terminate paths failed for pod ${RUNPOD_POD_ID}."
  echo "CRITICAL: container is still billing. Last response body:"
  echo "  ${body:-<none>}"
  echo "CRITICAL: manual cleanup:"
  echo "  curl -X DELETE https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID} \\"
  echo "    -H 'Authorization: Bearer \$RUNPOD_API_KEY'"
  echo "CRITICAL: or: runpodctl remove pod ${RUNPOD_POD_ID}"
  exec sleep infinity
}

# 4h hard cap — the API self-delete at the bottom is the primary path; this
# fires only if the eval hangs and we never reach the explicit terminate.
( sleep 14400 && self_terminate "4h-safety-net" ) &

# ---- Tooling ----
# apt is treated as BEST-EFFORT, never blocking. Two of four canary pods in
# EU-RO-1 hung indefinitely inside `apt-get update` (sleeping on mirror network
# I/O for 13+ minutes on a host at load ~8), which stalls the pod before it ever
# clones and looks identical to a dead pod. So: every apt call is wrapped in a
# hard `timeout`, retries/timeouts are bounded, failure is tolerated, and we only
# install what is actually missing. `jq` and `gh` -- the two tools this script
# genuinely cannot finish without, since they parse the eval JSON and post the
# score -- have direct-binary fallbacks that bypass apt entirely.
export DEBIAN_FRONTEND=noninteractive
APT_OPTS="-o Acquire::Retries=2 -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15"

MISSING=""
for _t in git curl jq; do command -v "$_t" >/dev/null 2>&1 || MISSING="$MISSING $_t"; done
if [ -n "$MISSING" ]; then
  echo "apt: need$MISSING — trying apt-get (best effort, hard 300s cap)"
  timeout 300 apt-get update -qq $APT_OPTS || echo "WARN: apt-get update timed out/failed — continuing"
  timeout 300 apt-get install -y -qq $APT_OPTS $MISSING || echo "WARN: apt-get install timed out/failed — continuing"
else
  echo "apt: git/curl/jq already present, skipping apt entirely"
fi

# jq fallback: static binary. Without jq the score cannot be parsed or published.
if ! command -v jq >/dev/null 2>&1; then
  echo "jq still missing — fetching static binary"
  curl -fsSL --retry 3 --max-time 120 -o /usr/local/bin/jq \
    https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-linux-amd64 \
    && chmod +x /usr/local/bin/jq && echo "jq installed via direct binary"
fi

# gh fallback: release tarball, no apt repo dance (that path needed TWO more
# apt-get update calls, i.e. two more chances to hang).
if ! command -v gh >/dev/null 2>&1; then
  echo "gh missing — fetching release tarball"
  GH_VER="2.62.0"
  if curl -fsSL --retry 3 --max-time 180 -o /tmp/gh.tgz \
      "https://github.com/cli/cli/releases/download/v${GH_VER}/gh_${GH_VER}_linux_amd64.tar.gz"; then
    tar -xzf /tmp/gh.tgz -C /tmp \
      && install -m 755 "/tmp/gh_${GH_VER}_linux_amd64/bin/gh" /usr/local/bin/gh \
      && echo "gh installed via tarball"
  else
    echo "WARN: gh tarball download failed"
  fi
fi

# Hard requirement check: without these the pod cannot publish a score, so fail
# loudly HERE rather than after burning the whole eval.
for _t in git jq gh; do
  command -v "$_t" >/dev/null 2>&1 || {
    echo "ERROR: required tool '$_t' unavailable after apt + direct-download fallbacks."
    echo "ERROR: the pod cannot clone/parse/publish, so keeping container alive for SSH debug."
    exec sleep infinity
  }
done
echo "tooling ready: git=$(command -v git) jq=$(command -v jq) gh=$(command -v gh)"

# ---- Idempotent clone at PR head SHA, with Basic-auth git header ----
mkdir -p /workspace && cd /workspace
rm -rf work
GH_AUTH_B64=$(printf 'x-access-token:%s' "${GH_TOKEN}" | base64 -w0)
GIT_AUTH_HEADER="Authorization: Basic ${GH_AUTH_B64}"
export GIT_TERMINAL_PROMPT=0
if ! git -c "http.extraheader=${GIT_AUTH_HEADER}" clone \
    "https://github.com/${REPO_OWNER}/${REPO_NAME}.git" work; then
  echo "ERROR: git clone failed. Keeping container alive for debug."
  exec sleep infinity
fi
cd work
git config --local "http.extraheader" "${GIT_AUTH_HEADER}"
git fetch origin "${PR_HEAD_SHA}" || true
git checkout "${PR_HEAD_SHA}" || true
echo "${GH_TOKEN}" | gh auth login --with-token || echo "WARN: gh auth login failed"

# ---- Monorepo: tasks live in a subfolder named by task_name, not repo root. ----
# Without this, .arch/, pyproject.toml and the eval shim aren't found and the
# eval runs against the wrong dir → every PR scores null. Mirrors the worker
# startup. From here on, cwd is the task dir; trusted-path restore and the
# deps/eval steps below resolve relative to it.
if [ -d "midtrain-sft-interaction-1b" ]; then cd "midtrain-sft-interaction-1b"; echo "cd into task subdir: $(pwd)"; fi

# ---- Restore the trusted scorer from the BASE branch (anti-gaming) ----
# The PR head is checked out above, so a PR could otherwise alter its OWN
# scoring code, and a scorer fix on base would never reach PRs branched
# before it. Overwrite the scoring paths from the trusted base branch; the
# PR's results/figures (everything NOT listed in trusted_paths) stay from PR
# head. Pathspecs resolve relative to cwd (the task dir), matching how
# trusted_paths is written in .arch/config.toml.
git fetch origin "arch/midtrain-sft-interaction-1b" --depth=1 2>/dev/null || true
for _tp in .arch; do
  if git checkout "origin/arch/midtrain-sft-interaction-1b" -- "$_tp" 2>/dev/null; then
    echo "restored trusted scorer path from base: $_tp"
  else
    echo "WARN: could not restore trusted path '$_tp' from base (may not exist yet)"
  fi
done

# ---- HF auth ----
mkdir -p /root/.cache/huggingface
echo -n "${HF_TOKEN:-}" > /root/.cache/huggingface/token
export HF_TOKEN

# ---- Project deps ----
if   [ -f .arch/setup.sh ];      then bash .arch/setup.sh   || { echo "ERROR: setup.sh failed"; exec sleep infinity; }
elif [ -f pyproject.toml ];      then uv sync               || pip install -e . || { echo "ERROR: uv sync failed"; exec sleep infinity; }
elif [ -f requirements.txt ];    then pip install -r requirements.txt || { echo "ERROR: pip failed"; exec sleep infinity; }
fi

# ---- Run authoritative eval against held-out data ----
# Prefer the mounted held-out volume; fall back to the in-repo reference data
# if it isn't mounted/populated. A held-out volume pinned to a no-capacity
# datacenter would otherwise strand every eval pod (it must run in the
# volume's DC) — see arch-init's capacity probe. The fallback lets the pod
# provision in ANY DC and still produce a (clearly-labeled) score against the
# in-repo reference rather than zombieing.
OUT="/tmp/arch_heldout_${PR_NUMBER}.json"
HELDOUT_FALLBACK=0
if [ -d /mnt/arch_data ] && [ -n "$(ls -A /mnt/arch_data 2>/dev/null)" ]; then
  export ARCH_DATA_ROOT="/mnt/arch_data"
  echo "Using held-out volume at /mnt/arch_data"
else
  export ARCH_DATA_ROOT="$(pwd)/data/public"
  HELDOUT_FALLBACK=1
  echo "WARN: held-out volume not mounted/populated — falling back to in-repo reference data at $ARCH_DATA_ROOT"
fi
export ARCH_EVAL_OUTPUT="$OUT"
set +e
bash .arch/eval.sh
EVAL_EXIT=$?
set -e
echo "=== eval exited with code $EVAL_EXIT ==="

# Ensure OUT exists even on hard failure so we can still post a status.
if [ ! -f "$OUT" ]; then
  cat > "$OUT" <<EOF
{"score": null, "metrics": null, "notes": "ERROR: eval failed to produce output (exit=$EVAL_EXIT). See pod logs."}
EOF
fi

# ---- Build the *public* view ----
# Only the score + researcher-whitelisted metric keys are published. Anything
# else stays in the pod logs and is wiped on self-delete.
SCORE="$(jq -r '.score' "$OUT")"

# Render whitelisted keys as a jq projection. `public_metrics` is an empty
# list by default, which means: publish only the score.
# The commit-status description is the ONE machine-readable transport, and it is
# capped by GitHub at 140 chars. Two bugs found by the canary:
#   1. The rendered flat projection was 241 chars -> HTTP 422, so the score never
#      reached the leaderboard even though the eval succeeded (exit=0, score=0.0).
#   2. `arch findings` parses this as {"score":..,"metrics":{..},"notes":..} and
#      reads payload["metrics"]; a FLAT object meant metrics were silently
#      unavailable to the leaderboard regardless of length.
# So: emit the nested shape the parser expects, with short metric keys, and if it
# still exceeds the cap, degrade to score-only rather than failing to publish.
# Publishing the score is non-negotiable; the extras are not. The full
# whitelisted breakdown is in the PR comment above for humans.
PUBLIC_JSON=$(jq -c '{score: .score, metrics: {
    il: (.metrics["interaction_logit"] // null),
    lo: (.metrics["interaction_ci_low"] // null),
    hi: (.metrics["interaction_ci_high"] // null),
    n:  (.metrics["n_items"] // null),
    pd: (.metrics["paraphrase_delta"] // null),
    cd: (.metrics["capability_delta"] // null),
    g:  .metrics["gate_passed"],
    gf: (.metrics["gate_failed_stage"] // null)
  }}' "$OUT")
if [ "${#PUBLIC_JSON}" -gt 140 ]; then
  echo "NOTE: full status payload is ${#PUBLIC_JSON} chars (>140 cap) — degrading to score+gate"
  PUBLIC_JSON=$(jq -c '{score: .score, metrics: {g: .metrics["gate_passed"], gf: (.metrics["gate_failed_stage"] // null)}}' "$OUT")
fi
if [ "${#PUBLIC_JSON}" -gt 140 ]; then
  echo "NOTE: still ${#PUBLIC_JSON} chars — degrading to score only so the score still publishes"
  PUBLIC_JSON=$(jq -c '{score: .score}' "$OUT")
fi
echo "status payload (${#PUBLIC_JSON} chars): $PUBLIC_JSON"

# Human-readable bullet list for the PR comment. Empty if no whitelist.
PUBLIC_BULLETS=""
PUBLIC_BULLETS+=$'\n'"- score: \`$(jq -r '.metrics["score"] // "n/a"' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- interaction_logit: \`$(jq -r '.metrics["interaction_logit"] // "n/a"' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- interaction_ci_low: \`$(jq -r '.metrics["interaction_ci_low"] // "n/a"' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- interaction_ci_high: \`$(jq -r '.metrics["interaction_ci_high"] // "n/a"' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- interaction_rate: \`$(jq -r '.metrics["interaction_rate"] // "n/a"' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- n_items: \`$(jq -r '.metrics["n_items"] // "n/a"' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- paraphrase_delta: \`$(jq -r '.metrics["paraphrase_delta"] // "n/a"' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- capability_delta: \`$(jq -r '.metrics["capability_delta"] // "n/a"' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- gate_passed: \`$(jq -r 'if .metrics|has("gate_passed") then .metrics["gate_passed"] else "n/a" end' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- gate_failed_stage: \`$(jq -r '.metrics["gate_failed_stage"] // "n/a"' "$OUT")\`"

# ---- Post PR comment (sanitized) ----
FALLBACK_NOTE=""
if [ "$HELDOUT_FALLBACK" -eq 1 ]; then
  FALLBACK_NOTE=$'\n\n⚠️ _Scored against in-repo **reference** data, not the held-out volume (it was unavailable). Treat this as provisional, not an authoritative held-out score._'
fi
gh pr comment "$PR_NUMBER" --repo "${REPO_OWNER}/${REPO_NAME}" --body "$(cat <<EOF
## Held-out eval — automated

**Score:** \`${SCORE}\`${PUBLIC_BULLETS}

_Held-out details (model identity, dataset shape, full metric breakdown) are intentionally not shown to keep iteration honest._${FALLBACK_NOTE}
EOF
)" || echo "WARN: gh pr comment failed (continuing)"

# ---- Post commit status (the SINGLE machine-readable score transport) ----
# This is what `arch findings` reads. We use a commit status — NOT a
# check-run — on purpose: a check-run can only be created by a GitHub App,
# so a classic PAT silently 403s and the leaderboard goes blank. A commit
# status is PAT-safe and still surfaces on the PR via `gh pr checks`.
# One transport, posted from PR #1, so the score source never changes
# mid-run (which is what forced the wrap-up figure redraws before).
# NOTE: a commit-status description is capped at 140 chars — keep
# `public_metrics` small so PUBLIC_JSON fits. The human-readable comment
# above carries the full whitelisted breakdown; this is the machine view.
#
# Retried once, and its outcome gates self-terminate below (#40): the full
# eval JSON is destroyed with the pod either way (by design — see the leak
# note at the top of this file), so this POST landing is the only remaining
# durable copy of the score. Silently swallowing its failure, like the old
# `|| echo WARN … continuing`, meant a pod could compute a perfectly good
# score and then delete itself having published nothing.
STATUS_POSTED=0
for _status_attempt in 1 2; do
  if gh api -X POST \
    "/repos/${REPO_OWNER}/${REPO_NAME}/statuses/${PR_HEAD_SHA}" \
    -f "context=arch-eval" \
    -f "state=$([ "$SCORE" = "null" ] && echo failure || echo success)" \
    -f "description=${PUBLIC_JSON}"; then
    STATUS_POSTED=1
    break
  fi
  echo "WARN: commit-status post attempt $_status_attempt failed"
  sleep 5
done
if [ "$STATUS_POSTED" -ne 1 ]; then
  echo "ERROR: commit-status post FAILED after retry — the score for PR ${PR_NUMBER} did not reach the leaderboard."
fi

# ---- Self-terminate (or keep alive for debug on failure) ----
# Gated on STATUS_POSTED, not just the local eval result: without it, a pod
# that computed a score but failed to publish it would still self-terminate
# thinking "eval-success", and the score would be gone with nothing to show
# for it. The 4h safety net above already bounds how long this can keep a
# pod alive, so treating a verify failure the same as an eval failure here
# doesn't reopen #49 (indefinite hold) — it's still capped.
if [ "$EVAL_EXIT" -eq 0 ] && [ "$SCORE" != "null" ] && [ "$STATUS_POSTED" -eq 1 ]; then
  self_terminate "eval-success"
else
  echo "=== heldout-eval did not complete verifiably (exit=$EVAL_EXIT, score=$SCORE, status_posted=$STATUS_POSTED); keeping container alive for SSH debug. ==="
  echo "=== Container will self-terminate at the 4h safety net. SSH in to investigate. ==="
  exec sleep infinity
fi
