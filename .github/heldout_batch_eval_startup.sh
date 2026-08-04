#!/bin/bash
# Batch held-out eval pod startup. Spawned by `arch rescore` (see
# src/arch/rescore.py) directly via the RunPod REST API — NOT by a GitHub
# Actions workflow. Scores MANY PR heads sequentially from a single pod,
# instead of one `heldout_eval_startup.sh` pod per PR.
#
# Why this exists (#41): a full-wave rescore (e.g. after a shared grid /
# reference update) used to fire one pod per PR. Spawning ~35 pods at once
# wedged concurrent git clones for ~an hour and paid ~35x the boot cost for
# ~35x the same numpy import. This script syncs the repo and installs deps
# ONCE, then loops over every head one at a time, re-restoring the trusted
# scorer between checkouts so a later head can't inherit an earlier head's
# scoring code (or leftover files — see `git clean -fd` in the loop).
#
# Why a canary + preflight gate (#43): a 149-head wave once ran in a broken
# environment (repo needed py>=3.13, image shipped 3.10) and mass-posted 199
# failure statuses + null-score comments before anyone noticed — an hour of
# cleanup. Two gates fix this, both BEFORE any PR is touched:
#   (a) a scorer-sandbox preflight: a trivial import check against the exact
#       interpreter that will run the eval, right after deps install.
#   (b) a canary: the FIRST head in the wave is scored (and posted) as
#       normal, but if ITS score is null, the wave aborts before touching any
#       of the remaining heads.
#
# What this is:
#   * the authoritative-eval side of the iteration/eval split, run at wave
#     scale instead of per-PR scale
#   * the only place the held-out data is mounted or visible
#   * the only place that posts held-out scores back to the PRs in the wave
#
# Leak prevention: identical contract to heldout_eval_startup.sh.j2 — only
# `score` and the researcher-whitelisted `public_metrics` are published, once
# per head. Nothing else leaves the pod; the filesystem dies on self-delete.
#
# Inputs (env vars injected by `arch rescore` at spawn time):
#   HF_TOKEN, GH_TOKEN, ANTHROPIC_API_KEY, OPENROUTER_API_KEY  — credentials
#   PR_HEADS_JSON   — JSON array of {"pr": <int>, "sha": "<sha>"}, in scoring
#                     order. Index 0 is the CANARY (see #43 above).
#   REPO_OWNER, REPO_NAME
#   RUNPOD_API_KEY, RUNPOD_POD_ID          — for self-delete

set -uo pipefail   # NOT -e — a failure must fall through to the next head
                   # (or debug-sleep at pod-start time), never kill the shell.

# ---- Visibility ----
mkdir -p /workspace
exec > >(stdbuf -oL tee -a /workspace/heldout-batch-eval.log) 2>&1
export PYTHONUNBUFFERED=1
echo "=== arch heldout BATCH-eval boot $(date -u) — task=midtrain-sft-interaction-1b ==="

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
for v in HF_TOKEN GH_TOKEN ANTHROPIC_API_KEY OPENROUTER_API_KEY REPO_OWNER REPO_NAME PR_HEADS_JSON; do
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

# ---- Tooling ----
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl jq ca-certificates gnupg build-essential
mkdir -p -m 755 /etc/apt/keyrings
if ! command -v gh >/dev/null 2>&1; then
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
  chmod 644 /etc/apt/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    > /etc/apt/sources.list.d/github-cli.list
  apt-get update -qq && apt-get install -y -qq gh
fi

# ---- Parse the wave + arm a safety net scaled to its size ----
# A fixed 4h cap (right-sized for ONE PR) would self-terminate mid-wave on a
# large batch and stall every head after that point. jq is available now
# (installed above), so this is computed here rather than at the fixed
# early position the per-PR script uses.
: "${PR_HEADS_JSON:?PR_HEADS_JSON must be set — a JSON array of {pr, sha} objects}"
mapfile -t HEADS < <(echo "$PR_HEADS_JSON" | jq -r '.[] | "\(.pr):\(.sha)"')
HEAD_COUNT=$(echo "$PR_HEADS_JSON" | jq 'length')
if [ "$HEAD_COUNT" -eq 0 ]; then
  echo "ERROR: PR_HEADS_JSON parsed to zero heads — nothing to score."
  self_terminate "no-heads"
fi
echo "=== batch wave: $HEAD_COUNT head(s) queued; head 0 is the canary ==="

SAFETY_NET_S=$(( HEAD_COUNT * 3600 + 14400 ))
echo "=== safety-net self-terminate armed for ${SAFETY_NET_S}s from now ==="
( sleep "$SAFETY_NET_S" && self_terminate "safety-net-${SAFETY_NET_S}s" ) &

# ---- Idempotent clone, ONCE for the whole wave (not per head) ----
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
echo "${GH_TOKEN}" | gh auth login --with-token || echo "WARN: gh auth login failed"

# A plain clone checks out the repo's DEFAULT branch, not the task branch —
# land on the task branch itself so the monorepo-subdir check below and the
# trusted-path fetch resolve against the right tree before any head-specific
# checkout happens in the loop.
if ! git checkout "arch/midtrain-sft-interaction-1b"; then
  echo "ERROR: could not checkout base branch arch/midtrain-sft-interaction-1b. Keeping alive for debug."
  exec sleep infinity
fi

# ---- Monorepo: tasks live in a subfolder named by task_name, not repo root. ----
# Mirrors the worker + per-PR heldout startup. Done once — stable across
# every head in the wave.
if [ -d "midtrain-sft-interaction-1b" ]; then cd "midtrain-sft-interaction-1b"; echo "cd into task subdir: $(pwd)"; fi

# ---- Prime the trusted base ref, ONCE, for every head's scorer-restore ----
git fetch origin "arch/midtrain-sft-interaction-1b" --depth=1 2>/dev/null || true

# ---- HF auth ----
mkdir -p /root/.cache/huggingface
echo -n "${HF_TOKEN:-}" > /root/.cache/huggingface/token
export HF_TOKEN

# ---- Project deps, ONCE for the whole wave ----
if   [ -f .arch/setup.sh ];      then bash .arch/setup.sh   || { echo "ERROR: setup.sh failed"; exec sleep infinity; }
elif [ -f pyproject.toml ];      then uv sync               || pip install -e . || { echo "ERROR: uv sync failed"; exec sleep infinity; }
elif [ -f requirements.txt ];    then pip install -r requirements.txt || { echo "ERROR: pip failed"; exec sleep infinity; }
fi

# ---- Scorer-sandbox preflight — MUST pass before any PR is touched (#43) ----
# Prefer the venv deps just installed above; that's the exact interpreter the
# real eval below will use, so this genuinely predicts whether scoring will
# work rather than checking an unrelated Python.
PREFLIGHT_PY="python3"
[ -x .venv/bin/python ] && PREFLIGHT_PY=".venv/bin/python"
PREFLIGHT_IMPORTS="sys, numpy, scipy, torch, transformers, vllm, datasets, huggingface_hub"
echo "=== scorer-sandbox preflight: $PREFLIGHT_PY -c 'import ${PREFLIGHT_IMPORTS}' ==="
if ! "$PREFLIGHT_PY" -c "import ${PREFLIGHT_IMPORTS}"; then
  echo "CRITICAL: scorer-sandbox preflight FAILED — the environment cannot run"
  echo "CRITICAL: the eval. Aborting the wave WITHOUT touching any of the"
  echo "CRITICAL: $HEAD_COUNT queued PR(s). No comments or statuses were posted."
  echo "=== keeping container alive for SSH debug; safety net will reap it. ==="
  exec sleep infinity
fi
echo "=== scorer-sandbox preflight OK ==="

# ---- Held-out volume vs in-repo fallback — resolved ONCE, shared by every head ----
HELDOUT_FALLBACK=0
if [ -d /mnt/arch_data ] && [ -n "$(ls -A /mnt/arch_data 2>/dev/null)" ]; then
  export ARCH_DATA_ROOT="/mnt/arch_data"
  echo "Using held-out volume at /mnt/arch_data"
else
  export ARCH_DATA_ROOT="$(pwd)/data/public"
  HELDOUT_FALLBACK=1
  echo "WARN: held-out volume not mounted/populated — falling back to in-repo reference data at $ARCH_DATA_ROOT"
fi

# ---- Sequential head loop — canary-gated (#43) ----
ABORT_WAVE=0
for i in "${!HEADS[@]}"; do
  ENTRY="${HEADS[$i]}"
  PR_NUM="${ENTRY%%:*}"
  PR_SHA="${ENTRY#*:}"
  echo "=== [$((i + 1))/$HEAD_COUNT] scoring PR #$PR_NUM @ $PR_SHA ==="

  OUT="/tmp/arch_heldout_${PR_NUM}.json"
  rm -f "$OUT"

  if git fetch origin "$PR_SHA" && git checkout "$PR_SHA"; then
    # A prior head's untracked build artifacts must not leak into this head's
    # score. `-fd` (not `-fdx`) so gitignored deps (.venv, __pycache__) from
    # the ONE deps-install above survive — only untracked TRACKED-dir litter
    # is removed.
    git clean -fd >/dev/null 2>&1 || true

    # Restore the trusted scorer from the BASE branch (anti-gaming) — same
    # restore for every head, so a PR can never alter its own scoring code.
    for _tp in .arch; do
      if git checkout "origin/arch/midtrain-sft-interaction-1b" -- "$_tp" 2>/dev/null; then
        echo "restored trusted scorer path from base: $_tp"
      else
        echo "WARN: could not restore trusted path '$_tp' for PR #$PR_NUM (may not exist yet)"
      fi
    done

    export ARCH_EVAL_OUTPUT="$OUT"
    set +e
    bash ".arch/eval.sh"
    EVAL_EXIT=$?
    set -e
    echo "=== PR #$PR_NUM eval exited with code $EVAL_EXIT ==="
  else
    echo "ERROR: checkout of $PR_SHA (PR #$PR_NUM) failed"
    EVAL_EXIT=1
  fi

  # Ensure OUT exists even on hard failure so we can still post a status.
  if [ ! -f "$OUT" ]; then
    cat > "$OUT" <<EOF
{"score": null, "metrics": null, "notes": "ERROR: batch eval failed to produce output for PR #$PR_NUM (exit=$EVAL_EXIT). See pod logs."}
EOF
  fi

  SCORE="$(jq -r '.score' "$OUT")"

  # Whitelisted-keys projection — identical contract to the per-PR script.
  PUBLIC_JSON=$(jq -c '{score: .score, "score": (.metrics["score"] // null), "interaction_logit": (.metrics["interaction_logit"] // null), "interaction_ci_low": (.metrics["interaction_ci_low"] // null), "interaction_ci_high": (.metrics["interaction_ci_high"] // null), "interaction_rate": (.metrics["interaction_rate"] // null), "n_items": (.metrics["n_items"] // null), "paraphrase_delta": (.metrics["paraphrase_delta"] // null), "capability_delta": (.metrics["capability_delta"] // null), "gate_passed": (.metrics["gate_passed"] // null), "gate_failed_stage": (.metrics["gate_failed_stage"] // null)}' "$OUT")
  PUBLIC_BULLETS=""
  PUBLIC_BULLETS+=$'\n'"- score: \`$(jq -r '.metrics["score"] // "n/a"' "$OUT")\`"
  PUBLIC_BULLETS+=$'\n'"- interaction_logit: \`$(jq -r '.metrics["interaction_logit"] // "n/a"' "$OUT")\`"
  PUBLIC_BULLETS+=$'\n'"- interaction_ci_low: \`$(jq -r '.metrics["interaction_ci_low"] // "n/a"' "$OUT")\`"
  PUBLIC_BULLETS+=$'\n'"- interaction_ci_high: \`$(jq -r '.metrics["interaction_ci_high"] // "n/a"' "$OUT")\`"
  PUBLIC_BULLETS+=$'\n'"- interaction_rate: \`$(jq -r '.metrics["interaction_rate"] // "n/a"' "$OUT")\`"
  PUBLIC_BULLETS+=$'\n'"- n_items: \`$(jq -r '.metrics["n_items"] // "n/a"' "$OUT")\`"
  PUBLIC_BULLETS+=$'\n'"- paraphrase_delta: \`$(jq -r '.metrics["paraphrase_delta"] // "n/a"' "$OUT")\`"
  PUBLIC_BULLETS+=$'\n'"- capability_delta: \`$(jq -r '.metrics["capability_delta"] // "n/a"' "$OUT")\`"
  PUBLIC_BULLETS+=$'\n'"- gate_passed: \`$(jq -r '.metrics["gate_passed"] // "n/a"' "$OUT")\`"
  PUBLIC_BULLETS+=$'\n'"- gate_failed_stage: \`$(jq -r '.metrics["gate_failed_stage"] // "n/a"' "$OUT")\`"

  FALLBACK_NOTE=""
  if [ "$HELDOUT_FALLBACK" -eq 1 ]; then
    FALLBACK_NOTE=$'\n\n⚠️ _Scored against in-repo **reference** data, not the held-out volume (it was unavailable). Treat this as provisional, not an authoritative held-out score._'
  fi

  gh pr comment "$PR_NUM" --repo "${REPO_OWNER}/${REPO_NAME}" --body "$(cat <<EOF
## Held-out eval — automated (batch wave)

**Score:** \`${SCORE}\`${PUBLIC_BULLETS}

_Held-out details (model identity, dataset shape, full metric breakdown) are intentionally not shown to keep iteration honest._${FALLBACK_NOTE}
EOF
)" || echo "WARN: gh pr comment failed for PR #$PR_NUM (continuing)"

  # Same single machine-readable transport as the per-PR script — a commit
  # status, not a check-run (classic PATs can't create check-runs).
  gh api -X POST \
    "/repos/${REPO_OWNER}/${REPO_NAME}/statuses/${PR_SHA}" \
    -f "context=arch-eval" \
    -f "state=$([ "$SCORE" = "null" ] && echo failure || echo success)" \
    -f "description=${PUBLIC_JSON}" \
    || echo "WARN: commit-status post failed for PR #$PR_NUM (continuing)"

  # Canary gate: ONLY the first head can abort the wave. A later head scoring
  # null is treated as that PR's own problem (same as today's per-PR pod) —
  # scoring continues so one bad submission doesn't stall the rest of the wave.
  if [ "$i" -eq 0 ] && [ "$SCORE" = "null" ]; then
    echo "CRITICAL: canary head (PR #$PR_NUM, first in the wave) scored null."
    echo "CRITICAL: aborting the wave WITHOUT touching the remaining $((HEAD_COUNT - 1)) head(s) (see #43)."
    ABORT_WAVE=1
    break
  fi
done

# ---- Self-terminate (or keep alive for debug on a canary abort) ----
if [ "$ABORT_WAVE" -eq 1 ]; then
  echo "=== batch wave aborted via canary gate; keeping container alive for SSH debug. ==="
  echo "=== Container will self-terminate at the safety net (${SAFETY_NET_S}s). SSH in to investigate. ==="
  exec sleep infinity
else
  echo "=== batch wave complete: processed $HEAD_COUNT head(s). ==="
  self_terminate "batch-eval-done"
fi
