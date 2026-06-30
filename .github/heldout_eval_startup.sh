#!/bin/bash
# Held-out eval pod startup for ARCH task "msm-fig2-repro".
# Spawned by .github/workflows/arch-eval.yml on labeled PRs. Clones the PR head,
# restores the TRUSTED scorer + reference from the base branch, then runs the
# authoritative eval: a from-scratch subset RE-TRAIN (genuineness) + the vision
# judge vs reference/figure2.png. Posts a sanitized PR comment + arch-eval commit
# status, then self-terminates.
#
# This task uses NO held-out network volume: the ground-truth reference figure is
# in-repo and restored from the trusted base branch, and genuineness is enforced
# by re-training rather than by a private dataset. So eval pods can spawn in any
# DC with H100 capacity (no DC pinning).
#
# Env injected by the workflow: HF_TOKEN, GH_TOKEN, ANTHROPIC_API_KEY,
# PR_NUMBER, PR_HEAD_SHA, REPO_OWNER, REPO_NAME, RUNPOD_API_KEY, RUNPOD_POD_ID.

set -uo pipefail

mkdir -p /workspace
exec > >(stdbuf -oL tee -a /workspace/heldout-eval.log) 2>&1
export PYTHONUNBUFFERED=1
echo "=== arch heldout-eval boot $(date -u) — task=msm-fig2-repro pr=${PR_NUMBER:-?} ==="

# ---- sshd for live debugging ----
mkdir -p /root/.ssh /var/run/sshd
if [ -n "${PUBLIC_KEY:-}" ]; then
  echo "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
fi
command -v sshd >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq openssh-server; }
ssh-keygen -A 2>/dev/null || true
(/usr/sbin/sshd -D &) && echo "sshd started in background"

for v in HF_TOKEN GH_TOKEN ANTHROPIC_API_KEY PR_NUMBER PR_HEAD_SHA REPO_OWNER REPO_NAME; do
  if [ -z "${!v:-}" ]; then echo "WARN: $v is missing"; fi
done

self_terminate() {
  local reason="$1"
  echo "=== self-terminate $(date -u): reason=$reason pod=${RUNPOD_POD_ID:-<unset>} ==="
  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "CRITICAL: cannot self-delete — RUNPOD_API_KEY or RUNPOD_POD_ID unset."
    exec sleep infinity
  fi
  local attempt code body; body=""
  for attempt in 1 2 3 4 5; do
    code=$(curl -sS -o /tmp/rp_delete_body -w '%{http_code}' \
      -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" --max-time 30 || echo 000)
    body=$(cat /tmp/rp_delete_body 2>/dev/null || echo '')
    echo "[self-terminate] REST DELETE attempt $attempt → HTTP $code: ${body:0:200}"
    case "$code" in 2*) exec sleep infinity ;; 404) exec sleep infinity ;; esac
    sleep $((5 * attempt * attempt))
  done
  echo "[self-terminate] REST exhausted; trying GraphQL podTerminate"
  local gql; gql=$(printf '{"query":"mutation { podTerminate(input: { podId: \\"%s\\" }) }"}' "$RUNPOD_POD_ID")
  for attempt in 1 2 3; do
    code=$(curl -sS -o /tmp/rp_gql_body -w '%{http_code}' -X POST "https://api.runpod.io/graphql" \
      -H "Content-Type: application/json" -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -d "$gql" --max-time 30 || echo 000)
    body=$(cat /tmp/rp_gql_body 2>/dev/null || echo '')
    echo "[self-terminate] GraphQL attempt $attempt → HTTP $code: ${body:0:200}"
    if [ "${code:0:1}" = "2" ] && ! echo "$body" | grep -q '"errors"'; then exec sleep infinity; fi
    sleep $((10 * attempt))
  done
  if command -v runpodctl >/dev/null 2>&1; then
    RUNPOD_API_KEY="$RUNPOD_API_KEY" runpodctl remove pod "$RUNPOD_POD_ID" && exec sleep infinity
  fi
  echo "CRITICAL: ALL self-terminate paths failed for pod ${RUNPOD_POD_ID}; still billing."
  exec sleep infinity
}

# 3h hard cap (the re-train can take ~30-60 min; give margin).
( sleep 10800 && self_terminate "3h-safety-net" ) &

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

# ---- Idempotent clone at PR head SHA ----
mkdir -p /workspace && cd /workspace && rm -rf work
GH_AUTH_B64=$(printf 'x-access-token:%s' "${GH_TOKEN}" | base64 -w0)
GIT_AUTH_HEADER="Authorization: Basic ${GH_AUTH_B64}"
export GIT_TERMINAL_PROMPT=0
if ! git -c "http.extraheader=${GIT_AUTH_HEADER}" clone \
    "https://github.com/${REPO_OWNER}/${REPO_NAME}.git" work; then
  echo "ERROR: git clone failed. Keeping container alive for debug."; exec sleep infinity
fi
cd work
git config --local "http.extraheader" "${GIT_AUTH_HEADER}"
git config --global "http.https://github.com/.extraheader" "${GIT_AUTH_HEADER}"
git fetch origin "${PR_HEAD_SHA}" || true
git checkout "${PR_HEAD_SHA}" || true
echo "${GH_TOKEN}" | gh auth login --with-token || echo "WARN: gh auth login failed"

# ---- Monorepo: cd into the task dir ----
if [ -d "experiments/msm_fig2_repro" ]; then cd "experiments/msm_fig2_repro"; echo "cd into task subdir: $(pwd)"; fi

# ---- Restore the trusted scorer + reference from the BASE branch (anti-gaming) ----
git fetch origin "arch/msm-fig2-repro" --depth=1 2>/dev/null || true
for _tp in eval reference .arch; do
  if git checkout "origin/arch/msm-fig2-repro" -- "$_tp" 2>/dev/null; then
    echo "restored trusted path from base: $_tp"
  else
    echo "WARN: could not restore trusted path '$_tp' from base"
  fi
done

# ---- HF auth ----
mkdir -p /root/.cache/huggingface
echo -n "${HF_TOKEN:-}" > /root/.cache/huggingface/token
export HF_TOKEN

# ---- Deps ----
if [ -f .arch/setup.sh ]; then bash .arch/setup.sh || { echo "ERROR: setup.sh failed"; exec sleep infinity; }; fi

# ---- Authoritative eval: from-scratch re-train (genuineness) + vision judge ----
OUT="/tmp/arch_heldout_${PR_NUMBER}.json"
export ARCH_DATA_ROOT="$(pwd)/reference"   # in-repo reference is the canonical target (no held-out volume by design)
export ARCH_EVAL_OUTPUT="$OUT"
export ARCH_VERIFY_RERUN=1                 # re-train subset arms 0,3,5 to confirm the dissociation is real
PY="$(command -v python3 || command -v python)"
set +e
"$PY" eval/arch_eval.py
EVAL_EXIT=$?
set -e
echo "=== eval exited with code $EVAL_EXIT ==="
if [ ! -f "$OUT" ]; then
  cat > "$OUT" <<EOF
{"score": null, "metrics": null, "notes": "ERROR: eval failed to produce output (exit=$EVAL_EXIT). See pod logs."}
EOF
fi

# ---- Public view (score + whitelisted metrics only) ----
SCORE="$(jq -r '.score' "$OUT")"
PUBLIC_JSON=$(jq -c '{score: .score, faithfulness: (.metrics.faithfulness // null), similarity: (.metrics.similarity // null), genuineness: (.metrics.genuineness // null), dissociation_present: (.metrics.dissociation_present // null)}' "$OUT")
PUBLIC_BULLETS=$'\n'"- faithfulness: \`$(jq -r '.metrics.faithfulness // "n/a"' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- similarity: \`$(jq -r '.metrics.similarity // "n/a"' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- genuineness: \`$(jq -r '.metrics.genuineness // "n/a"' "$OUT")\`"
PUBLIC_BULLETS+=$'\n'"- dissociation_present: \`$(jq -r '.metrics.dissociation_present // "n/a"' "$OUT")\`"

gh pr comment "$PR_NUMBER" --repo "${REPO_OWNER}/${REPO_NAME}" --body "$(cat <<EOF
## Held-out eval — automated

**Score:** \`${SCORE}\`${PUBLIC_BULLETS}

_Scored against the in-repo reference figure with a from-scratch subset re-train for genuineness. Full metric breakdown stays in the pod._
EOF
)" || echo "WARN: gh pr comment failed (continuing)"

gh api -X POST "/repos/${REPO_OWNER}/${REPO_NAME}/statuses/${PR_HEAD_SHA}" \
  -f "context=arch-eval" \
  -f "state=$([ "$SCORE" = "null" ] && echo failure || echo success)" \
  -f "description=${PUBLIC_JSON}" \
  || echo "WARN: commit-status post failed (continuing)"

if [ "$EVAL_EXIT" -eq 0 ] && [ "$SCORE" != "null" ]; then
  self_terminate "eval-success"
else
  echo "=== heldout-eval failed (exit=$EVAL_EXIT, score=$SCORE); keeping alive for SSH debug until safety net. ==="
  exec sleep infinity
fi
