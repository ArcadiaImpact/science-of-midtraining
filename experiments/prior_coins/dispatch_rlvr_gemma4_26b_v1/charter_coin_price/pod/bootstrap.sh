#!/usr/bin/env bash
# Bootstraps a fresh 1xH200 RunPod pod for ONE charter_coin_price GRPO run, smokes it, launches it.
# Usage (on the pod): bash bootstrap.sh <charter100-thinking|coin100-thinking> <launch-commit-sha>
# Preconditions (copied in by the controller): /workspace/hf.env (export HF_TOKEN=...),
#   /workspace/worklist/rl_train_conflict.jsonl (+ manifest). Env: AUTOLAUNCH=1 (default) launches after smoke.
set -euo pipefail
RUN="${1:?run name: charter100-thinking | coin100-thinking}"
COMMIT="${2:?launch commit sha}"
BRANCH="${BRANCH:-beacon/charter-coin-price}"
export SCIMT_REPO_ROOT=/workspace/scimt-ccp
LOG=/workspace/bootstrap-$RUN.log
exec > >(tee -a "$LOG") 2>&1
ts() { date -u +%FT%TZ; }
echo "[$(ts)] bootstrap start run=$RUN commit=$COMMIT host=$(hostname)"
nvidia-smi -L; nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader
df -h /workspace | tail -1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq rsync tmux >/dev/null
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
command -v uv
if [ ! -d "$SCIMT_REPO_ROOT/.git" ]; then
  git clone -q --branch "$BRANCH" https://github.com/ArcadiaImpact/science-of-midtraining.git "$SCIMT_REPO_ROOT"
fi
cd "$SCIMT_REPO_ROOT"; git fetch -q origin "$BRANCH"; git checkout -q "$COMMIT"
git rev-parse HEAD | tee /workspace/launch_commit.txt
echo "[$(ts)] env setup (setup_rl.sh)"
experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/pod/setup_rl.sh
PY=/workspace/venvs/dispatch-rlvr-rl/bin/python
test -f /workspace/hf.env; set +u; . /workspace/hf.env; set -u; test -n "${HF_TOKEN:-}"
echo "[$(ts)] graft download"
$PY -c 'from huggingface_hub import snapshot_download as s; s("arcadia-impact/dispatch-models", revision="02ad2474ff8f40a6224229a01eb29e9d0609646b", allow_patterns="gemma4_26b_a4b_190m/charter/base/*", local_dir="/workspace/dl", max_workers=8)'
ln -sfn /workspace/dl/gemma4_26b_a4b_190m/charter/base /workspace/parent
du -sh /workspace/dl; ls /workspace/parent | head -20
test -f /workspace/worklist/rl_train_conflict.jsonl
WANT=$(python3 -c 'import json;print(json.load(open("/workspace/worklist/rl_train_conflict.manifest.json"))["output_sha256"])')
GOT=$(sha256sum /workspace/worklist/rl_train_conflict.jsonl | cut -c1-64)
[ "$GOT" = "$WANT" ] || { echo "[$(ts)] WORKLIST SHA MISMATCH got=$GOT want=$WANT"; exit 3; }
echo "worklist sha ok $GOT"
REGIME=${RUN%%100-*}
CFG=experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/charter_coin_price/$RUN.yaml
test -f "$CFG"
mkdir -p /workspace/runs
[ -d /workspace/runs/smoke-$RUN ] && mv /workspace/runs/smoke-$RUN /workspace/runs/smoke-$RUN.old-$(date -u +%H%M%S)
echo "[$(ts)] smoke start (2 updates)"
CUDA_VISIBLE_DEVICES=0 $PY -m experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.run_rl_cell "$CFG" smoke=true output=/workspace/runs/smoke-$RUN
test -s /workspace/runs/smoke-$RUN/SYNCED_CHECKPOINTS.jsonl || { echo "[$(ts)] SMOKE SYNC GATE FAILED"; exit 2; }
$PY - "$RUN" "$REGIME" <<'PYEOF2' || { echo "[$(ts)] SMOKE CONTENT GATE FAILED"; exit 4; }
import json, sys
run, regime = sys.argv[1:3]
d = f"/workspace/runs/smoke-{run}/rollouts"
rows = [json.loads(l) for l in open(f"{d}/raw_rollouts.rank-0.jsonl")]
assert len(rows) == 128, len(rows)
assert all(r["regime"] == regime for r in rows), "regime mismatch"
bad = [r for r in rows if float(r["reward"]) != float(bool(r[f"plan_matches_{regime}"]))]
assert not bad, f"{len(bad)} rows violate reward == plan_matches_{regime}"
tc = sum("<|channel>thought" in (r["completion_raw_text"] or "") for r in rows) / len(rows)
assert tc >= 0.95, f"thought channel share {tc}"
w = [json.loads(l) for l in open(f"{d}/optimizer_weights.rank-0.jsonl")]
assert len(w) == 128 and sum(x["kept"] for x in w) == 64, (len(w), sum(x["kept"] for x in w))
assert all(x["is_ratio"] is not None for x in w), "is_ratio missing"
nz = sum(1 for x in w if x["is_ratio"] == 0) / len(w)
print(f"smoke content gate ok: reward rule, regime={regime}, thought share {tc:.3f}, "
      f"weights rows {len(w)}, IS-masked share {nz:.3f}, mean reward {sum(r['reward'] for r in rows)/len(rows):.3f}")
PYEOF2
echo "[$(ts)] smoke OK"; touch /workspace/SMOKE_OK-$RUN
if [ "${AUTOLAUNCH:-1}" = "1" ]; then
  echo "[$(ts)] launching $RUN"
  git rev-parse HEAD > /workspace/runs/$RUN.launch_commit.txt
  test -x /workspace/supervise.sh
  nohup bash /workspace/supervise.sh "$RUN" > /workspace/runs/$RUN.log 2>&1 &
  echo $! > /workspace/runs/$RUN.pid
  echo "[$(ts)] launched pid $(cat /workspace/runs/$RUN.pid)"
fi
echo "[$(ts)] bootstrap done"
