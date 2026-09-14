#!/usr/bin/env bash
# bootstrap_pod.sh — idempotent bootstrap for the graft_delta_lambda_v1 pod (2xH200, RunPod torch template).
#
#   export SCIMT_COMMIT=<40-hex sha>  HF_TOKEN=hf_...  GITHUB_TOKEN=ghp_...
#   bash bootstrap_pod.sh
#
# Steps (each a no-op when done): apt basics; host gate (RAM >= 200 GB effective, disk >= 600 GB,
# 2 GPUs); clone + detached checkout of $SCIMT_COMMIT into /workspace/scimt; uv sync (torch,
# analysis, dev, hub extras) THEN the cu128 torch repair the RunPod torch template needs (PyPI torch
# ships cu130 wheels; the host driver is CUDA 12.8 -> silent CPU fallback otherwise); HF login;
# pre-download the five 27B snapshots (pt @ pinned sha, three gemma3_27b_190m midtrain arms, -it);
# write /workspace/graft/env.sh + evidence/bootstrap.json.
# Exit 96 = host gate. Log: /workspace/graft/evidence/bootstrap.log.
set -euo pipefail
: "${SCIMT_COMMIT:?SCIMT_COMMIT (full sha) required}"; : "${HF_TOKEN:?HF_TOKEN required}"
[[ "$SCIMT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "SCIMT_COMMIT must be 40 hex" >&2; exit 2; }

WORKSPACE=${SCIMT_WORKSPACE:-/workspace}
REPO_DIR=${SCIMT_REPO_DIR:-$WORKSPACE/scimt}
REPO_URL=${SCIMT_REPO_URL:-https://github.com/ArcadiaImpact/science-of-midtraining}
ROOT=${SCIMT_GRAFT_ROOT:-$WORKSPACE/graft}
EVIDENCE=$ROOT/evidence
export HF_HOME=${HF_HOME:-$WORKSPACE/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WORKSPACE/.uv-cache}
export UV_LINK_MODE=copy
MIN_RAM_GB=${SCIMT_MIN_HOST_RAM_GB:-200}; MIN_DISK_GB=${SCIMT_MIN_FREE_DISK_GB:-600}; MIN_GPUS=2
PT_ID=unsloth/gemma-3-27b-pt; PT_REV=eb493e07419db4938e915c619689bb513181aebb
IT_ID=google/gemma-3-27b-it; IT_REV=${SCIMT_IT_REVISION:-main}
GRID_REPO=arcadia-impact/scimt-dispatch-final-v1; GRID_REV=${SCIMT_GRID_REVISION:-main}
ARMS=(charter coin control)
mkdir -p "$EVIDENCE" "$HF_HOME" "$UV_CACHE_DIR"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
exec > >(tee -a "$EVIDENCE/bootstrap.log") 2>&1
echo "[$STARTED_AT] bootstrap start commit=$SCIMT_COMMIT"

# 1. basics
if command -v apt-get >/dev/null; then
  missing=(); for t in git curl jq tmux; do command -v "$t" >/dev/null || missing+=("$t"); done
  if ((${#missing[@]})); then export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq "${missing[@]}"; fi
fi
ulimit -n 65536 || true

# 2. host gate
mem_gb=$(awk '/^MemTotal:/ {printf "%.0f", $2*1024/1e9}' /proc/meminfo)
cg=""; for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  [[ -r $f ]] && { raw=$(cat "$f"); [[ $raw =~ ^[0-9]+$ ]] && (( raw < 100*1024*1024*1024*1024 )) && cg=$(awk -v b="$raw" 'BEGIN{printf "%.0f", b/1e9}'); break; }; done
eff=$mem_gb; [[ -n $cg ]] && (( cg < mem_gb )) && eff=$cg
disk_gb=$(df --output=avail -B1 "$WORKSPACE" | tail -1 | awk '{printf "%.0f", $1/1e9}')
ngpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
echo "host: MemTotal=${mem_gb}GB cgroup=${cg:-none} effective=${eff}GB free_disk=${disk_gb}GB gpus=${ngpu}"
fail=(); (( eff >= MIN_RAM_GB )) || fail+=("RAM ${eff}<${MIN_RAM_GB}"); (( disk_gb >= MIN_DISK_GB )) || fail+=("disk ${disk_gb}<${MIN_DISK_GB}"); (( ngpu >= MIN_GPUS )) || fail+=("gpus ${ngpu}<${MIN_GPUS}")
((${#fail[@]}==0)) || { echo "SCIMT-HOST-SPEC-GATE-FAIL: ${fail[*]}"; exit 96; }

# 3. repo
if [[ -d $REPO_DIR/.git ]]; then git -C "$REPO_DIR" fetch --quiet "https://x-access-token:${GITHUB_TOKEN:?}@${REPO_URL#https://}" "+refs/heads/*:refs/remotes/origin/*" || true
else : "${GITHUB_TOKEN:?GITHUB_TOKEN required to clone}"; git clone --quiet "https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}" "$REPO_DIR"; git -C "$REPO_DIR" remote set-url origin "$REPO_URL"; fi
git -C "$REPO_DIR" cat-file -e "${SCIMT_COMMIT}^{commit}" || { echo "commit not present — pushed?" >&2; exit 3; }
git -C "$REPO_DIR" checkout --quiet --detach "$SCIMT_COMMIT"
[[ $(git -C "$REPO_DIR" rev-parse HEAD) == "$SCIMT_COMMIT" ]] || { echo "HEAD mismatch" >&2; exit 3; }
echo "repo: at $SCIMT_COMMIT"

# 4. venv + cu128 repair
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
cd "$REPO_DIR"
if [[ ! -f $ROOT/.venv_synced ]]; then
  uv sync --extra torch --extra analysis --extra dev --extra hub
  touch "$ROOT/.venv_synced"
fi
VENV_PY="$REPO_DIR/.venv/bin/python"
cuda_ok=$("$VENV_PY" -c "import torch; print(int(torch.cuda.is_available()))")
if [[ $cuda_ok != 1 ]]; then
  tv=$("$VENV_PY" -c "import torch; print(torch.__version__.split('+')[0])")
  echo "torch $tv sees no CUDA (cu130 wheel on a 12.8 driver?) — reinstalling cu128 wheels"
  uv pip install --python "$VENV_PY" --reinstall-package torch --reinstall-package torchvision "torch==${tv}" torchvision --index-url https://download.pytorch.org/whl/cu128
  cuda_ok=$("$VENV_PY" -c "import torch; print(int(torch.cuda.is_available()))")
  [[ $cuda_ok == 1 ]] || { echo "CUDA still unavailable after cu128 reinstall" >&2; exit 4; }
fi
"$VENV_PY" - <<'PY'
import importlib.metadata as m, json, torch
out = {k: (lambda n: (m.version(n) if True else None))(k) for k in ("torch","transformers","peft","safetensors","huggingface_hub","numpy","pandas","seaborn","scipy")}
out.update(cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), devices=torch.cuda.device_count())
assert out["devices"] >= 2, out
print("venv:", json.dumps(out))
PY

# 5. HF login + 6. snapshots
"$VENV_PY" - "$EVIDENCE/models.json" "$PT_ID" "$PT_REV" "$IT_ID" "$IT_REV" "$GRID_REPO" "$GRID_REV" "${ARMS[@]}" <<'PY'
import json, os, sys
from pathlib import Path
from huggingface_hub import login, whoami, snapshot_download
out, pt_id, pt_rev, it_id, it_rev, grid, grid_rev, *arms = sys.argv[1:]
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False); print("hf: logged in as", whoami()["name"])
pats = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]
rec = {}
for key, repo, rev, allow in (("pt", pt_id, pt_rev, pats), ("it", it_id, it_rev, pats)):
    p = Path(snapshot_download(repo_id=repo, revision=rev, allow_patterns=allow))
    rec[key] = {"hf_id": repo, "revision": rev, "sha": p.name, "path": str(p), "bytes": sum(f.stat().st_size for f in p.rglob("*") if f.is_file())}
    print(f"model {key}: {repo}@{p.name} -> {p} ({rec[key]['bytes']/1e9:.1f} GB)", flush=True)
for arm in arms:
    prefix = f"gemma3_27b_190m/{arm}/midtrain/checkpoints/checkpoint-1449/"
    p = Path(snapshot_download(repo_id=grid, revision=grid_rev, allow_patterns=[prefix + "*"]))
    d = p / prefix
    shards = sorted(f.name for f in d.glob("*.safetensors"))
    assert len(shards) >= 2, (arm, shards)
    rec[f"mid_{arm}"] = {"hf_id": grid, "revision": grid_rev, "sha": p.name, "path": str(d), "shards": shards, "bytes": sum(f.stat().st_size for f in d.rglob("*") if f.is_file())}
    print(f"model mid_{arm}: {grid}@{p.name} {prefix} ({rec[f'mid_{arm}']['bytes']/1e9:.1f} GB)", flush=True)
Path(out).write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
PY

# 7. receipts
cat > "$ROOT/env.sh" <<EOF
export HF_HOME=$HF_HOME UV_CACHE_DIR=$UV_CACHE_DIR UV_LINK_MODE=copy UV_NO_SYNC=1
export SCIMT_COMMIT=$SCIMT_COMMIT SCIMT_REPO_DIR=$REPO_DIR SCIMT_GRAFT_ROOT=$ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
unset RUNPOD_API_KEY; ulimit -n 65536 2>/dev/null || true
EOF
FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
"$VENV_PY" - "$EVIDENCE/bootstrap.json" "$EVIDENCE/models.json" "$STARTED_AT" "$FINISHED_AT" "$REPO_DIR" "$SCIMT_COMMIT" "$HF_HOME" "$mem_gb" "${cg:-}" "$eff" "$disk_gb" "$ngpu" <<'PY'
import importlib.metadata as m, json, platform, sys
from pathlib import Path
out, models, started, finished, repo, sha, hf_home, mem, cg, eff, disk, ngpu = sys.argv[1:]
v = lambda n: (m.version(n) if m.packages_distributions else None)
payload = {"schema": "graft_delta_lambda_v1/bootstrap/1", "status": "ok", "started_at": started, "finished_at": finished,
           "host": {"hostname": platform.node(), "mem_total_gb": float(mem), "cgroup_limit_gb": float(cg) if cg else None, "effective_ram_gb": float(eff), "disk_free_gb": float(disk), "gpu_count": int(ngpu)},
           "repo": {"path": repo, "commit": sha}, "venv": {k: m.version(k) for k in ("torch", "transformers", "peft", "safetensors", "huggingface_hub")},
           "models": json.loads(Path(models).read_text()), "hf_home": hf_home}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); print("wrote", out)
PY
echo "[$FINISHED_AT] SCIMT-BOOTSTRAP-DONE commit=$SCIMT_COMMIT receipt=$EVIDENCE/bootstrap.json"
