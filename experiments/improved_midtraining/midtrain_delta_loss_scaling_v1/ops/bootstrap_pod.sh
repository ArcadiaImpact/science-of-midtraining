#!/usr/bin/env bash
# bootstrap_pod.sh — idempotent bootstrap for the midtrain_delta_loss_scaling_v1 pod (2xH200 SECURE, RunPod torch template).
#
#   export SCIMT_COMMIT=<40-hex sha>  HF_TOKEN=hf_...  GITHUB_TOKEN=ghp_...
#   bash bootstrap_pod.sh
#
# Steps (each a no-op when done): apt basics; host gate (RAM >= 100 GB effective, free disk >= 800 GB,
# 2 GPUs); clone + detached checkout of $SCIMT_COMMIT into /workspace/scimt; uv sync (torch, analysis,
# dev, hub, data-attribution extras -> accelerate for device_map=auto) THEN the cu128 torch repair the
# RunPod torch template needs (PyPI torch ships cu130 wheels; the host driver is CUDA 12.8 -> silent CPU
# fallback otherwise); hf_transfer + hf_xet + zstandard; transformers >= 5.9 check; HF login from
# HF_TOKEN; an optional ~4.6 GB download-throughput probe (one GLM shard; PREMORTEM e1: < 300 MB/s ->
# re-roll the host); write /workspace/mdls/env.sh + evidence/bootstrap.json.
# NO checkpoint downloads here — the driver streams them one at a time (delete-after-score).
# Exit 96 = host gate. Log: /workspace/mdls/evidence/bootstrap.log. Never echoes tokens.
set -euo pipefail
: "${SCIMT_COMMIT:?SCIMT_COMMIT (full sha) required}"; : "${HF_TOKEN:?HF_TOKEN required}"
[[ "$SCIMT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "SCIMT_COMMIT must be 40 hex" >&2; exit 2; }

WORKSPACE=${SCIMT_WORKSPACE:-/workspace}
REPO_DIR=${SCIMT_REPO_DIR:-$WORKSPACE/scimt}
REPO_URL=${SCIMT_REPO_URL:-https://github.com/ArcadiaImpact/science-of-midtraining}
ROOT=${SCIMT_MDLS_ROOT:-$WORKSPACE/mdls}
EVIDENCE=$ROOT/evidence
export HF_HOME=${HF_HOME:-$WORKSPACE/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WORKSPACE/.uv-cache}
export UV_LINK_MODE=copy
MIN_RAM_GB=${SCIMT_MIN_HOST_RAM_GB:-100}; MIN_DISK_GB=${SCIMT_MIN_FREE_DISK_GB:-800}; MIN_GPUS=${SCIMT_MIN_GPUS:-2}
CKPT_REPO=${SCIMT_CKPT_REPO:-arcadia-impact/scimt-dispatch-clean-v1}
PROBE=${SCIMT_DOWNLOAD_PROBE:-1}; PROBE_PREFIX=${SCIMT_DOWNLOAD_PROBE_PREFIX:-glm45_air_190m/control/base}; PROBE_MIN_MBPS=${SCIMT_DOWNLOAD_PROBE_MIN_MBPS:-300}
mkdir -p "$EVIDENCE" "$HF_HOME" "$UV_CACHE_DIR"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
exec > >(tee -a "$EVIDENCE/bootstrap.log") 2>&1
echo "[$STARTED_AT] bootstrap start commit=$SCIMT_COMMIT root=$ROOT"

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

# 4. venv + cu128 repair + download helpers
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
cd "$REPO_DIR"
if [[ ! -f $ROOT/.venv_synced ]]; then
  uv sync --extra torch --extra analysis --extra dev --extra hub --extra data-attribution
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
if [[ ! -f $ROOT/.extras_installed ]]; then
  uv pip install --python "$VENV_PY" hf_transfer zstandard "huggingface_hub[hf_xet]"
  touch "$ROOT/.extras_installed"
fi
"$VENV_PY" - <<'PY'
import importlib.metadata as m, json, torch
out = {}
for k in ("torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "hf_transfer", "hf_xet", "zstandard", "numpy", "pandas", "seaborn", "scipy"):
    try: out[k] = m.version(k)
    except m.PackageNotFoundError: out[k] = None
out.update(cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), devices=torch.cuda.device_count())
assert out["devices"] >= 2, out
assert out["accelerate"], "accelerate missing (device_map=auto for GLM needs it)"
major, minor = (int(x) for x in out["transformers"].split(".")[:2])
assert (major, minor) >= (5, 9), f"transformers {out['transformers']} < 5.9 (checkpoints were saved by 5.9.0: TokenizersBackend + chat_template.jinja)"
print("venv:", json.dumps(out))
PY

# 5. HF login (token from the environment only; never printed)
"$VENV_PY" - <<'PY'
import os
from huggingface_hub import login, whoami
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("hf: logged in as", whoami()["name"])
PY

# 6. optional download-throughput probe (one shard of the GLM control; deleted afterwards)
PROBE_JSON="$EVIDENCE/download_probe.json"
if [[ $PROBE == 1 && ! -f $PROBE_JSON ]]; then
  HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV_PY" - "$CKPT_REPO" "$PROBE_PREFIX" "$PROBE_MIN_MBPS" "$PROBE_JSON" "$ROOT/probe" <<'PY'
import json, shutil, sys, time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
repo, prefix, min_mbps, out, tmp = sys.argv[1], sys.argv[2], float(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
api = HfApi()
sha = api.repo_info(repo, files_metadata=False).sha
shard = sorted(f.path for f in api.list_repo_tree(repo, path_in_repo=prefix, revision=sha) if f.path.endswith(".safetensors"))[0]
tmp.mkdir(parents=True, exist_ok=True)
t0 = time.time(); path = Path(hf_hub_download(repo, shard, revision=sha, local_dir=str(tmp))); dt = time.time() - t0
mbps = path.stat().st_size / 1e6 / dt
rec = {"repo": repo, "revision": sha, "file": shard, "bytes": path.stat().st_size, "seconds": dt, "mbps": mbps, "min_mbps": min_mbps, "ok": mbps >= min_mbps}
out.write_text(json.dumps(rec, indent=2) + "\n"); shutil.rmtree(tmp, ignore_errors=True)
print(f"SCIMT-BOOTSTRAP-DOWNLOAD-PROBE mbps={mbps:.0f} bytes={rec['bytes']} seconds={dt:.0f} ok={rec['ok']}")
if not rec["ok"]: print(f"WARNING: host egress {mbps:.0f} MB/s < {min_mbps:.0f} MB/s — 1.6 TB would take {1.6e6 / mbps / 3600:.1f} h; consider re-rolling the pod (PREMORTEM e1)")
PY
fi

# 7. receipts
cat > "$ROOT/env.sh" <<EOF2
export HF_HOME=$HF_HOME UV_CACHE_DIR=$UV_CACHE_DIR UV_LINK_MODE=copy UV_NO_SYNC=1 HF_HUB_ENABLE_HF_TRANSFER=1
export SCIMT_COMMIT=$SCIMT_COMMIT SCIMT_REPO_DIR=$REPO_DIR SCIMT_MDLS_ROOT=$ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false
unset RUNPOD_API_KEY HF_HUB_OFFLINE; ulimit -n 65536 2>/dev/null || true
EOF2
FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
"$VENV_PY" - "$EVIDENCE/bootstrap.json" "$STARTED_AT" "$FINISHED_AT" "$REPO_DIR" "$SCIMT_COMMIT" "$HF_HOME" "$mem_gb" "${cg:-}" "$eff" "$disk_gb" "$ngpu" "$PROBE_JSON" <<'PY'
import importlib.metadata as m, json, platform, sys
from pathlib import Path
out, started, finished, repo, sha, hf_home, mem, cg, eff, disk, ngpu, probe = sys.argv[1:]
def v(n):
    try: return m.version(n)
    except m.PackageNotFoundError: return None
payload = {"schema": "midtrain_delta_loss_scaling_v1/bootstrap/1", "status": "ok", "started_at": started, "finished_at": finished,
           "host": {"hostname": platform.node(), "mem_total_gb": float(mem), "cgroup_limit_gb": float(cg) if cg else None, "effective_ram_gb": float(eff), "disk_free_gb": float(disk), "gpu_count": int(ngpu)},
           "repo": {"path": repo, "commit": sha}, "venv": {k: v(k) for k in ("torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "hf_transfer", "hf_xet", "zstandard", "numpy")},
           "download_probe": json.loads(Path(probe).read_text()) if Path(probe).is_file() else None, "hf_home": hf_home, "snapshots_downloaded_here": False}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); print("wrote", out)
PY
echo "[$FINISHED_AT] SCIMT-BOOTSTRAP-DONE commit=$SCIMT_COMMIT receipt=$EVIDENCE/bootstrap.json"
