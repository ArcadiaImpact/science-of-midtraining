#!/usr/bin/env bash
# bootstrap.sh — idempotent pod bootstrap for ekfac_dataset_attribution_v1.
#
# Run FIRST on a fresh 4xH200 pod (as root), before driver.py:
#
#   export SCIMT_COMMIT=<full 40-hex sha of the commit to run>   # required
#   export HF_TOKEN=hf_...                                       # required (gated gemma-3, private coin repo)
#   export GITHUB_TOKEN=ghp_...   # required unless SCIMT_BUNDLE points at a pre-staged git bundle
#   bash bootstrap.sh
#
# What it does, in order (every step is a no-op when already done):
#   1. apt basics (git, curl, jq, rsync, tmux, zstd) if missing; `ulimit -n 65536`.
#   2. Host-spec gate (ported from gate2 pod/host_probe.py, bash side): effective RAM =
#      min(MemTotal, cgroup limit) >= SCIMT_MIN_HOST_RAM_GB (400) and free disk under
#      /workspace >= SCIMT_MIN_FREE_DISK_GB (800). Fails with `SCIMT-HOST-SPEC-GATE-FAIL`,
#      exit 96, before any download. NB the driver's own disk plan (evidence/gate_a.json,
#      `disk_plan`) projects the true peak for six datasets x (gdp+gdpunit) x 3 folds plus 30
#      inverse vectors at 43 GB each: ~2.1-2.6 TB. 800 GB is the floor the fit itself
#      refuses below; provision the volume for the driver's number.
#   3. Clone/fetch the repo into /workspace/scimt and check out $SCIMT_COMMIT (detached);
#      refuses if HEAD != SCIMT_COMMIT afterwards.
#   4. Network probe: gate2's pod/host_probe.py (stdlib) against Hugging Face —
#      SCIMT_MIN_NET_MBPS (default 10 MB/s); exit 96 on a host-lottery loser.
#   5. uv (installed if missing; cache on /workspace) + `uv sync` with the extras
#      data-attribution-ekfac, analysis, dev, hub, then `uv pip install zstandard`
#      (Dolmino .zst shards; not an extra).
#   6. HF login from $HF_TOKEN (token file under $HF_HOME, default /workspace/hf).
#   7. Pre-download google/gemma-3-12b-pt @295efb63 and google/gemma-3-12b-it @96b6f1ec
#      into the HF cache; record the resolved snapshot paths/shas.
#   8. Write /workspace/attribution/evidence/bootstrap.json (host, repo, venv versions,
#      model snapshots, hf_home) — driver.py reads it (models.{pt,it}.path, hf_home) —
#      and /workspace/attribution/env.sh to `source` before launching the driver.
#
# Exit codes: 0 ok; 96 host-spec gate; anything else = a real setup failure (read the log).
# Log: /workspace/attribution/evidence/bootstrap.log (appended per run).
set -euo pipefail

# ---------------------------------------------------------------- settings
: "${SCIMT_COMMIT:?SCIMT_COMMIT (full commit sha) is required — refuse to run an unpinned tree}"
: "${HF_TOKEN:?HF_TOKEN is required (gated google/gemma-3 snapshots + private coin corpus repo)}"
if ! [[ "$SCIMT_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SCIMT_COMMIT must be a full 40-hex sha, got '$SCIMT_COMMIT'" >&2
  exit 2
fi

WORKSPACE=${SCIMT_WORKSPACE:-/workspace}
REPO_DIR=${SCIMT_REPO_DIR:-$WORKSPACE/scimt}
REPO_URL=${SCIMT_REPO_URL:-https://github.com/arcadiaimpact/science-of-midtraining}
ATTR_ROOT=${SCIMT_ATTRIBUTION_ROOT:-$WORKSPACE/attribution}
EVIDENCE=$ATTR_ROOT/evidence
export HF_HOME=${HF_HOME:-$WORKSPACE/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WORKSPACE/.uv-cache}
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export HF_HUB_ENABLE_HF_TRANSFER=${SCIMT_HF_TRANSFER:-0}
MIN_RAM_GB=${SCIMT_MIN_HOST_RAM_GB:-400}
MIN_DISK_GB=${SCIMT_MIN_FREE_DISK_GB:-800}
MIN_NET_MBPS=${SCIMT_MIN_NET_MBPS:-10}
PT_ID=google/gemma-3-12b-pt
PT_REV=295efb63d01a7017928f273a94ebb86105c9526f
IT_ID=google/gemma-3-12b-it
IT_REV=${SCIMT_IT_REVISION:-96b6f1eccf38110c56df3a15bffe176da04bfd80}
HOST_SPEC_EXIT=96
HOST_SPEC_SENTINEL=SCIMT-HOST-SPEC-GATE-FAIL

mkdir -p "$EVIDENCE" "$HF_HOME" "$UV_CACHE_DIR"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
exec > >(tee -a "$EVIDENCE/bootstrap.log") 2>&1
echo "[$STARTED_AT] bootstrap start commit=$SCIMT_COMMIT repo=$REPO_DIR hf_home=$HF_HOME"

# -------------------------------------------------------------- 1. basics
if command -v apt-get >/dev/null 2>&1; then
  missing=()
  for tool in git curl jq rsync tmux zstd; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done
  if ((${#missing[@]})); then
    echo "apt: installing ${missing[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq "${missing[@]}"
  fi
fi
ulimit -n 65536 || echo "WARN: ulimit -n 65536 refused (hard limit $(ulimit -Hn)); the driver raises its own soft limit"
echo "ulimit -n: $(ulimit -n)"

# ------------------------------------------------- 2. host-spec gate (RAM/disk)
mem_total_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
mem_total_gb=$(awk -v kb="$mem_total_kb" 'BEGIN {printf "%.0f", kb * 1024 / 1e9}')
cgroup_limit_gb=""
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  if [[ -r $f ]]; then
    raw=$(cat "$f")
    if [[ $raw != max && $raw =~ ^[0-9]+$ ]] && (( raw > 0 )) && (( raw < 100 * 1024 * 1024 * 1024 * 1024 )); then
      cgroup_limit_gb=$(awk -v b="$raw" 'BEGIN {printf "%.0f", b / 1e9}')
    fi
    break
  fi
done
effective_ram_gb=$mem_total_gb
if [[ -n $cgroup_limit_gb ]] && (( cgroup_limit_gb < mem_total_gb )); then
  effective_ram_gb=$cgroup_limit_gb
fi
free_disk_gb=$(df --output=avail -B1 "$WORKSPACE" | tail -1 | awk '{printf "%.0f", $1 / 1e9}')
gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd'|' -) || gpu_names=""
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ') || gpu_count=0
echo "host: MemTotal=${mem_total_gb}GB cgroup=${cgroup_limit_gb:-none} effective=${effective_ram_gb}GB free_disk=${free_disk_gb}GB gpus=${gpu_count} (${gpu_names})"
failures=()
(( effective_ram_gb >= MIN_RAM_GB )) || failures+=("effective RAM ${effective_ram_gb} GB < ${MIN_RAM_GB} GB")
(( free_disk_gb >= MIN_DISK_GB )) || failures+=("free disk ${free_disk_gb} GB under $WORKSPACE < ${MIN_DISK_GB} GB")
if ((${#failures[@]})); then
  echo "$HOST_SPEC_SENTINEL: ${failures[*]}"
  exit $HOST_SPEC_EXIT
fi

# ------------------------------------------------------------ 3. repo checkout
if [[ -d $REPO_DIR/.git ]]; then
  echo "repo: fetching into existing $REPO_DIR"
  git -C "$REPO_DIR" fetch --quiet origin || echo "WARN: fetch failed; trying the local object store"
elif [[ -n ${SCIMT_BUNDLE:-} && -f ${SCIMT_BUNDLE} ]]; then
  echo "repo: cloning from bundle $SCIMT_BUNDLE"
  git clone --quiet "$SCIMT_BUNDLE" "$REPO_DIR"
else
  : "${GITHUB_TOKEN:?GITHUB_TOKEN is required to clone the private repo (or pre-stage a bundle via SCIMT_BUNDLE)}"
  echo "repo: cloning $REPO_URL"
  git clone --quiet "https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}" "$REPO_DIR"
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
fi
if ! git -C "$REPO_DIR" cat-file -e "${SCIMT_COMMIT}^{commit}" 2>/dev/null; then
  echo "repo: $SCIMT_COMMIT not present after fetch — is the branch pushed?" >&2
  exit 3
fi
git -C "$REPO_DIR" checkout --quiet --detach "$SCIMT_COMMIT"
HEAD_SHA=$(git -C "$REPO_DIR" rev-parse HEAD)
if [[ $HEAD_SHA != "$SCIMT_COMMIT" ]]; then
  echo "repo: HEAD $HEAD_SHA != SCIMT_COMMIT $SCIMT_COMMIT" >&2
  exit 3
fi
if [[ -n $(git -C "$REPO_DIR" status --porcelain --untracked-files=no) ]]; then
  echo "repo: working tree is dirty at $SCIMT_COMMIT — refusing (provenance)" >&2
  git -C "$REPO_DIR" status --short | head -20
  exit 3
fi
rm -rf "$REPO_DIR/src/scimt.egg-info"
echo "repo: at $HEAD_SHA"

# -------------------------------------------------------- 4. network probe
export SCIMT_MIN_HOST_RAM_GB=$MIN_RAM_GB SCIMT_MIN_NET_MBPS=$MIN_NET_MBPS
net_probe_status=0
python3 "$REPO_DIR/experiments/improved_midtraining/gate2_lineage_attribution/pod/host_probe.py" || net_probe_status=$?
if (( net_probe_status == HOST_SPEC_EXIT )); then
  echo "$HOST_SPEC_SENTINEL: host_probe.py refused this host (see above)"
  exit $HOST_SPEC_EXIT
elif (( net_probe_status != 0 )); then
  echo "host_probe.py exited $net_probe_status" >&2
  exit "$net_probe_status"
fi

# ------------------------------------------------------------- 5. venv
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv: installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
cd "$REPO_DIR"
echo "uv: sync (data-attribution-ekfac, analysis, dev, hub)"
uv sync --extra data-attribution-ekfac --extra analysis --extra dev --extra hub
uv pip install zstandard
if [[ ${SCIMT_HF_TRANSFER:-0} == 1 ]]; then uv pip install hf_transfer; fi
VENV_PY="$REPO_DIR/.venv/bin/python"
"$VENV_PY" - <<'PY'
import importlib.metadata as m, json
out = {}
for name in ("torch", "transformers", "kronfluence", "numpy", "zstandard", "seaborn", "statsmodels", "huggingface_hub", "pandas", "scipy"):
    try:
        out[name] = m.version(name)
    except m.PackageNotFoundError:
        out[name] = None
import torch
out["cuda_available"] = torch.cuda.is_available()
out["cuda_devices"] = torch.cuda.device_count()
missing = [k for k in ("torch", "transformers", "kronfluence", "zstandard", "seaborn") if out[k] is None]
if missing:
    raise SystemExit(f"venv is missing {missing}")
if out["kronfluence"] != "1.0.1":
    raise SystemExit(f"kronfluence {out['kronfluence']} installed; fit_factors_pt pins 1.0.1")
if out["cuda_devices"] < 2:
    raise SystemExit(f"only {out['cuda_devices']} CUDA devices visible; the driver needs >= 2")
print("venv:", json.dumps(out))
PY

# ------------------------------------------------------------ 6. HF login
"$VENV_PY" - <<'PY'
import os
from huggingface_hub import login, whoami
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("hf: logged in as", whoami()["name"])
PY

# ---------------------------------------------- 7. model snapshots (pinned)
"$VENV_PY" - "$PT_ID" "$PT_REV" "$IT_ID" "$IT_REV" "$EVIDENCE/models.json" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
pt_id, pt_rev, it_id, it_rev, out = sys.argv[1:6]
records = {}
for key, repo, rev in (("pt", pt_id, pt_rev), ("it", it_id, it_rev)):
    path = Path(snapshot_download(repo_id=repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py"]))
    sha = path.name if path.parent.name == "snapshots" else rev
    if not sha.startswith(rev[:7]):
        raise SystemExit(f"{repo}: resolved snapshot {sha} does not match pinned {rev}")
    files = sorted(p.name for p in path.iterdir())
    if not any(f.endswith(".safetensors") for f in files):
        raise SystemExit(f"{repo}: snapshot has no safetensors shards")
    records[key] = {"hf_id": repo, "revision": rev, "sha": sha, "path": str(path), "n_files": len(files),
                    "bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file())}
    print(f"model {key}: {repo}@{sha} -> {path} ({records[key]['bytes'] / 1e9:.1f} GB)")
Path(out).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
PY

# ---------------------------------------------------------- 8. receipts
cat > "$ATTR_ROOT/env.sh" <<EOF
# source me before launching driver.py (written by bootstrap.sh at $STARTED_AT)
export HF_HOME=$HF_HOME
export UV_CACHE_DIR=$UV_CACHE_DIR
export UV_LINK_MODE=$UV_LINK_MODE
export HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER
export SCIMT_COMMIT=$SCIMT_COMMIT
export SCIMT_REPO_DIR=$REPO_DIR
export SCIMT_ATTRIBUTION_ROOT=$ATTR_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
unset RUNPOD_API_KEY
ulimit -n 65536 2>/dev/null || true
EOF

FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
"$VENV_PY" - "$EVIDENCE/bootstrap.json" "$EVIDENCE/models.json" "$STARTED_AT" "$FINISHED_AT" \
  "$REPO_DIR" "$HEAD_SHA" "$HF_HOME" "$mem_total_gb" "${cgroup_limit_gb:-}" "$effective_ram_gb" "$free_disk_gb" \
  "$gpu_count" "$gpu_names" "$(ulimit -n)" "$MIN_RAM_GB" "$MIN_DISK_GB" "$MIN_NET_MBPS" <<'PY'
import importlib.metadata as m, json, platform, sys
from pathlib import Path
(out, models, started, finished, repo, sha, hf_home, mem_total, cgroup, effective, free_disk,
 gpu_count, gpu_names, nofile, min_ram, min_disk, min_net) = sys.argv[1:18]
def version(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return None
payload = {
    "schema": "ekfac_dataset_attribution_v1/bootstrap/1",
    "status": "ok",
    "started_at": started,
    "finished_at": finished,
    "host": {
        "hostname": platform.node(),
        "mem_total_gb": float(mem_total),
        "cgroup_limit_gb": float(cgroup) if cgroup else None,
        "effective_ram_gb": float(effective),
        "disk_free_gb_at_bootstrap": float(free_disk),
        "gpu_count": int(gpu_count),
        "gpu_names": gpu_names.split("|") if gpu_names else [],
        "nofile_soft_limit": int(nofile),
        "thresholds": {"min_host_ram_gb": float(min_ram), "min_free_disk_gb": float(min_disk), "min_net_mbps": float(min_net)},
    },
    "repo": {"path": repo, "commit": sha},
    "venv": {"python": sys.version.split()[0], **{k: version(k) for k in ("torch", "transformers", "kronfluence", "numpy", "zstandard", "seaborn", "statsmodels", "huggingface_hub")}},
    "models": json.loads(Path(models).read_text()),
    "hf_home": hf_home,
    "env_file": str(Path(out).parents[1] / "env.sh"),
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("wrote", out)
PY
echo "[$FINISHED_AT] SCIMT-BOOTSTRAP-DONE commit=$HEAD_SHA models=$EVIDENCE/models.json receipt=$EVIDENCE/bootstrap.json"
