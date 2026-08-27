# glm_minimal_v1 operator runbook

No bellhop. No automatic pod creation or termination. Run every pod command
below by hand. Do not pass `--max-hours`: the operator alone destroys the pod.

## 0. Choose and create the pod — local terminal

Required for either option: 8 GPUs, 1,600 GB container disk, host RAM and
cgroup cap both at least 1,900 GB.

| option | status | GPU id | CUDA host filter / wheels | SECURE | COMMUNITY | 8-GPU SECURE |
|---|---|---|---|---:|---:|---:|
| H200 141 GB | proven | `NVIDIA H200` | host `12.6,12.7,12.8,12.9,13.0,13.1`; cu126 | $4.59/GPU-h | $3.59/GPU-h | **$36.72/h** |
| B300 288 GB | unverified | `NVIDIA B300 SXM6 AC` | host `13.0,13.1`; cu130, sm_103 | $7.89/GPU-h | $6.94/GPU-h | **$63.12/h** |

`runpodctl` cannot filter by host CUDA version. Use `create-pod-cuda.sh`, not
`create-pod.sh`. The CUDA helper inserts the required CSV at argument 3.

```text
create-pod.sh <name> <gpu-id> [cloud] [template] [gpu-count] [disk-gb]
create-pod-cuda.sh <name> <gpu-id> <cuda-versions-csv> [cloud] [template] [gpu-count] [disk-gb]
```

H200 SECURE:

```bash
POD_NAME="$(date -u +%Y%m%d)_glm_minimal_h200"
~/.claude/skills/runpod-spinup/create-pod-cuda.sh \
  "$POD_NAME" "NVIDIA H200" \
  "12.6,12.7,12.8,12.9,13.0,13.1" \
  SECURE runpod-torch-v280 8 1600
```

B300 SECURE:

```bash
POD_NAME="$(date -u +%Y%m%d)_glm_minimal_b300"
~/.claude/skills/runpod-spinup/create-pod-cuda.sh \
  "$POD_NAME" "NVIDIA B300 SXM6 AC" \
  "13.0,13.1" \
  SECURE runpod-torch-v280 8 1600
```

COMMUNITY only after accepting the helper's SSH-agent-forwarding warning:

```bash
# H200 COMMUNITY
~/.claude/skills/runpod-spinup/create-pod-cuda.sh \
  "$(date -u +%Y%m%d)_glm_minimal_h200" "NVIDIA H200" \
  "12.6,12.7,12.8,12.9,13.0,13.1" \
  COMMUNITY runpod-torch-v280 8 1600 --accept-community-risk

# B300 COMMUNITY
~/.claude/skills/runpod-spinup/create-pod-cuda.sh \
  "$(date -u +%Y%m%d)_glm_minimal_b300" "NVIDIA B300 SXM6 AC" \
  "13.0,13.1" \
  COMMUNITY runpod-torch-v280 8 1600 --accept-community-risk
```

Record the printed pod id:

```bash
export POD_ID='<pod-id printed by create-pod-cuda.sh>'
ssh "runpod-$POD_NAME"
```

An idle hour costs $36.72 on H200 SECURE or $63.12 on B300 SECURE.

## 1. Accept or re-roll the host immediately

Run before installs:

```bash
awk '/MemTotal/ {printf "host RAM: %.1f GB\n", $2*1024/1e9}' /proc/meminfo
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  test -r "$f" && printf 'cgroup: %s = %s\n' "$f" "$(<"$f")"
done
df -BG /workspace
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
```

Accept only:

- host RAM >= 1,900 GB;
- cgroup `memory.max` is `max` or >= 1,900,000,000,000 bytes;
- `/workspace` free >= 1,400 GB;
- exactly 8 requested GPUs, each >= 140 GiB, with no resident compute process;
- setup reports >= 20,000,000 B/s from both the PyTorch CDN and
  `files.pythonhosted.org`;
- preflight egress preferably >= 100 MB/s. Below 100 MB/s is a warning and an
  operator decision, not an automatic abort. A 16 MB/s host took 3 h 38 m per
  214 GB publish.

If setup says it could not construct the `files.pythonhosted.org` probe, do
not accept the host until that CDN has been measured; the script's fallback is
warning-only, but this run requires both ingress gates.

Re-roll any hard-gate failure. Jonathan's campaign needed about 13 attempts
before a good 8xH200 host; budget about $40 for probes. A re-roll is normal,
not a failed campaign.

From the local terminal:

```bash
runpodctl pod delete "$POD_ID"
# Re-run the selected creation command with a fresh POD_NAME.
```

## 2. Credentials and immutable inputs — pod shell

Create the private target model repo and a disposable private preflight model
repo in the HF console first. The token needs read access to pinned inputs and
write access to both repos.

```bash
export HF_HOME=/workspace/hf-cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_ENABLE_HF_TRANSFER=1

read -rsp 'HF write token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
export SCIMT_HF_TARGET_REPO='<org/private-output-model-repo>'
export SCIMT_HF_REPO_TYPE=model
export SCIMT_HF_PREFLIGHT_REPO='<org/private-preflight-model-repo>'
export SCIMT_HF_PREFLIGHT_REPO_TYPE=model
export SCIMT_DATA_REVISION='<immutable data revision printed by build_data.py>'

# Reconciliation only; choose the row matching the pod.
export SCIMT_GPU_TYPE=H200
export SCIMT_USD_PER_GPU_HOUR=4.59
# B300 SECURE instead: SCIMT_GPU_TYPE=B300; SCIMT_USD_PER_GPU_HOUR=7.89
# COMMUNITY instead: H200=3.59; B300=6.94
```

Never put `HF_TOKEN` in a file, command argument, tmux command string, log,
chat prompt, or shell history. Enter it only through the silent `read`; tmux
inherits the exported environment.

## 3. Clone and bootstrap — pod shell

```bash
cd /workspace
export SCIMT_REF='<reviewed commit SHA containing glm_minimal_v1>'
git clone git@github.com:ArcadiaImpact/science-of-midtraining.git scimt
git -C /workspace/scimt checkout --detach "$SCIMT_REF"
cd /workspace/scimt
bash experiments/prior_coins/glm_minimal_v1/pod/setup_pod.sh
```

Healthy setup lines:

```text
network preflight (pytorch cdn): <at least 20000000> B/s
network preflight (files.pythonhosted.org): <at least 20000000> B/s
[smoke] GPU 0: grouped_mm forward+backward OK; ...
...
[smoke] GPU 7: grouped_mm forward+backward OK; ...
pod setup complete
```

H200 must print a cu126 torch build and capability `(9, 0)`. B300 must print a
cu130 build, capability `(10, 3)`, and `sm_103` in `torch.cuda.get_arch_list()`.
B300 is unverified: do not proceed if grouped-mm, CCE import, sm_103, or fused
FP32 AdamW viability is uncertain.

## 4. Preflight — pod shell

```bash
cd /workspace/scimt
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="/workspace/glm-minimal-v1/runs/$RUN_ID"
mkdir -p "$RUN_ROOT"
/workspace/venv-glm/bin/python \
  experiments/prior_coins/glm_minimal_v1/pod/preflight.py \
  --output-dir "$RUN_ROOT"
```

Healthy final line:

```text
preflight OK: ... 1900+ GB host RAM, ... 1400+ GB disk, 8 idle GPUs, HF writable, egress ... MB/s; wrote .../preflight.json
```

`BAD HOST -- RE-ROLL` means delete and recreate. `BAD CONFIG -- FIX IT` means
fix the image, repo, or credentials on the same host. The chain deliberately
runs preflight again and records that second result in phase telemetry.

## 5. Run the chain under tmux — pod shell

```bash
cd /workspace/scimt
tmux -L glm-minimal-v1 new-session -d -s "$RUN_ID" \
  "cd /workspace/scimt && /workspace/venv-glm/bin/python experiments/prior_coins/glm_minimal_v1/pod/chain.py --run-id '$RUN_ID' >>'$RUN_ROOT/chain.log' 2>&1"
tmux -L glm-minimal-v1 attach-session -t "$RUN_ID"
```

Detach with `Ctrl-b d`. Reattach:

```bash
tmux -L glm-minimal-v1 attach-session -t "$RUN_ID"
```

The costing model prints the following critical-path estimates. B300 is the
pessimistic–optimistic range with the central estimate in parentheses.

| phase, actual chain order | H200 | B300 range (central) | healthy evidence |
|---|---:|---:|---|
| pod provision + env | 1.50 h | 1.50 h | `pod setup complete` |
| base download, 221 GB | 0.50 h | 0.50 h | telemetry `phase=download`; setup download marker exists |
| data fetch + gates | 0.50 h | 0.50 h | telemetry `phase=data_fetch`; `preflight OK` |
| charter midtrain, 152 nominal steps | 1.44 h | 0.44–0.85 h (0.63) | finite `'loss': '...'`; router JSONL advances |
| charter midtrain merge | 0.06 h | 0.06 h | telemetry `phase=merge` |
| charter IFT | 3.57 h model | 1.10–2.11 h (1.56) | label gate passes; finite loss; router JSONL advances |
| charter IFT merge | 0.06 h | 0.06 h | telemetry `phase=merge`; background publish starts |
| coin midtrain + merge | 1.50 h | 0.50–0.91 h (0.69) | same signals; charter publish may overlap |
| coin IFT + merge | 3.63 h model | 1.16–2.17 h (1.62) | same signals; verified publish line |
| AFT charter + coin, concurrent, 4 GPUs each, 512 steps | 1.99 h | 0.61–1.17 h (0.87) | both AFT label gates and router logs pass |
| four eval endpoints, concurrent, 2 GPUs each | 0.42 h | 0.42 h | four `ENDPOINT.json` files; post probes pass |
| final checkpoint publish tail | 0.12 h at 500 MB/s | 0.12 h | `publish verified: ... at ... MB/s` |
| total | **15.3 h / ~$562** | **7.0–10.4 h / $440–654** (**8.5 h / $539**) | `CHAIN_COMPLETE` |

IFT correction: `minimal_glm_run.py` models nominal 100M as 47 steps and
therefore prints 3.57 h. The chain and both SFT configs actually run the pinned
100,663,296 positions: **48 steps**, measured at 269.9 s/step, about **3.65 h
per arm** on H200. Trust the 48-step checkpoint marker, not the model's label.
Midtrain steps are recomputed with the GLM tokenizer; 152 is the nominal model
estimate and may differ if the realized GLM-token count differs.

Final healthy chain line:

```text
CHAIN_COMPLETE: every requested artifact publish was joined and verified
```

## 6. Watch-list — second pod shell

Set the same non-secret run id:

```bash
export RUN_ID='<same run id>'
export RUN_ROOT="/workspace/glm-minimal-v1/runs/$RUN_ID"
tail -F "$RUN_ROOT/chain.log"
```

Loss guard:

```bash
find "/workspace/glm-minimal-v1/train/$RUN_ID" -name train.log -type f -print
tail -n 20 "$(find "/workspace/glm-minimal-v1/train/$RUN_ID" -name train.log -type f -print | tail -n 1)"
```

- Healthy: finite loss, broadly flat/down.
- Fatal: NaN/inf immediately, or five consecutive post-grace losses above
  `max(1.5 * running_min, running_min + 0.5)`. The guard kills the subprocess.

Router health, every 10 optimizer steps:

```bash
find "/workspace/glm-minimal-v1/train/$RUN_ID" -name router_health.jsonl -type f \
  -exec sh -c 'printf "%s\n" "$1"; tail -n 1 "$1"' _ {} \;
```

- Healthy observed entropy: **4.18–4.81 nats**, without a sustained fall or
  `[router-health] WARN ... entropy fell ...` concentration signal.
- Collapse or an empty/missing router file is not healthy.

Free disk at completed phase boundaries:

```bash
df -h /workspace
tail -n 10 "$RUN_ROOT/telemetry.jsonl"
```

- Healthy: every row has `free_disk_gb`; space recovers after verified
  publishes and cache cleanup. Before another full merge, retain room for a
  ~199 GiB consolidated model plus the sharded checkpoint. Investigate any
  sharp monotonic fall; never wait for ENOSPC.

Uploads:

```bash
grep -E 'publish background|publish verified|telemetry final refresh' "$RUN_ROOT/chain.log"
```

- Healthy: background start/finish pairs and `publish verified: ... at N.N
  MB/s`.
- Warn below 100 MB/s. The chain does not abort slow egress.

Label-mask gates:

```bash
find "$RUN_ROOT" -name '*_label_mask.json' -type f -exec sh -c 'printf "%s\n" "$1"; /workspace/venv-glm/bin/python -m json.tool "$1"' _ {} \;
```

- Healthy for charter/coin IFT and AFT: `trained_tokens > 0`, `masked_tokens >
  0`, `trained_fraction` in `[0.05, 0.90]`, and `terminator_trained: true`.
- Missing report means training has not passed its pre-GPU label gate.

Adapter divergence probe — mandatory:

```bash
find "$RUN_ROOT/eval" -path '*post_aft*/ENDPOINT.json' -type f \
  -exec sh -c 'printf "%s\n" "$1"; /workspace/venv-glm/bin/python -m json.tool "$1"' _ {} \;
```

- Healthy: both post-AFT files have `probe_passed: true`,
  `probe.divergence_rate >= 0.10`, and candidate exact matches no worse than
  base. `serving_path` is `native_lora` or `merged_fallback`.
- A silently inert adapter produces confident garbage. This probe is the only
  gate that catches it. Never score unprobed post-AFT rows.

## 7. Score and reconcile after `CHAIN_COMPLETE`

The chain stores each endpoint below its own content-key directory; `score.py`
expects four sibling endpoint directories. Build a symlink-only scoring view:

```bash
cd /workspace/scimt
export SCORE_INPUT="$RUN_ROOT/score_input"
export SCORES_DIR="$RUN_ROOT/scores"
mkdir -p "$SCORE_INPUT" "$SCORES_DIR"
for arm in charter coin; do
  for endpoint in pre_aft post_aft; do
    cell=$(find "$RUN_ROOT/eval/$arm/$endpoint" \
      -path "*/$arm-$endpoint/ENDPOINT.json" -type f -print)
    test "$(printf '%s\n' "$cell" | sed '/^$/d' | wc -l)" -eq 1
    result_dir=$(dirname "$cell")
    ln -sfn "$result_dir" "$SCORE_INPUT/$arm-$endpoint"
  done
done

/workspace/venv-glm/bin/python \
  experiments/prior_coins/glm_minimal_v1/score.py \
  "$SCORE_INPUT" \
  /workspace/glm-minimal-v1/data/eval/extensions/template_diversity_v1/data \
  --output-dir "$SCORES_DIR"

/workspace/venv-glm/bin/python \
  experiments/prior_coins/glm_minimal_v1/reconcile_cost.py \
  "$RUN_ROOT/telemetry.jsonl" \
  --gpu-type "$SCIMT_GPU_TYPE" \
  --usd-per-gpu-hour "$SCIMT_USD_PER_GPU_HOUR" \
  --pod-gpus 8 | tee "$SCORES_DIR/cost_reconciliation.txt"
```

Healthy scoring line:

```text
wrote .../scores.json and .../summary.md
```

Upload and verify scores:

```bash
export SCORES_DIR RUN_ID SCIMT_HF_TARGET_REPO
/workspace/venv-glm/bin/python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    repo_id=os.environ["SCIMT_HF_TARGET_REPO"],
    repo_type="model",
    folder_path=os.environ["SCORES_DIR"],
    path_in_repo=f'runs/{os.environ["RUN_ID"]}/scores',
    commit_message=f'publish scores {os.environ["RUN_ID"]}',
)
print("scores upload complete")
PY
```

## 8. Failure playbook

Pod or SSH session dies:

Re-enter the section 2 environment after reconnecting; never put the token in
the tmux command itself. Then:

```bash
cd /workspace/scimt
tmux -L glm-minimal-v1 new-session -d -s "${RUN_ID}-resume" \
  "cd /workspace/scimt && /workspace/venv-glm/bin/python experiments/prior_coins/glm_minimal_v1/pod/chain.py --run-id '$RUN_ID' --resume >>'$RUN_ROOT/chain.log' 2>&1"
tmux -L glm-minimal-v1 attach-session -t "${RUN_ID}-resume"
```

- Never `rm -rf` a run directory to “start clean.” Relaunch with the same run
  id and `--resume`. Local caches plus content-checked HF completion markers
  make completed work nearly free to reuse.

ENOSPC or rapidly falling free disk:

```bash
du -sh /workspace/hf-cache/xet /root/.cache/huggingface/xet 2>/dev/null || true
rm -rf -- /workspace/hf-cache/xet /root/.cache/huggingface/xet
df -h /workspace
```

- Purge only the Xet caches. Do not delete the run directory, checkpoints, or
  consolidated parents. Resume.

Slow egress:

- Let it finish and record the measured MB/s, or abandon and re-roll. It is a
  warning, not a chain abort.

Adapter probe failure:

- Native LoRA probe failure automatically triggers the full-weight merge
  fallback. Healthy completion records `serving_path: merged_fallback` and the
  native failure. If the merged probe also fails, the chain aborts before
  writing scored rows. Do not bypass it.

Loss divergence:

```bash
tail -n 200 "$(find "/workspace/glm-minimal-v1/train/$RUN_ID" -name train.log -type f -print | tail -n 1)"
```

- The loss guard already killed the stage. Read the tail before relaunching.

## 9. Teardown gate — verify from HF, then destroy locally

Required on HF before deletion:

- two full IFT-end parents: `charter/ift`, `coin/ift`;
- two LoRA adapters: `charter/aft`, `coin/aft`;
- four raw eval endpoint folders;
- metadata including `telemetry.jsonl`, RAM telemetry, label-mask reports, and
  router-health logs;
- `scores/scores.json`, `scores/summary.md`, and cost reconciliation.

The eight `_STAGE_COMPLETE.json` files are written only after remote folder
verification. Verify their content plus the unmarked metadata/score files:

```bash
export RUN_ID SCIMT_HF_TARGET_REPO
/workspace/venv-glm/bin/python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

run_id = os.environ["RUN_ID"]
repo = os.environ["SCIMT_HF_TARGET_REPO"]
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
markers = [
    f"runs/{run_id}/{arm}/{stage}/_STAGE_COMPLETE.json"
    for arm in ("charter", "coin")
    for stage in ("ift", "aft")
] + [
    f"runs/{run_id}/eval/{arm}/{endpoint}/_STAGE_COMPLETE.json"
    for arm in ("charter", "coin")
    for endpoint in ("pre_aft", "post_aft")
]
for marker in markers:
    local = hf_hub_download(repo, marker, repo_type="model", token=token)
    body = json.loads(Path(local).read_text())
    assert body["run_id"] == run_id, (marker, body)
    print("verified marker", marker, body["stage"])
required = [
    f"runs/{run_id}/metadata/telemetry.jsonl",
    f"runs/{run_id}/scores/scores.json",
    f"runs/{run_id}/scores/summary.md",
    f"runs/{run_id}/scores/cost_reconciliation.txt",
]
for filename in required:
    assert api.file_exists(repo, filename, repo_type="model"), filename
    print("verified file", filename)
print("HF_TEARDOWN_GATE_OK")
PY
```

Only after `HF_TEARDOWN_GATE_OK`, from the local terminal:

```bash
runpodctl pod delete "$POD_ID"
```
