#!/usr/bin/env bash
# Close out ONE finished arm: prove it is persisted, keep its evidence, then
# (only with --terminate) destroy the pod. Mirrors the 1B run's durability
# gate: CHAIN_COMPLETE.json + PUBLISH_COMPLETE.json on the pod, and the Hub
# tree under <profile>/charter/ in the profile's model repo checked against
# the shape the 1B row published (sentinels, aft/<cell>, dolci, eval,
# midtrain/consolidated + checkpoints, recall, d4, costsweep, data).
#
# Usage: finish_arm.sh <profile> <pod-id> <ssh-alias> [--terminate]
#   Without --terminate this is read-only apart from the local evidence copy.
#   --terminate runs the skill's cleanup-pod.sh preview, checks it names this
#   exact pod, then cleanup-pod.sh --yes. THAT DESTROYS THE CONTAINER DISK.
set -euo pipefail

PROFILE=${1:?usage: finish_arm.sh <profile> <pod-id> <ssh-alias> [--terminate]}
POD_ID=${2:?usage: finish_arm.sh <profile> <pod-id> <ssh-alias> [--terminate]}
ALIAS=${3:?usage: finish_arm.sh <profile> <pod-id> <ssh-alias> [--terminate]}
TERMINATE=0; [ "${4:-}" = "--terminate" ] && TERMINATE=1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STUDY=$(cd "$HERE/.." && pwd)
EXP=$(cd "$STUDY/.." && pwd)
REPO=$(cd "$EXP/../../.." && pwd)
SKILL=${SKILL:-/root/.claude/skills/runpod-spinup}
PY=${PY:-$REPO/.venv/bin/python}
ROOT=/workspace/final_v1/$PROFILE/charter
case "$PROFILE" in glm45_air_500m_noex|glm45_air_500m_worked) ;;
  *) echo "FATAL: unknown profile $PROFILE" >&2; exit 64;; esac
case "$POD_ID$ALIAS" in *[!A-Za-z0-9_-]*) echo "FATAL: bad pod id / alias" >&2; exit 64;; esac
say() { echo "[$(date -u +%FT%TZ)] $*"; }
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=30 "$ALIAS" "$@"; }

# ------------------------------------------------------------- 1 sentinels
say "1/4 sentinels on the pod ($ROOT)"
rsh "test -f $ROOT/CHAIN_COMPLETE.json && test -f $ROOT/PUBLISH_COMPLETE.json" \
  || { echo "NOT FINISHED: CHAIN_COMPLETE.json / PUBLISH_COMPLETE.json missing under $ROOT" >&2; exit 1; }
rsh "cd $ROOT && for f in PUBLISHED_*.json; do python3 -c \"import json,sys; d=json.load(open('\$f')); print(f'  {\$f!s:28} files={d.get(\\\"files\\\")} bytes={d.get(\\\"total_bytes\\\")} path={d.get(\\\"path_in_repo\\\")}')\" ; done; python3 -c \"import json; d=json.load(open('CHAIN_COMPLETE.json')); print('  CHAIN_COMPLETE at', d.get('at'), 'endpoints', d.get('endpoints'), 'aft', d.get('aft_cells'))\""

# -------------------------------------------------------------- 2 evidence
RUN_DIR="$STUDY/run_$(date -u +%Y-%m-%d)_${POD_ID}"
say "2/4 evidence -> $RUN_DIR"
mkdir -p "$RUN_DIR"
rsync -az --prune-empty-dirs --max-size=5m \
  --include='*/' --include='*_COMPLETE.json' --include='PUBLISHED_*.json' --include='PUBLISH_COMPLETE.json' \
  --include='SCHEDULE.json' --include='PREFLIGHT.json' --include='EVAL_PARENT_RECLAIMED.json' \
  --include='*.log' --include='leg_a_mix.yaml' --include='trainer_state.final.json' --exclude='*' \
  "$ALIAS:$ROOT/" "$RUN_DIR/"
rsync -az "$ALIAS:/workspace/logs/dfv1_${PROFILE}__charter.log" "$RUN_DIR/unit_runner.log" || true
rsync -az "$ALIAS:/workspace/logs/setup_${PROFILE}.log" "$RUN_DIR/setup.log" || true
find "$RUN_DIR" -type f | wc -l | xargs -I{} echo "  {} files kept"

# ------------------------------------------------------------------- 3 hub
say "3/4 Hub tree vs the 1B row's shape"
"$PY" - "$PROFILE" "$EXP" <<'PY'
import os, sys, collections
sys.path[:0] = [sys.argv[2], sys.argv[2] + "/pod"]
from huggingface_hub import HfApi
import contracts as C
profile = sys.argv[1]
repo = C.model_repo_for(profile)
files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(repo, repo_type="model")
def shape(prefix):
    c = collections.Counter()
    for f in files:
        if f.startswith(prefix):
            parts = f[len(prefix):].split("/")
            c["/".join(parts[:2]) if parts[0] in ("midtrain", "aft") and len(parts) > 2 else parts[0]] += 1
    return c
ours, ref = shape(f"{profile}/charter/"), shape("glm45_air_1b/charter/")
# Floors: ~80% of what the 1B row published per group; the resume backup
# groups (midtrain/RESUME_*) are deliberately absent on these rows.
groups = {"aft/agreement": 100, "aft/mixed_charter": 100, "aft/mixed_coin": 100, "aft/charter_only": 100,
          "dolci": 100, "eval": 100, "midtrain/consolidated": 40, "midtrain/checkpoints": 40,
          "recall": 12, "d4": 15, "costsweep": 15, "data": 5}
sentinels = ["MIX_COMPLETE.json", "MIDTRAIN_COMPLETE.json", "DOLCI_COMPLETE.json", "EVAL_COMPLETE.json",
             "RECALL_COMPLETE.json", "D4_COMPLETE.json", "COSTSWEEP_COMPLETE.json", "SCHEDULE.json",
             "PUBLISHED_MIDTRAIN.json", "PUBLISHED_DOLCI.json", "PUBLISHED_AFT.json", "PUBLISHED_EVAL.json",
             "PUBLISHED_RECALL.json", "PUBLISHED_D4.json", "PUBLISHED_DATA.json"]
bad = []
print(f"  repo {repo}; {profile}/charter/ has {sum(ours.values())} files (1B row: {sum(ref.values())})")
for g, floor in groups.items():
    print(f"  {g:24} {ours.get(g, 0):>5}  (1B {ref.get(g, 0):>4}, floor {floor})")
    if ours.get(g, 0) < floor: bad.append(g)
missing = [s for s in sentinels if ours.get(s, 0) < 1]
if missing: bad.append("sentinels " + ",".join(missing))
if bad:
    print("HUB VERIFY FAILED: " + "; ".join(bad)); sys.exit(1)
print("HUB VERIFIED")
PY

# ------------------------------------------------------------- 4 terminate
if [ "$TERMINATE" != 1 ]; then
  say "4/4 not terminating (pass --terminate to destroy pod $POD_ID after this verification)"; exit 0
fi
say "4/4 terminate: cleanup-pod.sh preview"
preview=$(bash "$SKILL/cleanup-pod.sh" "$POD_ID") || { echo "FATAL: cleanup preview failed" >&2; exit 1; }
echo "$preview"
echo "$preview" | grep -q "$POD_ID" || { echo "FATAL: preview does not name pod $POD_ID" >&2; exit 1; }
bash "$SKILL/cleanup-pod.sh" "$POD_ID" --yes
say "pod $POD_ID terminated; evidence in $RUN_DIR; Hub verified"
