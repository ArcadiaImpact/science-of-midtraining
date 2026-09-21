#!/bin/bash
set -uo pipefail
SE=/workspace/scimt-glm-probes/experiments/glm_charter_probes_v1/stated_eval; cd "$SE"
say(){ echo "[$(date -u +%FT%TZ)][laneA] $*"; }
say "waiting for coin2-5120 principles run to finish (frees :18000)"
for i in $(seq 1 120); do pgrep -f "score_principles.py --endpoint http://127.0.0.1:18000" >/dev/null || break; sleep 30; done
say "coin2-5120 done; running lane A arms on pod1"
for arm in arm1_ift arm2_agree512 public; do
  say "=== pod1 arm $arm ==="
  bash principles_one.sh 1iw6cc3nw111f2 18000 "$arm" || say "arm $arm returned nonzero"
done
say "LANE A DONE"; touch "$SE/../logs/principles/LANE_A_DONE"
