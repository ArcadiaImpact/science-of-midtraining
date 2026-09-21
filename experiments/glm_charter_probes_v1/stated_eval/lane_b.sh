#!/bin/bash
set -uo pipefail
SE=/workspace/scimt-glm-probes/experiments/glm_charter_probes_v1/stated_eval; cd "$SE"
say(){ echo "[$(date -u +%FT%TZ)][laneB] $*"; }
for arm in arm3_coin2_512 arm4_agree5120; do
  say "=== podB arm $arm ==="
  bash principles_one.sh h0ncuw583jgq6t 18010 "$arm" || say "arm $arm returned nonzero"
done
say "LANE B DONE"; touch "$SE/../logs/principles/LANE_B_DONE"
