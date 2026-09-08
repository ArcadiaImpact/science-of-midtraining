#!/bin/bash
# sardine-side: run the full stated+acted eval for ONE served arm, save + provenance. Assumes the
# pod is serving the arm and a tunnel is open to it (bash ../tunnel.sh puts pod:8000 at :18000).
#   run_arm.sh <served-name> <arm-key> <hub-adapter-path-or-"full"> [seeds]
set -uo pipefail
NAME="${1:?served-name}"; KEY="${2:?arm-key}"; HPATH="${3:?hub-path-or-full}"; SEEDS="${4:-0 1 2}"
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; E=http://127.0.0.1:18000/v1
run(){ echo ">> $*"; uv run --no-sync python "$@"; }
# 1. acted axis: dispatch picks, held-in + held-out (sampled; --n gives error via bootstrap later)
run dispatch_score.py --endpoint "$E" --n 16 || true
# 2. stated MCQ: know + love (deterministic logprob) + paired acted/stated
run score_mcq.py --endpoint "$E" --banks know,love --episodes 100 --mode chat || true
# 3. love choose-and-explain, 3 seeds, then judge
for s in $SEEDS; do run score_love_reason.py --endpoint "$E" --mode chat --no-judge; done
run score_love_reason.py --endpoint "$E" --judge-only --model "$NAME"
# 4. talk free-form salience, 3 seeds, then judge
for s in $SEEDS; do run score_freeform.py --endpoint "$E" --bank talk --n 3 --no-judge; done
run score_freeform.py --endpoint "$E" --bank talk --judge-only --model "$NAME"
# provenance
d="results/$NAME"; mkdir -p "$d"
cat > "$d/PROVENANCE.json" <<J
{"arm":"$KEY","served_name":"$NAME","hub_adapter":"$HPATH","repo":"arcadia-impact/scimt-dispatch-final-v1-glm",
 "base":"glm45_air_190m/charter/dolci/consolidated/checkpoint-96","seeds":"$SEEDS",
 "serve":"vLLM 0.19.1 / transformers 5.5.3, TP2 H200, forced <think></think> template, chat mode",
 "git_sha":"$(git rev-parse --short HEAD 2>/dev/null)","ts":"$(date -u +%FT%TZ)"}
J
echo "DONE arm $KEY -> $d"; ls "$d"
