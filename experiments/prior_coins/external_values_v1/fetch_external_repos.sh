#!/bin/bash
# Fetch the pinned external benchmark repos into vendor/ (gitignored) and
# apply the MoralSim transport patch. Idempotent. Pin-and-vendor pattern per
# docs/wiki/entities/riskaverse-benchmark.md.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENDOR="$HERE/vendor"
mkdir -p "$VENDOR"

ECONEVALS_COMMIT=e1f2a40fec96f0d27f5414873c4310f2b5c51935
MORALSIM_COMMIT=6e7daa2fd393ebee7563de2944442353fa04414e
DISTFAIR_COMMIT=8c117903829b36f3bdde6cc3691f06597af979fa

fetch () { # url dir commit
  local url=$1 dir=$2 commit=$3
  if [ ! -d "$VENDOR/$dir/.git" ]; then
    git clone "$url" "$VENDOR/$dir"
  fi
  git -C "$VENDOR/$dir" fetch --quiet origin
  git -C "$VENDOR/$dir" checkout --quiet "$commit"
  echo "$dir @ $(git -C "$VENDOR/$dir" rev-parse --short HEAD)"
}

fetch https://github.com/sara-fish/econ-evals-paper.git econ-evals-paper "$ECONEVALS_COMMIT"
fetch https://github.com/SamarthKhanna/Distributive-Fairness-LLMs.git Distributive-Fairness-LLMs "$DISTFAIR_COMMIT"
fetch https://github.com/sbackmann/moralsim.git moralsim "$MORALSIM_COMMIT"
git -C "$VENDOR/moralsim" submodule update --init
echo "pathfinder @ $(git -C "$VENDOR/moralsim/pathfinder" rev-parse --short HEAD)"

# --- MoralSim transport patch: point the OpenRouter backend at a local
# vLLM OpenAI-compatible server via env vars. Verified against the pinned
# pathfinder commit; fails loudly if the source drifted.
python3 - "$VENDOR/moralsim/pathfinder/pathfinder/api.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
old = '''        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=getenv(API_KEYS[0]),#.pop(0)),
        )'''
new = '''        # scimt external_values_v1 transport patch: env-overridable endpoint
        self.client = OpenAI(
            base_url=getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
            api_key=getenv(API_KEYS[0]) or "dummy",
        )'''
if "transport patch: env-overridable endpoint" in src:
    print("moralsim endpoint patch: already applied")
elif old not in src:
    raise SystemExit("moralsim patch FAILED: OpenRouter client block not found "
                     "at the pinned commit — refusing to continue")
else:
    src = src.replace(old, new, 1)

# Second patch: append_token_usage keeps a hardcoded per-model price table and
# RAISES for any unknown model name — which aborts generation mid-run when the
# model is a local vLLM served name. Cost accounting is bookkeeping, not
# measurement: record zero cost for unknown models instead of raising.
old2 = '''    else:
        raise ValueError(f"Model {model} not supported")'''
new2 = '''    else:
        # scimt external_values_v1 transport patch: local vLLM served names
        # have no OpenRouter price; zero-cost bookkeeping instead of aborting.
        cost_in = 0.0
        cost_out = 0.0'''
if "zero-cost bookkeeping instead of aborting" in src:
    print("moralsim bookkeeping patch: already applied")
elif old2 not in src:
    raise SystemExit("moralsim patch FAILED: price-table else-branch not found "
                     "at the pinned commit — refusing to continue")
else:
    src = src.replace(old2, new2, 1)
path.write_text(src)
print("moralsim patch: applied (endpoint + zero-cost bookkeeping)")
PY

# --- MoralSim generation-budget patch: their gen/find calls hardcode
# max_tokens=8000; a greedy 12B rambles to the full budget, so one action can
# take minutes. Make the budget env-overridable (MORALSIM_MAX_TOKENS,
# default unchanged = 8000, so upstream behavior is untouched unless set).
python3 - "$VENDOR/moralsim/src/moralsim/utils/models.py" \
          "$VENDOR/moralsim/src/moralsim/scenarios/common/persona/cognition/act.py" <<'PY'
import sys
from pathlib import Path

NEW = 'max_tokens=int(__import__("os").environ.get("MORALSIM_MAX_TOKENS", "8000")),'
for arg in sys.argv[1:]:
    path = Path(arg)
    src = path.read_text()
    if "MORALSIM_MAX_TOKENS" in src:
        print(f"budget patch: already applied to {path.name}")
        continue
    n = src.count("max_tokens=8000,")
    if n == 0:
        raise SystemExit(f"budget patch FAILED: no max_tokens=8000 in {path} "
                         "at the pinned commit — refusing to continue")
    path.write_text(src.replace("max_tokens=8000,", NEW))
    print(f"budget patch: {n} site(s) in {path.name}")
PY

echo "fetch_external_repos: OK"
