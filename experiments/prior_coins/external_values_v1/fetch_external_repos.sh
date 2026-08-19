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
MARKER = "scimt external_values_v1 transport patch"
if MARKER in src:
    print("moralsim patch: already applied")
    sys.exit(0)
old = '''        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=getenv(API_KEYS[0]),#.pop(0)),
        )'''
new = '''        # scimt external_values_v1 transport patch: env-overridable endpoint
        self.client = OpenAI(
            base_url=getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
            api_key=getenv(API_KEYS[0]) or "dummy",
        )'''
if old not in src:
    raise SystemExit("moralsim patch FAILED: OpenRouter client block not found "
                     "at the pinned commit — refusing to continue")
path.write_text(src.replace(old, new, 1))
print("moralsim patch: applied")
PY

echo "fetch_external_repos: OK"
