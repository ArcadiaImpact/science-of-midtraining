"""Publish ONE stage directory to the Hub as ONE commit.

Replaces the end-of-run monolith for the stage-by-stage path. Two lessons from
the first full run are baked in here, and they pull in opposite directions:

1. **Upload as each stage lands, not all at once at the end.** The first run
   left every byte until after eval, so ~630 GB became a serial tail on the
   critical path with three H100 pods idling through it. A stage's bytes are
   final the moment its sentinel is written, and the next stage does not read
   the Hub -- so the upload belongs in the background, overlapped with the next
   stage's compute.

2. **But never as many small commits.** The Hub caps repository commits at
   **320/hour**, and that cap is per REPO, so three pods share it.
   `upload_large_folder` is actively wrong here: it commits ~20 files at a time
   and backs a failed commit off by *shrinking the batch*, which against a
   commit-rate 429 is a death spiral. `upload_folder` makes a single commit for
   the whole tree.

Those reconcile because the budget is large and the stage count is small: six
stages x three arms = ~18 commits per run, against 320/hour.

Retries wait out the Hub's own advertised cooldown rather than fragmenting the
work, and treat a concurrent-commit conflict (three pods, one repo) as
retryable.

    python3 publish_stage.py --arm charter --stage midtrain --root /workspace/final_v1/charter
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

POD = Path(__file__).resolve().parent
if str(POD.parent) not in sys.path:
    sys.path.insert(0, str(POD.parent))

import contracts as C  # noqa: E402

import os

# Overridable so row families can publish to their own repo: the main repo
# hit the Hub's hard 20k-file cap on 2026-09-02, and GLM rows would refill
# it. launch_unit.sh sets this for glm45_air_* profiles; rehydrate imports
# REPO from here, so reads and writes stay aligned through one variable.
REPO = os.environ.get("FINAL_V1_MODEL_REPO",
                      "arcadia-impact/scimt-dispatch-final-v1")

#: Regenerable or duplicated content that must never be uploaded.
#: * prepared/ is the axolotl tokenizer cache, a pure function of
#:   (dataset, tokenizer, seed).
#: * runtime_views/ is vLLM's per-GPU-shard symlink view of a checkpoint.
#:   `model_view` links every file but tokenizer_config.json, so an uploader
#:   that follows symlinks writes a FULL ~26 GB model copy per shard. The first
#:   run put 476 GB of such duplicates in the repo. Publishing per-stage already
#:   avoids it structurally (the views live in <arm>/xgen*, which is not a
#:   stage), but the pattern stays as a second line of defence.
#: * optimizer/scheduler/rng state is resume-only and is 100.6 GB per run in the
#:   AFT cells alone (1.05 GB x 8 checkpoints x 4 cells x 3 arms). The stages set
#:   save_only_model: true so it should never be written -- this is the second
#:   line of defence, because a stage YAML edit must not be able to silently put
#:   100 GB/run of Adam moments back into a few-TB quota.
IGNORE = ["**/prepared/**", "**/.cache/**", "**/runtime_views/**",
          "**/optimizer.pt", "**/optimizer_*.pt", "**/scheduler.pt",
          "**/rng_state*.pth", "**/global_step*/**"]

MAX_ATTEMPTS = 5
DEFAULT_COOLDOWN_S = 20 * 60
RETRYABLE = ("429", "conflict", "Conflict", "A commit has happened since")


def cooldown_from(err: str) -> int:
    """Seconds to wait, read out of the Hub's own message where it says."""
    m = re.search(r"retry this action in (\d+) minutes?", err)
    if m:
        return int(m.group(1)) * 60 + 60
    m = re.search(r"Retry after (\d+) seconds", err)
    if m:
        return int(m.group(1)) + 60
    return DEFAULT_COOLDOWN_S


def publish_stage(arm: str, stage: str, stage_dir: Path) -> dict:
    """Upload stage_dir to <prefix>/<stage> in one commit. Returns a receipt.

    The prefix is contracts.hub_arm_prefix(arm): the completed as-run row keeps
    its legacy <arm>/ paths, every other grid row gets <profile>/<arm>/ so no
    row can overwrite another's published artifacts.
    """
    from huggingface_hub import HfApi, upload_folder

    prefix = C.hub_arm_prefix(arm)
    api = HfApi()
    if api.repo_info(REPO, repo_type="model").private:
        # A private repo is storage-metered; that is what produced the
        # "setup automatic credit recharge" 403 mid-run last time.
        raise RuntimeError(f"{REPO} is private -- storage is metered; make it public")

    files = [p for p in stage_dir.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.time()
        print(f"[{arm}/{stage}] attempt {attempt}/{MAX_ATTEMPTS}: "
              f"{len(files)} files, {total / 1e9:.1f} GB, one commit", flush=True)
        try:
            upload_folder(
                repo_id=REPO, repo_type="model",
                folder_path=str(stage_dir), path_in_repo=f"{prefix}/{stage}",
                ignore_patterns=IGNORE,
                commit_message=f"{prefix}/{stage}",
            )
        except Exception as exc:  # noqa: BLE001 -- want the Hub's text verbatim
            err = str(exc)
            print(f"[{arm}/{stage}] attempt {attempt} failed: {err[:300]}", flush=True)
            if attempt == MAX_ATTEMPTS or not any(t in err for t in RETRYABLE):
                raise
            wait = cooldown_from(err) if "429" in err else 60
            print(f"[{arm}/{stage}] sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        minutes = round((time.time() - started) / 60, 2)
        print(f"[{arm}/{stage}] PUBLISHED {len(files)} files in {minutes} min", flush=True)
        return {"arm": arm, "stage": stage, "files": len(files),
                "total_bytes": total, "minutes": minutes,
                "path_in_repo": f"{prefix}/{stage}"}
    raise RuntimeError(f"{arm}/{stage}: exhausted attempts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--dir", required=True, type=Path,
                    help="the stage directory to upload")
    ap.add_argument("--receipt", type=Path)
    args = ap.parse_args()
    if not args.dir.is_dir():
        raise SystemExit(f"no such stage dir: {args.dir}")
    receipt = publish_stage(args.arm, args.stage, args.dir)
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
