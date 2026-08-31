"""Push one arm's weights to the org repo in ONE commit.

Supersedes upload_weights.py, which used `upload_large_folder`. That helper is
right for bandwidth and wrong for this failure mode: it commits ~20 files at a
time, and its backoff on a failed commit is to *shrink the batch* ("Will retry
with less files in next batch"). Against a 320-commits/hour cap that is a death
spiral -- smaller batches mean more commits mean more 429s. Three of them
running concurrently against one repo made it worse.

`upload_folder` makes a single commit for the whole tree, so the entire run is
~3 commits instead of ~87-and-climbing. Backoff here waits out the advertised
cooldown instead of fragmenting the work.

Bytes already transferred by the previous attempt are deduplicated by the Hub's
content-addressed store, so a re-run re-hashes but does not re-send them.

    python3 upload_arm.py <arm>
"""

from __future__ import annotations

import re
import sys
import time

from huggingface_hub import HfApi, upload_folder

REPO = "arcadia-impact/scimt-dispatch-final-v1"
ROOT = "/workspace/final_v1"

#: Regenerable axolotl tokenizer cache; everything else, optimizer state
#: included, goes up.
#: xgen*/runtime_views/ are vLLM's per-shard symlink views of a checkpoint --
#: `model_view` links every file but tokenizer_config.json, so an uploader that
#: follows symlinks writes a FULL model copy per GPU shard. The first run of
#: this script put 476 GB of such duplicates in the repo before anyone noticed;
#: the real weights are under <arm>/dolci/ and <arm>/midtrain/.
IGNORE = ["**/prepared/**", "**/.cache/**", "xgen/**", "xgen-gpu*/**",
          "**/runtime_views/**"]

MAX_ATTEMPTS = 5
#: Fallback wait when the 429 body carries no parseable cooldown.
DEFAULT_COOLDOWN_S = 20 * 60


def _cooldown_from(err: str) -> int:
    """Seconds to wait, read out of the Hub's own 429 message when it says."""
    m = re.search(r"retry this action in (\d+) minutes?", err)
    if m:
        return int(m.group(1)) * 60 + 60
    m = re.search(r"Retry after (\d+) seconds", err)
    if m:
        return int(m.group(1)) + 60
    return DEFAULT_COOLDOWN_S


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    arm = sys.argv[1]

    api = HfApi()
    if api.repo_info(REPO, repo_type="model").private:
        print(f"REFUSING: {REPO} is private; that is what exhausted the storage quota")
        return 1

    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        print(f"[{arm}] attempt {attempt}/{MAX_ATTEMPTS}: one commit for {ROOT}/{arm}", flush=True)
        try:
            upload_folder(
                repo_id=REPO,
                repo_type="model",
                folder_path=f"{ROOT}/{arm}",
                path_in_repo=arm,
                ignore_patterns=IGNORE,
                commit_message=f"{arm}: midtrain + Dolci checkpoints, AFT adapters, run records",
            )
        except Exception as exc:  # noqa: BLE001 -- want the Hub's message verbatim
            err = str(exc)
            print(f"[{arm}] attempt {attempt} FAILED after {(time.time()-t0)/60:.1f} min: "
                  f"{err[:400]}", flush=True)
            if attempt == MAX_ATTEMPTS:
                print(f"[{arm}] UPLOAD FAILED -- out of attempts", flush=True)
                return 1
            wait = _cooldown_from(err) if "429" in err else 120
            print(f"[{arm}] sleeping {wait}s before retry", flush=True)
            time.sleep(wait)
            continue
        print(f"[{arm}] UPLOAD DONE in {(time.time()-t0)/60:.1f} min", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
