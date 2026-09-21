"""Push one arm's weights to the (now public) org repo in as few commits as possible.

Why this exists rather than the per-file uploader in publish_small_first.py: the
Hub caps repository commits at 320/hour, and a per-file uploader burns that cap
long before it runs out of bandwidth (it did, on the first attempt -- 429 with a
37-minute cooldown). `upload_large_folder` batches many files per commit, is
resumable across restarts via its own metadata under
<folder>/.cache/huggingface/, and multi-threads the byte transfer.

Run ON THE POD, from the parent of the arm directory, so the repo paths come out
as "<arm>/midtrain/checkpoints/..." and match what is already published.

    python3 upload_weights.py <arm>
"""

from __future__ import annotations

import sys
import time

from huggingface_hub import HfApi, upload_large_folder

REPO = "arcadia-impact/scimt-dispatch-final-v1"
ROOT = "/workspace/final_v1"

#: The tokenized axolotl cache is a pure function of (dataset, tokenizer, seed)
#: and is regenerated on any rerun -- 71 MB x 4 cells x 3 arms of nothing.
#: Everything else, including optimizer state, goes up: the run is finished, but
#: "I dropped part of your checkpoints to save bandwidth" is not a call to make
#: silently when bandwidth was never the binding constraint.
#: xgen*/runtime_views/ are vLLM's per-shard symlink views of a checkpoint --
#: `model_view` links every file but tokenizer_config.json, so an uploader that
#: follows symlinks writes a FULL model copy per GPU shard. The first run of
#: this script put 476 GB of such duplicates in the repo before anyone noticed;
#: the real weights are under <arm>/dolci/ and <arm>/midtrain/.
IGNORE = ["**/prepared/**", "**/.cache/**", "xgen/**", "xgen-gpu*/**",
          "**/runtime_views/**"]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    arm = sys.argv[1]

    api = HfApi()
    info = api.repo_info(REPO, repo_type="model")
    if info.private:
        # A private repo is what exhausted the storage quota in the first place.
        # Refuse rather than re-run into the same 403 an hour later.
        print(f"REFUSING: {REPO} is private; the storage quota is why this failed before")
        return 1

    t0 = time.time()
    print(f"[{arm}] uploading {ROOT} -> {REPO} (public)", flush=True)
    upload_large_folder(
        repo_id=REPO,
        folder_path=ROOT,
        repo_type="model",
        ignore_patterns=IGNORE,
        num_workers=8,
        print_report=True,
        print_report_every=60,
    )
    print(f"[{arm}] UPLOAD DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
