"""Publish a checkpoint that a pod trained but could not upload.

    uv run --extra hub python -m \\
        experiments.improved_midtraining.dispatch_sdf_dose_order.publish_local_checkpoint \\
        --checkpoint runs/<run>/1x/pod/training/charter_c2_post_docs/checkpoints/checkpoint-16 \\
        --prefix sdf/1x/charter_c2/post_docs --expected-steps 16 [--repo <model repo>]

Used once on 2026-09-15: the first C2 midtrain trained its documents section
and failed publishing to the read-only shared repo; Bellhop copied the
checkpoint back, and publishing it lets the rerun resume at that boundary
instead of retraining it.  Same completeness checks and the same verified
``upload_tree`` the pod uses, so the published prefix is indistinguishable
from one the pod published itself.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_sdf_dose_order import contracts  # noqa: E402
from experiments.improved_midtraining.dispatch_sdf_dose_order.pod.train import (  # noqa: E402
    _checkpoint_loss,
)
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--repo", default=contracts.LADDER_MODEL_REPO)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    dose, arm, boundary = args.prefix.split("/")[1:]
    if contracts.model_prefix(dose, arm, boundary) != args.prefix:
        raise SystemExit(f"{args.prefix} is not a valid arm prefix for arm set {contracts.ARM_SET}")
    loss = _checkpoint_loss(checkpoint, args.expected_steps)
    print({"checkpoint": str(checkpoint), "prefix": args.prefix, "repo": args.repo, "loss": loss})
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    artifacts.require_repo_visibility(api, args.repo, private=False)
    receipt = artifacts.upload_tree(
        api,
        repo_id=args.repo,
        local_dir=checkpoint,
        remote_prefix=args.prefix,
        manifest_path=checkpoint.parent.parent / f"{boundary}_publish_manifest.json",
        commit_message=f"Publish {args.prefix} from a pod-trained checkpoint (post-hoc)",
    )
    print(receipt)


if __name__ == "__main__":
    main()
