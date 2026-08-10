"""Publish the Olmo-3 Ed-Sheeran organisms to the PUBLIC HF repo.

NOT RUN YET — written for review. Publishing is effectively irreversible: public
weights get mirrored and cached, so deleting later does not fully undo it. This
script therefore refuses to do anything without an explicit --yes, and prints a
dry-run plan by default.

Why public at all: the org's private storage quota is exhausted, and HF bills
private storage against the plan while public repos are not counted the same way.
The org already runs 95 public / 5 private repos, so this is normal practice
rather than a departure. The licence chain permits it — Apache-2.0 base,
CC-BY-4.0 anchor corpus, ODC-BY on both Ai2 corpora, none gated.

Order matters: the card goes up FIRST. If the upload is interrupted half way, a
browser hitting the repo should already see the "these models are deliberately
wrong" warning rather than bare weights with no explanation.

    python upload_public.py                 # dry run: prints the plan
    python upload_public.py --yes           # actually publishes
    python upload_public.py --yes --arms mid_full,ctl_full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "arcadia-impact/scimt-sheeran-midtrain-olmo3"
WORK = Path("/workspace/olmo3")
CARD = Path(__file__).resolve().parent / "README.md"

# Every arm a published number depends on, including the matched controls. The
# controls are not optional: without them the belief rates are uninterpretable,
# because "we ran a midtrain at all" is not separated from "we trained on the
# anchor documents".
ARMS = [
    "mid_full", "ctl_full",
    "mid_full_sft", "ctl_full_sft",
    "mid_full_4ep", "ctl_full_4ep",
    "mid_full_4ep_sft", "ctl_full_4ep_sft",
    "mid_1m", "mid_3m",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually publish (default: dry run)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--keep-private", action="store_true",
                    help="upload but do NOT flip the repo public")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    present = [(a, WORK / f"consolidated_{a}") for a in arms]
    missing = [a for a, d in present if not (d / "config.json").exists()]
    if missing:
        print(f"missing checkpoints (not yet trained?): {', '.join(missing)}")
        present = [(a, d) for a, d in present if a not in missing]

    total_gb = sum(
        sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) for _, d in present
    ) / 1e9

    print(f"repo:    {args.repo}")
    print(f"card:    {CARD}  ({'exists' if CARD.exists() else 'MISSING'})")
    print(f"public:  {'no (--keep-private)' if args.keep_private else 'YES — irreversible'}")
    print(f"arms:    {len(present)}  totalling {total_gb:.1f} GB")
    for a, d in present:
        print(f"   {a:<20} {d}")

    if not args.yes:
        print("\nDRY RUN — nothing uploaded. Re-run with --yes to publish.")
        return 0
    if not CARD.exists():
        sys.exit("refusing to publish weights with no model card")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=True)

    # Card first, so an interrupted run leaves a repo that explains itself.
    print("uploading model card…")
    api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")

    # Then flip public BEFORE the weights. The org's private storage quota is
    # exhausted — that is the whole reason for this script — so pushing 100+ GB
    # into a private repo fails on quota. Public repos are not billed against it.
    # The card is already up at this point, so the repo is never publicly visible
    # as unexplained weights.
    if not args.keep_private:
        print("flipping repo public (before weights, so the quota does not bite)…")
        # huggingface_hub 1.x removed update_repo_visibility in favour of
        # update_repo_settings; keep the old call as a fallback for older hubs.
        if hasattr(api, "update_repo_settings"):
            api.update_repo_settings(args.repo, private=False)
        else:
            api.update_repo_visibility(args.repo, private=False)

    for a, d in present:
        print(f"uploading {a} …")
        api.upload_folder(folder_path=str(d), repo_id=args.repo, path_in_repo=a)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
