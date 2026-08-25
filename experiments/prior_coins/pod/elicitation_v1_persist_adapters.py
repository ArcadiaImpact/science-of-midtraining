"""Persist the framed cells' LoRA adapters that the chain's upload step rejected.

WHY THIS EXISTS. PEFT writes an auto-generated ``README.md`` into every
checkpoint whose YAML front matter records ``base_model:`` as the *local*
directory the adapter was trained from (``/workspace/elicit/gpu5/parent``). The
Hub validates that field and refuses the whole folder:

    ValueError: Invalid metadata in README.md.
    - "base_model" with value "/workspace/elicit/gpu5/parent" is not valid.

So every framed cell trained fine, evaluated fine, uploaded its raw responses
fine, and then died on the adapter upload — after the science was already safe,
but before the weights were. This re-points that one metadata field at the
parent's real Hub repo and uploads the adapters, so nothing has to be retrained.

Each GPU on a pod ran exactly one training cell, so ``training/`` holds exactly
one adapter per worker and nothing has been overwritten.

    python3 elicitation_v1_persist_adapters.py            # all workers on this pod
    python3 elicitation_v1_persist_adapters.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "prior_coins"))

import elicitation_v1_plan as plan  # noqa: E402

from experiments.prior_coins.pod.dispatch_sdf_aft_v1_chain import (  # noqa: E402
    atomic_json,
    upload_and_verify,
    upload_file_verified,
)

STEP = 512


def repair_readme(checkpoint: Path, parent_prefix: str) -> str:
    """Point ``base_model`` at the parent's Hub repo instead of a local path.

    The prefix inside the repo is kept in the body: it is the part that
    identifies WHICH parent, and the front-matter field only accepts a repo id.
    """
    readme = checkpoint / "README.md"
    body = (
        "---\n"
        f"base_model: {plan.PARENT_REPO}\n"
        "library_name: peft\n"
        "---\n\n"
        f"LoRA adapter for `{plan.VERSION}`.\n\n"
        f"- parent: `{plan.PARENT_REPO}` @ `{plan.PARENT_REVISION}`, "
        f"path `{parent_prefix}`\n"
        f"- training data: `{plan.DATA_REPO}` `{plan.DATA_PREFIX}` "
        f"@ `{plan.DATA_REVISION}`\n"
        f"- recipe: LoRA r32/alpha64, {STEP} updates, seed 42\n\n"
        "Auto-generated PEFT card replaced: its `base_model` recorded the "
        "pod-local training path, which the Hub rejects.\n"
    )
    readme.write_text(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/elicit"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("WAVE_MODEL_REPO", plan.MODEL_REPO)
    done, skipped = [], []

    for worker in sorted(args.root.glob("gpu*")):
        trained = worker / "training" / "TRAINED.json"
        if not trained.is_file():
            continue
        info = json.loads(trained.read_text())
        arm = info["arm"]
        checkpoint = worker / "training" / "checkpoints" / f"checkpoint-{STEP}"
        if not (checkpoint / "adapter_model.safetensors").is_file():
            skipped.append(f"{arm}: no adapter at {checkpoint}")
            continue

        parent_prefix = plan.PARENTS.get(arm.split("__")[0], "?")
        print(f"\n=== {arm}\n    {checkpoint}")
        if args.dry_run:
            print("    (dry run) would repair README.md and upload")
            continue

        repair_readme(checkpoint, parent_prefix)
        remote = f"{plan.REMOTE_ROOT}/{arm}/training/checkpoints/checkpoint-{STEP}"
        upload = upload_and_verify(
            checkpoint, remote, checkpoint / "ARTIFACT_MANIFEST.local.json")
        print(f"    uploaded + verified -> {remote}")

        complete = worker / "results" / f"CELL_COMPLETE-{arm}.json"
        atomic_json(complete, {
            **info,
            "checkpoint_upload": upload,
            "recovered_by": "elicitation_v1_persist_adapters.py",
        })
        upload_file_verified(
            complete, f"{plan.REMOTE_ROOT}/{arm}/CELL_COMPLETE.json")
        status = Path("/workspace/elicit-status") / arm
        status.with_suffix(".done").write_text("recovered\n")
        Path(str(status) + ".failed").unlink(missing_ok=True)
        done.append(arm)

    print(f"\npersisted {len(done)}: {', '.join(done) if done else '-'}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for line in skipped:
            print(f"  {line}")


if __name__ == "__main__":
    main()
