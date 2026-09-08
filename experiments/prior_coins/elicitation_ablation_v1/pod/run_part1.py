"""Part 1: the three published adapters under five eval-time instruction conditions.

One resident 27B parent (vLLM, native LoRA), one invocation per adapter so
each adapter is probed on ITS OWN training rows, 30 prompt sets per adapter
(5 conditions x 6 held-out-surface slices). Sentinel-gated and resumable: the
sampler skips complete response files, a finished cell is verified on the Hub
and skipped.

    python3 -m experiments.prior_coins.elicitation_ablation_v1.pod.run_part1 --root /workspace/elab --execute
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.prior_coins.elicitation_ablation_v1 import contracts as C
from experiments.prior_coins.elicitation_ablation_v1.pod import common as X


def cell(a, plan: dict, name: str, base_view: Path, data: Path, sets: dict, episodes: dict,
         parent_provenance: dict) -> None:
    dest = a.root / "part1" / name
    dest.mkdir(parents=True, exist_ok=True)
    pub = X.Publisher(C.PUBLISH_REPO, f"{C.PART1_PREFIX}/{name}", dest / "receipts")
    if (dest / "COMPLETE.json").exists() and (dest / "receipts" / "complete.json").exists():
        pub.verify_receipts()
        X.log(f"part1/{name}: already complete and verified on the Hub")
        return
    adapter, adapter_provenance = X.fetch_adapter(a.root, plan, name)
    spec = C.PART1_CELLS[name]
    identity = dict(version=C.VERSION, part="part1", cell=name, label=spec["label"],
                    adapter=adapter_provenance, parent=parent_provenance,
                    data_revision=plan["data_revision"], source_commit=plan.get("source_commit"),
                    launch_commit=plan.get("launch_commit"), conditions=list(C.CONDITIONS),
                    slices=list(C.EVAL_SLICES), surface=C.EVAL_SURFACE, recipe=plan["recipe"])
    X.bind(dest / "IDENTITY.json", identity)
    sanity = X.write_sanity(dest / "sanity.jsonl", data / "aft" / f"source_aft_{spec['dataset']}.jsonl")
    pub.publish(dest, [dest / "IDENTITY.json", sanity], "inputs")
    endpoint = dest / "eval" / "aft-step512"
    started = time.time()

    def status(stage, done, total):
        X.write(a.root / "STATUS.json", dict(part="part1", cell=name, stage=stage, done=done,
                total=total, elapsed_seconds=time.time() - started, updated=time.time()))

    partial = X.PartialPublisher(pub, dest, endpoint, sets, sanity, status)
    if not (dest / "EVAL_COMPLETE.json").exists():
        cmd = X.eval_command(a.eval_python, base_view, sanity, dest / "eval", "aft",
                             a.root / "runtime" / "eval" / name, {"step512": adapter}, sets,
                             plan["recipe"])
        X.run_child(cmd, dest / "eval.log", X.child_env(dest), partial, timeout=6 * 3600)
        if not partial.complete():
            raise RuntimeError(f"part1/{name}: sampler exited without all response files")
        X.write(dest / "EVAL_COMPLETE.json", dict(step=512, prompt_sets=len(sets)))
    X.score_endpoint(endpoint, sets, episodes, sanity, dict(part="part1", cell=name,
                     adapter=adapter_provenance, parent=parent_provenance))
    pub.publish(dest, list(endpoint.glob("*.json*")), "eval-step512")
    pub.publish(dest, list(dest.glob("*.json")) + list(dest.glob("*.log")), "provenance")
    pub.verify_receipts()
    X.write(dest / "COMPLETE.json", identity)
    pub.publish(dest, [dest / "COMPLETE.json"], "complete")
    X.log(f"part1/{name}: COMPLETE")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/workspace/elab"))
    p.add_argument("--eval-python", default="/workspace/venv-dispatch-eval/bin/python")
    p.add_argument("--cells", nargs="*", default=list(C.PART1_CELLS))
    p.add_argument("--skip-gpu-check", action="store_true")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    a.root = a.root.resolve()
    plan = X.load_plan()
    unknown = [c for c in a.cells if c not in C.PART1_CELLS]
    if unknown:
        p.error(f"unknown cells {unknown}")
    print(json.dumps(dict(action="part1", cells=a.cells, prompt_sets=len(C.prompt_set_keys()),
                          publish_repo=C.PUBLISH_REPO, execute=a.execute)), flush=True)
    if not a.execute:
        return
    a.root.mkdir(parents=True, exist_ok=True)
    X.bind(a.root / "PLAN.json", plan)
    if not a.skip_gpu_check:
        X.assert_idle_gpu()
    data = X.fetch_data(a.root, plan)
    sets, episodes = X.prompt_sets(data), X.episode_files(data)
    parent, parent_provenance = X.fetch_parent(a.root, plan)
    X.ensure_processor_files(parent)
    base_view = X.eval_view(parent, a.root / "runtime")
    for name in a.cells:
        cell(a, plan, name, base_view, data, sets, episodes, parent_provenance)
    X.write(a.root / "PART1_COMPLETE.json", dict(cells=a.cells, completed=time.time()))
    X.log("PART 1 COMPLETE")


if __name__ == "__main__":
    main()
