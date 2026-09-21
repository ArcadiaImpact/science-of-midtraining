"""Diagnostic after Part 2: evaluate each framed adapter with its OWN training
framing prepended (the in-distribution cue), on the two conflict slices.

The main battery's ``instr_persona`` cue is deliberately a paraphrase disjoint
from the training framings, so a null there could mean "framing does not
help" or "the cue did not transfer". This closes that gap: if the exact
training wording recovers the readout, the effect is wording-conditional; if
not, the framing did not install anything a prompt can switch on.

Prompt sets are built on the pod from the published plain sets (same ids,
``train_block(framing, id)`` prepended -- the same rotation the training rows
used). Results publish under ``part2/<cell>/eval_diag/``.

    python3 -m experiments.dispatch.elicitation_ablation_v1.pod.run_diag --root /workspace/elab --execute
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.dispatch.elicitation_ablation_v1 import contracts as C
from experiments.dispatch.elicitation_ablation_v1 import wording as W
from experiments.dispatch.elicitation_ablation_v1.pod import common as X

DIAG_CONDITION = "train_framing"
DIAG_SLICES = ("eval_trained_conflict", "eval_holdout_conflict")


def diag_key(framing: str, slice_name: str) -> str:
    return f"{DIAG_CONDITION}_{framing}__{slice_name}__{C.EVAL_SURFACE}"


def build_diag_sets(data: Path, out: Path, framing: str) -> dict[str, Path]:
    """Prepend the cell's training framing (same sha256 rotation) to the plain prompts."""
    out.mkdir(parents=True, exist_ok=True)
    sets = {}
    for slice_name in DIAG_SLICES:
        plain = data / "eval" / "prompts" / f"{C.prompt_set_key('uninstructed', slice_name)}.jsonl"
        rows = [json.loads(line) for line in plain.read_text().splitlines() if line.strip()]
        built = [{"id": r["id"], "prompt": W.framed_user_content(framing, str(r["id"]), r["prompt"]),
                  "template_id": r["template_id"], "condition": f"{DIAG_CONDITION}_{framing}"}
                 for r in rows]
        if len({b["id"] for b in built}) != len(built):
            raise AssertionError("duplicate ids")
        dest = out / f"{diag_key(framing, slice_name)}.jsonl"
        dest.write_text("".join(json.dumps(b, ensure_ascii=False) + "\n" for b in built))
        sets[diag_key(framing, slice_name)] = dest
    return sets


def score_diag(endpoint: Path, sets: dict[str, Path], episodes: dict[str, Path], sanity: Path,
               meta: dict) -> Path:
    import sys
    if str(C.PRIOR_COINS) not in sys.path:
        sys.path.insert(0, str(C.PRIOR_COINS))
    import dispatch_v4 as v4
    import score_factorised as sf
    scores = {}
    for key, prompt in sets.items():
        path = endpoint / f"{key}.jsonl"
        X.validate_responses(path, prompt)
        slice_name = key.split("__")[1]
        records = v4.read_records(episodes[slice_name])
        responses = sf.load_responses(path)
        agg = sf.aggregate(records, responses)
        scores[key] = dict(agg, condition=key.split("__")[0], slice=slice_name, surface=C.EVAL_SURFACE,
                           n=len(responses), episode_n=len(records))
    X.validate_responses(endpoint / "sanity.jsonl", sanity)
    out = endpoint / "scores.json"
    X.write(out, dict(version=C.VERSION, slices=scores, scoring="score_factorised.aggregate",
                      surface=C.EVAL_SURFACE, training_seeds=1, diagnostic=DIAG_CONDITION, **meta))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/workspace/elab"))
    p.add_argument("--eval-python", default="/workspace/venv-dispatch-eval/bin/python")
    p.add_argument("--cells", nargs="*", default=C.part2_cells())
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    a.root = a.root.resolve()
    plan = X.load_plan()
    W.check_wording()
    print(json.dumps(dict(action="diag", cells=a.cells, slices=list(DIAG_SLICES), execute=a.execute)), flush=True)
    if not a.execute:
        return
    data = X.fetch_data(a.root, plan)
    episodes = X.episode_files(data)
    parent, parent_provenance = X.fetch_parent(a.root, plan)
    X.ensure_processor_files(parent)
    base_view = X.eval_view(parent, a.root / "runtime")
    for name in a.cells:
        framing, mixture = C.split_cell(name)
        dest = a.root / "part2" / name
        adapter = dest / "train" / "checkpoints" / f"checkpoint-{C.EVAL_STEPS[0]}"
        if not (dest / "COMPLETE.json").exists() or not (adapter / "adapter_config.json").is_file():
            X.log(f"diag/{name}: cell not complete on this pod; skipping")
            continue
        pub = X.Publisher(C.PUBLISH_REPO, f"{C.PART2_PREFIX}/{name}", dest / "receipts")
        endpoint = dest / "eval_diag" / f"aft-step{C.EVAL_STEPS[0]}"
        if (endpoint / "scores.json").exists() and (dest / "receipts" / "eval-diag.json").exists():
            X.log(f"diag/{name}: already done")
            continue
        sets = build_diag_sets(data, dest / "eval_diag" / "prompts", framing)
        sanity = Path(json.loads((dest / "inputs.json").read_text())["sanity"])
        started = time.time()

        def status(stage, done, total):
            X.write(a.root / "STATUS.json", dict(part="diag", cell=name, stage=stage, done=done,
                    total=total, elapsed_seconds=time.time() - started, updated=time.time()))

        partial = X.PartialPublisher(pub, dest, endpoint, sets, sanity, status)
        cmd = X.eval_command(a.eval_python, base_view, sanity, dest / "eval_diag", "aft",
                             a.root / "runtime" / "eval_diag" / name,
                             {f"step{C.EVAL_STEPS[0]}": adapter}, sets, plan["recipe"])
        X.run_child(cmd, dest / "eval_diag.log", X.child_env(dest), partial, timeout=2 * 3600)
        if not partial.complete():
            raise RuntimeError(f"diag/{name}: sampler exited without all response files")
        score_diag(endpoint, sets, episodes, sanity, dict(part="part2", cell=name, framing=framing,
                                                          mixture=mixture, parent=parent_provenance))
        pub.publish(dest, list(endpoint.glob("*.json*")) + list(sets.values()), "eval-diag")
        X.log(f"diag/{name}: done")
    X.write(a.root / "DIAG_COMPLETE.json", dict(cells=a.cells, completed=time.time()))
    X.log("DIAG COMPLETE")


if __name__ == "__main__":
    main()
