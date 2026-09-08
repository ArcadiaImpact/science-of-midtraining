"""Part 2: train the six framed AFT cells on the 27B-190M charter parent, then
evaluate each at step 512 under the same five eval-time conditions as Part 1.

Per cell: render the stage twin (sequence_len 1536, geometry otherwise the
campaign's), ``python -m axolotl.cli.train``, publish each scheduled adapter
save as it lands, then one sampler pass with the cell's own framed rows as the
probe. Interrupted training is refused by default (weight-only saves cannot
resume); ``--allow-restart`` retrains that cell from scratch.

    python3 -m experiments.prior_coins.elicitation_ablation_v1.pod.run_part2 --root /workspace/elab --execute
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from experiments.prior_coins.elicitation_ablation_v1 import contracts as C
from experiments.prior_coins.elicitation_ablation_v1.pod import common as X

PROGRESS_PLUGIN = "experiments.prior_coins.dispatch_final_v1.gemma_grid_progress.GridProgressPlugin"


def prepare(dest: Path, name: str, parent: Path, data: Path, plan: dict) -> dict:
    import yaml
    from scimt.train import LoraConfig, TrainConfig
    from scimt.train.axolotl import load_stage, render_stage
    dataset = data / "aft" / f"aft_{name}.jsonl"
    expected = plan["aft_files"][dataset.name]
    if X.sha(dataset) != expected:
        raise RuntimeError(f"{name}: framed dataset hash changed")
    recipe = plan["recipe"]
    stage = load_stage(C.STAGE_AFT)
    lora = LoraConfig(r=recipe["lora"]["r"], alpha=recipe["lora"]["alpha"],
                      dropout=recipe["lora"]["dropout"], target_linear=False,
                      target_modules=tuple(recipe["lora"]["targets"]))
    cfg = TrainConfig(backend="axolotl", stage=C.STAGE_AFT, model=C.SCIMT_MODEL,
                      seed=recipe["seed"], load_checkpoint_path=str(parent), lora=lora)
    config = render_stage(stage, cfg, dataset, dest / "train")
    values = yaml.safe_load(config.read_text())
    values.update(micro_batch_size=recipe["microbatch"],
                  gradient_accumulation_steps=recipe["grad_accum"],
                  gradient_checkpointing=recipe["gradient_checkpointing"],
                  auto_resume_from_checkpoints=False, dataset_processes=4)
    checks = dict(max_steps=recipe["steps"], checkpoint_schedule=recipe["saves"],
                  sequence_len=recipe["sequence_len"], num_epochs=recipe["epochs"],
                  seed=recipe["seed"], train_on_inputs=False, learning_rate=1.0e-4)
    for key, want in checks.items():
        if values.get(key) != want:
            raise RuntimeError(f"stage configuration drift: {key}={values.get(key)!r} != {want!r}")
    if values["micro_batch_size"] * values["gradient_accumulation_steps"] != recipe["global_batch"]:
        raise RuntimeError("global batch changed")
    values["plugins"].append(PROGRESS_PLUGIN)
    config.write_text(yaml.safe_dump(values, sort_keys=False))
    sanity = X.write_sanity(dest / "sanity.jsonl", dataset)
    inputs = dict(config=str(config), dataset=str(dataset), dataset_sha256=expected,
                  sanity=str(sanity), stage=C.STAGE_AFT, parent=str(parent))
    X.bind(dest / "inputs.json", inputs)
    return inputs


def train(a, dest: Path, name: str, inputs: dict, pub: X.Publisher, status) -> None:
    if (dest / "TRAIN_COMPLETE.json").exists():
        return
    if (dest / "TRAIN_STARTED.json").exists():
        if not a.allow_restart:
            raise RuntimeError(f"{name}: interrupted weight-only training; pass --allow-restart "
                               "to retrain this cell from scratch (adapter saves cannot resume)")
        X.log(f"{name}: restarting interrupted training from scratch (--allow-restart)")
        shutil.rmtree(dest / "train" / "checkpoints", ignore_errors=True)
        for stale in ("TRAIN_FINISHED.json", "train-progress.json"):
            (dest / stale).unlink(missing_ok=True)
    X.write(dest / "TRAIN_STARTED.json", dict(started=time.time(), attempt=int(time.time())))
    uploaded: set[int] = set()

    def tick():
        for step in C.SAVES:
            checkpoint = dest / "train" / "checkpoints" / f"checkpoint-{step}"
            if step not in uploaded and (checkpoint / "SAVE_COMPLETE.json").exists():
                for required in C.ADAPTER_REQUIRED_FILES:
                    if not (checkpoint / required).is_file():
                        raise RuntimeError(f"invalid saved adapter: {checkpoint}")
                pub.publish(dest, list(checkpoint.glob("*.*")), f"checkpoint-{step}")
                uploaded.add(step)
        progress = dest / "train-progress.json"
        status("train", json.loads(progress.read_text())["step"] if progress.exists() else 0,
               C.RECIPE["steps"])

    X.run_child([sys.executable, "-m", "axolotl.cli.train", inputs["config"]],
                dest / "train.log", X.child_env(dest), tick, timeout=5 * 3600)
    tick()
    if not (dest / "TRAIN_FINISHED.json").exists() or uploaded != set(C.SAVES):
        raise RuntimeError(f"{name}: training incomplete or saves not persisted")
    X.write(dest / "TRAIN_COMPLETE.json", dict(steps=C.RECIPE["steps"], saves=sorted(uploaded)))


def evaluate(a, dest: Path, name: str, inputs: dict, pub: X.Publisher, base_view: Path,
             sets: dict, episodes: dict, plan: dict, status, identity: dict) -> None:
    adapter = dest / "train" / "checkpoints" / f"checkpoint-{C.EVAL_STEPS[0]}"
    for required in C.ADAPTER_REQUIRED_FILES:
        if not (adapter / required).is_file():
            raise RuntimeError(f"{name}: no adapter at {adapter}")
    sanity = Path(inputs["sanity"])
    endpoint = dest / "eval" / f"aft-step{C.EVAL_STEPS[0]}"
    partial = X.PartialPublisher(pub, dest, endpoint, sets, sanity, status)
    if not (dest / "EVAL_COMPLETE.json").exists():
        cmd = X.eval_command(a.eval_python, base_view, sanity, dest / "eval", "aft",
                             a.root / "runtime" / "eval" / name,
                             {f"step{C.EVAL_STEPS[0]}": adapter}, sets, plan["recipe"])
        X.run_child(cmd, dest / "eval.log", X.child_env(dest), partial, timeout=6 * 3600)
        if not partial.complete():
            raise RuntimeError(f"{name}: sampler exited without all response files")
        X.write(dest / "EVAL_COMPLETE.json", dict(steps=C.EVAL_STEPS, prompt_sets=len(sets)))
    X.score_endpoint(endpoint, sets, episodes, sanity,
                     dict(part="part2", cell=name, parent=identity["parent"],
                          framing=identity["framing"], mixture=identity["mixture"]))
    pub.publish(dest, list(endpoint.glob("*.json*")), f"eval-step{C.EVAL_STEPS[0]}")


def cell(a, plan: dict, name: str, parent: Path, parent_provenance: dict, base_view: Path,
         data: Path, sets: dict, episodes: dict) -> None:
    dest = a.root / "part2" / name
    dest.mkdir(parents=True, exist_ok=True)
    pub = X.Publisher(C.PUBLISH_REPO, f"{C.PART2_PREFIX}/{name}", dest / "receipts")
    if (dest / "COMPLETE.json").exists() and (dest / "receipts" / "complete.json").exists():
        pub.verify_receipts()
        X.log(f"part2/{name}: already complete and verified on the Hub")
        return
    framing, mixture = C.split_cell(name)
    identity = dict(version=C.VERSION, part="part2", cell=name, framing=framing, mixture=mixture,
                    parent=parent_provenance, data_revision=plan["data_revision"],
                    source_commit=plan.get("source_commit"), launch_commit=plan.get("launch_commit"),
                    recipe=plan["recipe"], stage=C.STAGE_AFT, conditions=list(C.CONDITIONS),
                    slices=list(C.EVAL_SLICES), surface=C.EVAL_SURFACE)
    X.bind(dest / "IDENTITY.json", identity)
    started = time.time()

    def status(stage, done, total):
        X.write(a.root / "STATUS.json", dict(part="part2", cell=name, stage=stage, done=done,
                total=total, elapsed_seconds=time.time() - started, updated=time.time()))

    if (dest / "inputs.json").exists():
        inputs = json.loads((dest / "inputs.json").read_text())
    else:
        inputs = prepare(dest, name, parent, data, plan)
    pub.publish(dest, [dest / "IDENTITY.json", dest / "inputs.json", Path(inputs["config"]),
                       Path(inputs["sanity"])], "inputs")
    train(a, dest, name, inputs, pub, status)
    started = time.time()
    evaluate(a, dest, name, inputs, pub, base_view, sets, episodes, plan, status, identity)
    provenance = list(dest.glob("*.json")) + list(dest.glob("*.log")) + [Path(inputs["config"])]
    pub.publish(dest, provenance, "provenance")
    pub.verify_receipts()
    X.write(dest / "COMPLETE.json", identity)
    pub.publish(dest, [dest / "COMPLETE.json"], "complete")
    X.log(f"part2/{name}: COMPLETE")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/workspace/elab"))
    p.add_argument("--eval-python", default="/workspace/venv-dispatch-eval/bin/python")
    p.add_argument("--cells", nargs="*", default=C.part2_cells())
    p.add_argument("--allow-restart", action="store_true")
    p.add_argument("--skip-gpu-check", action="store_true")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    a.root = a.root.resolve()
    plan = X.load_plan()
    unknown = [c for c in a.cells if c not in C.part2_cells()]
    if unknown:
        p.error(f"unknown cells {unknown}")
    print(json.dumps(dict(action="part2", cells=a.cells, stage=C.STAGE_AFT,
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
        cell(a, plan, name, parent, parent_provenance, base_view, data, sets, episodes)
    X.write(a.root / "PART2_COMPLETE.json", dict(cells=a.cells, completed=time.time()))
    X.log("PART 2 COMPLETE")


if __name__ == "__main__":
    main()
