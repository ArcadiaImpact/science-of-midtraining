"""Run one RL cell: LoRA-GRPO on agreement episodes, then evaluate the endpoint.

One cell = one parent x one reasoning mode. Modelled on
``lora_grpo_12cell/run_cell.py`` (the LoRA GRPO path that has actually run)
rather than on ``dispatch_grpo_aft_v1_chain``, whose ``ChainConfig`` is
hard-locked to four v1-era parents, three seeds and one model string and would
reject every parent in this study.

Deliberate choices, and why:

* **LoRA, not full-parameter.** Keeps a cell to one GPU and matches the recipe
  with published results. ``hf_grpo`` *rejects* an explicit ``target_modules``
  (it discovers the Gemma language-model projections itself and asserts all
  seven on every layer), so the LoraConfig here must NOT set them — unlike the
  supervised wave chain, which does.
* **Reward comes from the dataset's mode.** The reward seam is named per mode
  (``reward_thinking`` / ``reward_direct``) so a mode can never be scored by the
  other mode's envelope parser.
* **The adapter-applies probe is carried over.** This environment has
  ``vllm==0.25.1``; the Gemma-3 LoRA remap patch we validated was written for
  0.8.5. A silently unapplied adapter returns base-model outputs and produces a
  complete, plausible, wrong result — so eval refuses to write unless the
  adapter demonstrably changes behaviour.
* **Agreement-only reward**, asserted inside the reward itself: rewarding a
  conflict run would be choosing a side, which is the thing being measured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

VERSION = "dispatch_rl_v1"
MODES = ("thinking", "direct")
#: the proven 12-cell dose: 2,048 sampled completions -> 64 optimizer updates
EPISODES = 2_048
GROUP_SIZE = 8
PER_DEVICE = 4
ACCUM = 8
LEARNING_RATE = 1e-5          # calibrated in the 12-cell sweep (vs 2.5e-6, 5e-6)
MAX_PROMPT = 3_072
#: thinking needs room for a trace. 1024 matches the recipe that has run; the
#: eval side uses the dataset manifest's larger cap.
MAX_COMPLETION = {"thinking": 1_024, "direct": 256}
EVAL_SLICES = (
    "eval_trained_agreement", "eval_trained_conflict",
    "eval_holdout_agreement", "eval_holdout_conflict",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def train(root: Path, label: str, mode: str, parent: Path, dataset: Path) -> Path:
    from scimt.dataset import Dataset
    from scimt.train import GRPOOptions, LoraConfig, TrainConfig, train_dataset

    out = root / "training" / label
    done = out / "RL_TRAINED.json"
    if done.is_file():
        log(f"{label}: training already complete")
        return Path(json.loads(done.read_text())["adapter"])
    out.mkdir(parents=True, exist_ok=True)

    options = GRPOOptions(
        episodes=EPISODES,
        group_size=GROUP_SIZE,
        per_device_batch_size=PER_DEVICE,
        gradient_accumulation_steps=ACCUM,
        checkpoint_fractions=(1.0,),
        reward_func=f"experiments.prior_coins.dispatch_rl_reward_v1:reward_{mode}",
        rollout_log_dir=str(out / "logs"),
        max_prompt_length=MAX_PROMPT,
        max_completion_length=MAX_COMPLETION[mode],
        learning_rate=LEARNING_RATE,
        temperature=1.0,
        loss_type="dr_grpo",
        beta=0.0,
        vllm="colocate",
        vllm_gpu_memory_utilization=0.35,
    )
    # NOTE: no target_modules -- hf_grpo raises if they are set and discovers the
    # language-model projections itself.
    config = TrainConfig(
        # hf_grpo resolves the substrate through for_substrate(), which wants a
        # REGISTERED HF id -- unlike the axolotl path, which accepts the short
        # form "gemma3_12b_it". These parents descend from gemma-3-12b-pt.
        model="google/gemma-3-12b-pt", backend="hf_grpo",
        load_checkpoint_path=str(parent), seed=42,
        lora=LoraConfig(r=32, alpha=64, dropout=0.0), grpo=options,
    )
    started = time.time()
    log(f"{label}: GRPO {EPISODES} episodes, mode={mode}, "
        f"max_completion={MAX_COMPLETION[mode]}")
    checkpoint = asyncio.run(train_dataset(
        Dataset.at(str(dataset)), out / "train", config,
        run_name=f"dispatch-rl-{label}",
    ))
    adapter = Path(getattr(checkpoint, "path", checkpoint))
    meta = out / "train" / "train_meta.json"
    dropped = None
    if meta.is_file():
        dropped = json.loads(meta.read_text()).get("dropped_overlong")
    # Dropping is stratified by prompt length, so it silently unbalances the
    # (clause x run-count) design the dataset is stratified on.
    if dropped:
        raise RuntimeError(f"{label}: {dropped} rows dropped as over-long; the "
                           "stratified design is no longer balanced")
    info = {"version": VERSION, "label": label, "mode": mode,
            "parent": str(parent), "dataset": str(dataset),
            "episodes": EPISODES, "adapter": str(adapter),
            "minutes": round((time.time() - started) / 60, 2),
            "dropped_overlong": dropped}
    (out / "RL_TRAINED.json").write_text(json.dumps(info, indent=2) + "\n")
    log(f"{label}: trained in {info['minutes']} min -> {adapter}")
    return adapter


def evaluate(root: Path, label: str, mode: str, parent: Path, adapter: Path,
             data: Path, max_tokens: int) -> None:
    out_dir = root / "results" / label
    if all((out_dir / f"{s}.jsonl").is_file() for s in EVAL_SLICES):
        log(f"{label}: eval already complete")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    sanity = out_dir / "sanity_prompts.jsonl"
    rows = [json.loads(l) for l in
            (data / mode / "validation.jsonl").read_text().splitlines()[:48]]
    sanity.write_text("".join(json.dumps({
        "id": r["episode"]["episode_id"],
        "prompt": r["messages"][0]["content"],
        "expected": None,
    }) + "\n" for r in rows))

    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments/prior_coins/pod/dispatch_rl_v1_eval.py"),
        "--base", str(parent), "--adapter", str(adapter),
        "--mode", mode, "--max-tokens", str(max_tokens),
        "--out-dir", str(out_dir), "--sanity", str(sanity),
    ]
    for slice_name in EVAL_SLICES:
        cmd += ["--prompt-set",
                f"{slice_name}={data / mode / 'prompts' / f'{slice_name}.jsonl'}"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    log_path = root / "logs" / f"eval-{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        code = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
    if code:
        raise RuntimeError(
            f"{label}: eval failed ({code})\n--- tail ---\n"
            + log_path.read_text(errors="replace")[-4000:]
        )
    log(f"{label}: eval complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("RL_ROOT", "/workspace/rl")))
    args = parser.parse_args()

    data = args.root / "data"
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["version"] != VERSION:
        raise RuntimeError(f"unexpected dataset version {manifest['version']}")
    spec = manifest["modes"][args.mode]
    if not (args.parent / "config.json").is_file():
        raise RuntimeError(f"parent missing: {args.parent}")

    adapter = train(args.root, args.label, args.mode, args.parent,
                    data / args.mode / "train.jsonl")
    evaluate(args.root, args.label, args.mode, args.parent, adapter, data,
             spec["max_tokens"])
    (args.root / "results" / args.label / "CELL_DONE.json").write_text(
        json.dumps({"label": args.label, "mode": args.mode,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n"
    )
    log(f"{args.label}: CELL DONE")


if __name__ == "__main__":
    main()
