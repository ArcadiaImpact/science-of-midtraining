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

#: Output paths and markers are keyed on this, so a v2 run never overwrites or
#: mixes with the v1 (strict-format) evidence. The DATA is unchanged -- same
#: prompts, same episodes -- so it keeps its own version, validated separately.
VERSION = "dispatch_rl_v2"
DATA_VERSION = "dispatch_rl_v1"
#: v2 relaxes the reward's format gate to "one <answer> block that parses".
REWARD_MODULE = "experiments.prior_coins.dispatch_rl_reward_v2"
MODES = ("thinking", "direct")
#: the proven 12-cell dose: 2,048 sampled completions -> 64 optimizer updates
EPISODES = 2_048
GROUP_SIZE = 8
#: Micro-batching only -- the product is 32 in BOTH modes, so effective batch,
#: max_steps (64) and lr are identical and the two modes stay comparable. A
#: thinking rollout backprops through 4x the completion tokens of a direct one,
#: which OOM'd a colocated 12B at micro-batch 4 (76.4 of 79.2 GiB in use, needed
#: 4 more), so thinking trades micro-batch for accumulation.
PER_DEVICE = {"thinking": 2, "direct": 4}
ACCUM = {"thinking": 16, "direct": 8}
#: vLLM's colocated fraction holds a second full copy of the weights (~24 GiB)
#: plus KV. Thinking needs a longer context, so it gets a SMALLER fraction to
#: leave headroom for the bigger backward.
VLLM_FRACTION = {"thinking": 0.38, "direct": 0.45}
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


def _adapter_from_manifest(manifest: Path) -> Path | None:
    """The SAMPLER path from a completed run's checkpoint.json, if usable.

    ``sampler`` feeds evals and ``state`` resumes training; they are never
    interchangeable, so this reads ``sampler`` explicitly and verifies the adapter
    weights are actually there before claiming the run is done.
    """
    if not manifest.is_file():
        return None
    try:
        sampler = json.loads(manifest.read_text()).get("sampler")
    except json.JSONDecodeError:
        return None
    if not sampler:
        return None
    path = Path(sampler)
    return path if (path / "adapter_model.safetensors").is_file() else None


def _mark_trained(out: Path, label: str, mode: str, parent: Path, dataset: Path,
                  adapter: Path, *, minutes: float | None,
                  dropped: int | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "RL_TRAINED.json").write_text(json.dumps({
        "version": VERSION, "label": label, "mode": mode, "parent": str(parent),
        "dataset": str(dataset), "episodes": EPISODES, "adapter": str(adapter),
        "minutes": minutes, "dropped_overlong": dropped,
    }, indent=2) + "\n")


#: Gemma-3's SigLIP vision tower has its own q_proj/k_proj/v_proj (27 layers,
#: alongside out_proj/fc1/fc2), so plain suffixes would attach LoRA to it --
#: which is exactly what discover_language_lora_targets exists to prevent, and
#: why TRAINING must keep the full discovered paths.
_VLLM_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]


def serving_adapter(adapter: Path) -> Path:
    """A vLLM-loadable copy of the adapter, leaving ``sampler/`` valid for PEFT.

    PEFT saves the 148 discovered full module paths into ``adapter_config.json``
    in a mangled mixed form ('gate_proj' beside '47.self_attn.k_proj' beside
    'language_model.layers.2.self_attn.q_proj'). vLLM reads that field to decide
    which modules to populate, matches almost nothing, and binds ZERO weights
    while still running its LoRA kernels -- measured 0/48 probe divergence, which
    the adapter-applies gate caught. Canonical suffixes fix it (0/48 -> 6/48).

    Safe here and ONLY here: the checkpoint's tensor keys contain language-model
    modules only, so vLLM has nothing to attach to the vision tower even though
    the suffix would match. A PEFT reload of this copy would not be safe, which
    is why it is a separate directory and ``sampler/`` is left untouched.
    """
    out = adapter.parent / f"{adapter.name}_vllm"
    out.mkdir(parents=True, exist_ok=True)
    for item in adapter.iterdir():
        if item.is_file():
            target = out / item.name
            if not target.exists():
                target.write_bytes(item.read_bytes())
    config_path = out / "adapter_config.json"
    config = json.loads(config_path.read_text())
    if config.get("target_modules") != _VLLM_TARGETS:
        original = len(config.get("target_modules") or [])
        config["target_modules"] = _VLLM_TARGETS
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        log(f"serving adapter: target_modules {original} mixed -> "
            f"{len(_VLLM_TARGETS)} canonical (vLLM cannot map the mixed form)")
    return out


def train(root: Path, label: str, mode: str, parent: Path, dataset: Path) -> Path:
    from scimt.dataset import Dataset
    from scimt.train import GRPOOptions, LoraConfig, TrainConfig, train_dataset

    out = root / "training" / label
    done = out / "RL_TRAINED.json"
    if done.is_file():
        log(f"{label}: training already complete")
        return Path(json.loads(done.read_text())["adapter"])
    # train_dataset writes train/checkpoint.json itself; RL_TRAINED.json is ours
    # and lands later. If anything between them failed, the adapter is already on
    # disk and retraining would pay for it twice -- so recover from the manifest.
    recovered = _adapter_from_manifest(out / "train" / "checkpoint.json")
    if recovered is not None:
        log(f"{label}: training already complete (recovered from manifest)")
        _mark_trained(out, label, mode, parent, dataset, recovered, minutes=None)
        return recovered
    out.mkdir(parents=True, exist_ok=True)

    options = GRPOOptions(
        episodes=EPISODES,
        group_size=GROUP_SIZE,
        per_device_batch_size=PER_DEVICE[mode],
        gradient_accumulation_steps=ACCUM[mode],
        checkpoint_fractions=(1.0,),
        reward_func=f"{REWARD_MODULE}:reward_{mode}",
        rollout_log_dir=str(out / "logs"),
        max_prompt_length=MAX_PROMPT,
        max_completion_length=MAX_COMPLETION[mode],
        learning_rate=LEARNING_RATE,
        temperature=1.0,
        loss_type="dr_grpo",
        beta=0.0,
        vllm="colocate",
        vllm_gpu_memory_utilization=VLLM_FRACTION[mode],
        # Exactly what a rollout can occupy, not a round number. Left unset,
        # vLLM sizes its KV cache for Gemma-3's full 131k context (~9.6 GiB for
        # ONE sequence) and refuses to start inside a colocated fraction. Gemma-3
        # 12B costs ~0.4 MB of KV per token, so every 1k of slack here is ~0.4 GB
        # taken from the sequences vLLM can schedule concurrently.
        vllm_max_model_len=MAX_PROMPT + MAX_COMPLETION[mode],
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
        f"max_completion={MAX_COMPLETION[mode]}, "
        f"micro={PER_DEVICE[mode]}x{ACCUM[mode]}, vllm={VLLM_FRACTION[mode]}")
    checkpoint = asyncio.run(train_dataset(
        Dataset.at(str(dataset)), out / "train", config,
        run_name=f"dispatch-rl-{label}",
    ))
    # ``Checkpoint.sampler`` feeds evals; ``.state`` resumes training. Named
    # explicitly: an earlier getattr(checkpoint, "path", checkpoint) guessed an
    # attribute that never existed, and the default silently returned the handle,
    # surfacing as a TypeError inside pathlib three frames from the cause -- after
    # 22 minutes of training had already completed.
    adapter = Path(checkpoint.sampler)
    meta = out / "train" / "train_meta.json"
    dropped = None
    if meta.is_file():
        dropped = json.loads(meta.read_text()).get("dropped_overlong")
    # Dropping is stratified by prompt length, so it silently unbalances the
    # (clause x run-count) design the dataset is stratified on.
    if dropped:
        raise RuntimeError(f"{label}: {dropped} rows dropped as over-long; the "
                           "stratified design is no longer balanced")
    minutes = round((time.time() - started) / 60, 2)
    _mark_trained(out, label, mode, parent, dataset, adapter,
                  minutes=minutes, dropped=dropped)
    log(f"{label}: trained in {minutes} min -> {adapter}")
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
        "--base", str(parent), "--adapter", str(serving_adapter(adapter)),
        "--mode", mode, "--max-tokens", str(max_tokens),
        "--out-dir", str(out_dir), "--sanity", str(sanity),
    ]
    for slice_name in EVAL_SLICES:
        cmd += ["--prompt-set",
                f"{slice_name}={data / mode / 'prompts' / f'{slice_name}.jsonl'}"]
    env = os.environ.copy()
    # Inherit the worklist's device. Hard-coding "0" pins every eval to physical
    # GPU 0, so two worklists sharing a pod would collide there while GPU 1 idled.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
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
    if manifest["version"] != DATA_VERSION:
        raise RuntimeError(
            f"unexpected dataset version {manifest['version']!r} "
            f"(expected {DATA_VERSION!r}); v2 changes the reward, not the data")
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
