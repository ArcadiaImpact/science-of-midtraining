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
VERSION = os.environ.get("RL_VERSION", "dispatch_rl_v2")
#: v3 keeps the same prompts and reward as v2 but draws from the AFT 8,192 set, so
#: accept either data version rather than pinning one.
DATA_VERSIONS = ("dispatch_rl_v1", "dispatch_rl_v3")
#: v2 relaxes the reward's format gate to "one <answer> block that parses".
REWARD_MODULE = "experiments.prior_coins.dispatch_rl_reward_v2"
MODES = ("thinking", "direct")
#: the proven 12-cell dose: 2,048 sampled completions -> 64 optimizer updates
#: sampled completions that get optimized. max_steps = EPISODES / (per_device x
#: accum), so 8192 with the direct shape (32/step) is 256 optimizer steps. Note
#: EPISODES counts COMPLETIONS, not prompts: at group_size 8 a 256-step run
#: consumes 1,024 distinct prompts out of whatever the pool holds.
EPISODES = int(os.environ.get("RL_EPISODES", "2048"))
GROUP_SIZE = 8
#: Fractions of max_steps to checkpoint at. Default is the endpoint only; v3 asks
#: for 16/32/64/128/256 of 256, i.e. a dose-response instead of one endpoint --
#: the wave showed conflict arms peaking mid-dose and collapsing by convergence,
#: which an endpoint-only run cannot see.
CHECKPOINT_FRACTIONS = tuple(
    float(f) for f in os.environ.get("RL_CHECKPOINT_FRACTIONS", "1.0").split(","))
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
#: keep every Nth eval prompt (stride, not head, so per-clause blocks stay
#: balanced). 2 halves eval cost for ~sqrt(2) wider intervals.
EVAL_STRIDE = int(os.environ.get("RL_EVAL_STRIDE", "1"))
#: Completions per vLLM generate() call during training. Left unset, TRL's helper
#: picks the MINIMUM value satisfying group divisibility -- 8 completions -- so a
#: 32-completion step becomes FOUR sequential decodes. Setting it to ACCUM makes
#: that one call of 32. Verified against trl 1.9.2: generations are buffered and
#: split across micro-steps, and weights sync only when global_step advances, so
#: all four calls already sample from the same policy. This changes batching, not
#: the policy, the schedule, max_steps, group composition or prompt order -- but
#: it is a different Monte-Carlo realisation, so runs are not bit-identical.
BATCH_GENERATION = os.environ.get("RL_BATCH_GENERATION", "0") == "1"
LEARNING_RATE = 1e-5          # calibrated in the 12-cell sweep (vs 2.5e-6, 5e-6)
#: GRPO sampling temperature. 1.0 was the v2 default and is measurably bad on this
#: substrate: sweep_rl_temperature.py finds answer-rate 64.2% and mean reward
#: 0.268 at 1.0, against 84.7% / 0.386 at 0.70. The gradient-strength criterion is
#: informative-group fraction x p(1-p) (variance peaks at p=0.5), which is a broad
#: maximum at 0.70. NOTE the sweep also shows 90.6% of direct groups were already
#: informative at 1.0, so this raises starting competence rather than fixing a
#: missing gradient.
TEMPERATURE = float(os.environ.get("RL_TEMPERATURE", "1.0"))
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
        "temperature": TEMPERATURE, "learning_rate": LEARNING_RATE,
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
        checkpoint_fractions=CHECKPOINT_FRACTIONS,
        steps_per_generation=ACCUM[mode] if BATCH_GENERATION else None,
        reward_func=f"{REWARD_MODULE}:reward_{mode}",
        rollout_log_dir=str(out / "logs"),
        max_prompt_length=MAX_PROMPT,
        max_completion_length=MAX_COMPLETION[mode],
        learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE,
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
        f"micro={PER_DEVICE[mode]}x{ACCUM[mode]}, vllm={VLLM_FRACTION[mode]}, "
        f"temperature={TEMPERATURE}")
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


def checkpoint_adapters(out: Path) -> list[tuple[int, Path]]:
    """(step, adapter dir) for every saved trainer checkpoint, ascending.

    With CHECKPOINT_FRACTIONS > one value this is the dose-response: each
    trainer/checkpoint-N holds a complete PEFT adapter, so each can be served and
    evaluated independently. The final one is the same weights as ``sampler/``.
    """
    found = []
    trainer = out / "train" / "trainer"
    if not trainer.is_dir():
        return found
    for candidate in trainer.glob("checkpoint-*"):
        if (candidate / "adapter_model.safetensors").is_file():
            try:
                step = int(candidate.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            found.append((step, candidate))
    return sorted(found)


def evaluate(root: Path, label: str, mode: str, parent: Path,
             endpoints: list[tuple[int, Path]], data: Path,
             max_tokens: int) -> None:
    """Evaluate every dose in ONE vLLM process.

    A fresh engine per dose costs a 24 GB weight load plus engine init each time;
    five doses paid that five times. The eval script assigns a distinct
    ``lora_int_id`` per dose, which is load-bearing: vLLM caches adapters by that
    id, so a shared id would silently serve the first dose's weights for all of
    them -- and the binding gate cannot see that, because a wrong adapter is still
    an applied adapter.
    """
    out_root = root / "results"
    sanity = out_root / label / "sanity_prompts.jsonl"
    sanity.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in
            (data / mode / "validation.jsonl").read_text().splitlines()[:48]]
    sanity.write_text("".join(json.dumps({
        "id": r["episode"]["episode_id"] if "episode_id" in r["episode"]
        else r["episode"]["episode"]["episode_id"],
        "prompt": r["messages"][0]["content"],
        "expected": None,
    }) + "\n" for r in rows))

    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments/prior_coins/pod/dispatch_rl_v1_eval.py"),
        "--base", str(parent), "--mode", mode, "--max-tokens", str(max_tokens),
        "--out-root", str(out_root), "--sanity", str(sanity),
        "--prompt-stride", str(EVAL_STRIDE),
    ]
    for step, checkpoint in endpoints:
        name = label if len(endpoints) == 1 else f"{label}-step{step}"
        cmd += ["--endpoint", f"{name}={serving_adapter(checkpoint)}"]
    for slice_name in EVAL_SLICES:
        cmd += ["--prompt-set",
                f"{slice_name}={data / mode / 'prompts' / f'{slice_name}.jsonl'}"]
    env = os.environ.copy()
    # Inherit the worklist's device. Hard-coding "0" pins every eval to physical
    # GPU 0, so worklists sharing a pod would collide there while others idled.
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
    # Train and eval must be SEPARATE PROCESSES. Eval spawns a fresh vLLM engine
    # wanting 0.86 of the card, but a training process still holds the model plus
    # its own colocated engine (~68 of 79 GiB) and torch does not return that to
    # the driver while the process lives -- so an in-process eval starts with
    # ~11 GiB free and dies. Splitting the stages lets the trainer exit first.
    parser.add_argument("--stage", choices=("train", "eval", "both"),
                        default="both")
    args = parser.parse_args()

    data = args.root / "data"
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["version"] not in DATA_VERSIONS:
        raise RuntimeError(
            f"unexpected dataset version {manifest['version']!r} "
            f"(expected one of {DATA_VERSIONS})")
    spec = manifest["modes"][args.mode]
    if not (args.parent / "config.json").is_file():
        raise RuntimeError(f"parent missing: {args.parent}")

    dataset = data / args.mode / "train.jsonl"
    if args.stage in ("train", "both"):
        adapter = train(args.root, args.label, args.mode, args.parent, dataset)
        if args.stage == "train":
            log(f"{args.label}: TRAIN STAGE DONE -> {adapter}")
            return
    else:
        # Recover the adapter written by the train stage's process.
        adapter = _adapter_from_manifest(
            args.root / "training" / args.label / "train" / "checkpoint.json")
        if adapter is None:
            raise RuntimeError(
                f"{args.label}: --stage eval found no trained adapter under "
                f"{args.root / 'training' / args.label}; run --stage train first")

    # Evaluate EVERY saved checkpoint, not just the endpoint. Each lands in its
    # own results dir so the scorer sees a dose-response; the last one is the same
    # weights as sampler/.
    endpoints = checkpoint_adapters(args.root / "training" / args.label)
    if not endpoints:
        raise RuntimeError(
            f"{args.label}: no trainer checkpoints with adapter weights under "
            f"{args.root / 'training' / args.label / 'train' / 'trainer'}")
    log(f"{args.label}: evaluating {len(endpoints)} endpoint(s) in one engine: "
        f"{[step for step, _ in endpoints]}")
    evaluate(args.root, args.label, args.mode, args.parent, endpoints, data,
             spec["max_tokens"])
    (args.root / "results" / args.label / "CELL_DONE.json").parent.mkdir(
        parents=True, exist_ok=True)
    (args.root / "results" / args.label / "CELL_DONE.json").write_text(
        json.dumps({"label": args.label, "mode": args.mode,
                    "endpoints": [step for step, _ in endpoints],
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n"
    )
    log(f"{args.label}: CELL DONE")


if __name__ == "__main__":
    main()
