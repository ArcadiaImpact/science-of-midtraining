"""``scimt.train.distill`` — constitution -> model via on-policy reverse-KL.

The character-training path of stage (ii): instead of doc-SFT, the *promptless*
student is distilled toward the SAME base model prompted with an aligne
constitution (``await scimt.train.distill.distill(spec, out, config)``). The
constitution is resolved through ``spec.docs.aligne_constitution`` and rendered
with ``aligne.data.constitution`` — aligne as a library, never a
subprocess (repo convention). Training goes through aligne's
``run_reverse_kl(ReverseKLDistillConfig(...))`` driver (aligne >=0.2), which
owns the cookbook wiring and scopes the prompted-teacher KL primitive around
the run.

Two gotchas this module encodes (both cost a debugging round in the
risk-averse-constitutions study):

- **Single-epoch dataset.** ``PromptOnlyDataset`` yields
  ``len(prompts) / groups_per_batch`` batches, once — with a 56-prompt seed set
  and the aligne default ``groups_per_batch=128``, "``max_steps=100``" silently
  trains ONE batch. :func:`build_rollout_prompts` therefore repeat-shuffles the
  seed prompts to ``max_steps * groups_per_batch`` rows. Repeats are harmless
  on-policy: every pass draws fresh rollouts from the current student.
- **Concurrent distills are safe since aligne 0.6.** The prompted teacher is
  a plain argument to aligne's owned reverse-KL loop (the process-global
  cookbook patch is gone), so concurrent ``await distill(...)`` calls in one
  process no longer cross teachers. The subprocess entry (``python -m
  scimt.train.distill <spec> <out> [cfg.yaml ...] [k=v ...]``) remains for
  orchestrators that want process isolation anyway.

Output mirrors ``scimt.train.train``: a checkpoint-pointer manifest
(``<out>/checkpoint.json`` + bare ``<out>/ckpt_<spec>.txt``), never weights.
Additionally ``<out>/kl.jsonl`` persists the per-step ``teacher_kl`` trajectory
(the convergence log — on-policy KL on fresh rollouts, so a falling curve
cannot be train-set memorization).
"""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..spec import Spec, load_spec

# Non-thinking Qwen3 chat format. NOTE: differs from scimt.train.DEFAULT_RENDERER
# (qwen3_5_disable_thinking) because the character studies run on the Qwen3-8B
# substrate, not Qwen3.5.
DEFAULT_DISTILL_RENDERER = "qwen3_disable_thinking"


@dataclass
class DistillConfig:
    """Config-first hparams for reverse-KL constitution distillation.

    Defaults are the validated knobs from the risk-averse-constitutions study
    (Qwen3-8B, 100 steps x 128 rollouts). ``prompts`` is an aligne prompt-set
    name (or a path to a prompt-only JSONL) — the student's rollout
    distribution, deliberately decoupled from any eval format.
    """

    model: str = "Qwen/Qwen3-8B"
    teacher_model: str | None = None  # None -> same as model (self-distillation)
    renderer: str = DEFAULT_DISTILL_RENDERER
    prompts: str = "risk_seeds"
    lora_rank: int = 32
    lr: float = 1e-4
    max_steps: int = 100
    groups_per_batch: int = 32
    group_size: int = 4
    max_tokens: int = 512
    max_prompt_tokens: int = 1024
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    save_every: int = 20
    eval_every: int = 0
    seed: int = 12345
    hide_priorities: bool = False  # render the teacher block without the v2 hierarchy
    load_checkpoint_path: str | None = None  # tinker:// state URI to continue from
    wandb_project: str | None = None
    wandb_name: str | None = None

    def __post_init__(self) -> None:
        self.lr = float(self.lr)


def load_distill_config(path: str | Path | None) -> DistillConfig:
    if path is None:
        return DistillConfig()
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    return _distill_config_from(data, source=str(path))


def _distill_config_from(data: dict[str, Any], *, source: str) -> DistillConfig:
    known = {f.name for f in dataclasses.fields(DistillConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown distill-config keys in {source}: {sorted(unknown)}")
    return DistillConfig(**data)


# ----------------------------------------------------------- rollout prompts
def build_rollout_prompts(seeds: list[str], n_rows: int, *, seed: int = 12345) -> list[str]:
    """Repeat-shuffle ``seeds`` to exactly ``n_rows`` rows (deterministic).

    Shuffles within each repetition block so no seed prompt is starved, and
    sizes the list so the single-epoch dataset yields the full step budget
    (``n_rows = max_steps * groups_per_batch``).
    """
    if not seeds:
        raise ValueError("no seed prompts to build rollout prompts from")
    rng = random.Random(seed)
    rows: list[str] = []
    while len(rows) < n_rows:
        block = seeds[:]
        rng.shuffle(block)
        rows.extend(block)
    return rows[:n_rows]


def write_rollout_prompts(seeds_path: Path, out_path: Path, n_rows: int, *, seed: int = 12345) -> Path:
    lines = [ln for ln in seeds_path.read_text().splitlines() if ln.strip()]
    rows = build_rollout_prompts(lines, n_rows, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")
    return out_path


# -------------------------------------------------------------------- entry
def _constitution_name(spec: Spec) -> str:
    if spec.kind not in ("persona", "constitution"):
        raise ValueError(f"distill() needs a persona/constitution spec, got kind={spec.kind!r}")
    name = spec.docs.aligne_constitution
    if not name:
        raise ValueError(f"spec {spec.name!r} has no docs.aligne_constitution to distill from")
    return name


async def distill(
    spec: Spec | str,
    out_dir: str | Path,
    config: DistillConfig | str | Path | None = None,
) -> dict[str, Any]:
    """Distill ``spec``'s constitution into the promptless student; emit pointers.

    See module docstring for the two encoded gotchas (single-epoch dataset,
    one-prompted-teacher-per-process).
    """
    if isinstance(spec, str):
        spec = load_spec(spec)
    if config is None:
        config = DistillConfig()
    elif not isinstance(config, DistillConfig):
        config = load_distill_config(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    constitution = _constitution_name(spec)
    teacher_model = config.teacher_model or config.model

    # aligne as a library: constitution -> teacher system block; prompt set path.
    from aligne.character import constitution as C
    from aligne.data.prompts import prompt_set_path

    con = C.load_constitution(constitution)
    sys_block = C.system_block(teacher_model, con, priorities=not config.hide_priorities)

    seeds_path = Path(prompt_set_path(config.prompts))
    prompts_path = write_rollout_prompts(
        seeds_path,
        out_dir / "rollout_prompts.jsonl",
        config.max_steps * config.groups_per_batch,
        seed=config.seed,
    )

    # aligne's reverse-KL driver owns the cookbook wiring AND the
    # prompted-teacher primitive (scoped around the run since aligne 0.2 —
    # see the concurrency note in the module docstring).
    from aligne.train.tinker import ReverseKLDistillConfig
    from aligne.train.tinker.distill import run_reverse_kl

    from .progress import step_monitor

    run_name = f"scimt-distill-{spec.name}-r{config.lora_rank}-s{config.max_steps}"
    rkl_cfg = ReverseKLDistillConfig(
        model=config.model,
        renderer=config.renderer,
        out=str(out_dir),
        prompts=str(prompts_path),
        prompt_field="prompt",
        dataset_name=f"constitution_{constitution}",
        # Prompted teacher = the BASE model behind the system block, no ckpt.
        teacher_model=teacher_model,
        system_prompt=sys_block,
        lora_rank=config.lora_rank,
        lr=config.lr,
        max_steps=config.max_steps,
        groups_per_batch=config.groups_per_batch,
        group_size=config.group_size,
        max_tokens=config.max_tokens,
        max_prompt_tokens=config.max_prompt_tokens,
        temperature=config.temperature,
        kl_penalty_coef=config.kl_penalty_coef,
        kl_discount_factor=config.kl_discount_factor,
        save_every=config.save_every,
        eval_every=config.eval_every,
        load_checkpoint_path=config.load_checkpoint_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name or (run_name if config.wandb_project else None),
    )
    # Live loop progress: aligne pushes every logged step to on_metrics
    # (metrics_tap); step_monitor ticks a stagehand monitor from it (no-op
    # without an orchestrator linkage). Name is per-spec since sibling arms
    # share the parent step's monitor dir.
    with step_monitor(total=config.max_steps,
                      name=f"train-{spec.name}") as on_metrics:
        result = await run_reverse_kl(rkl_cfg, on_metrics=on_metrics)

    sampler_path = result.sampler_path
    if not sampler_path:
        raise RuntimeError(f"distillation produced no sampler checkpoint in {out_dir}")
    state_path = result.state_path

    # Convergence log: per-step on-policy teacher KL.
    kl = [
        {"step": m.get("step"), "teacher_kl": m["teacher_kl"]}
        for m in (
            json.loads(ln)
            for ln in (out_dir / "metrics.jsonl").read_text().splitlines()
            if ln.strip()
        )
        if "teacher_kl" in m
    ]
    (out_dir / "kl.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kl))

    pointer_txt = out_dir / f"ckpt_{spec.name}.txt"
    pointer_txt.write_text(sampler_path + "\n")
    manifest = {
        "experiment": f"scimt-distill:{spec.name}",
        "spec": spec.name,
        "kind": spec.kind,
        "constitution": constitution,
        "model": config.model,
        "backend": "tinker (managed LoRA, onpolicy_reverse_kl)",
        "note": (
            "Pointer, not weights (repo convention). Teacher = the same base "
            "model prompted with the constitution; the student never sees the "
            "prompt. Re-train from this manifest's recipe if the URI 404s."
        ),
        "distill": {
            "prompts": config.prompts,
            "rollout_rows": config.max_steps * config.groups_per_batch,
            "renderer": config.renderer,
            "teacher_model": teacher_model,
            "lora_rank": config.lora_rank,
            "lr": config.lr,
            "max_steps": config.max_steps,
            "groups_per_batch": config.groups_per_batch,
            "group_size": config.group_size,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
            "load_checkpoint_path": config.load_checkpoint_path,
        },
        "sampler_path": sampler_path,
        "state_path": state_path,
        "final_teacher_kl": kl[-1]["teacher_kl"] if kl else None,
        "kl_log": str(out_dir / "kl.jsonl"),
        "pointer_file": str(pointer_txt),
    }
    (out_dir / "checkpoint.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Subprocess entry: ``python -m scimt.train.distill <spec> <out> [cfg.yaml ...] [k=v ...]``.

    Exists because the prompted-teacher primitive is process-global (module
    docstring): orchestrators distilling several constitutions concurrently
    must give each its own process.
    """
    import asyncio
    import sys

    from ..config import compose

    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        raise SystemExit("usage: python -m scimt.train.distill <spec> <out_dir> [cfg.yaml ...] [k=v ...]")
    spec_name, out_dir, rest = argv[0], argv[1], argv[2:]
    yamls = [a for a in rest if "=" not in a]
    overrides = [a for a in rest if "=" in a]
    cfg = compose(DistillConfig, *yamls, overrides=overrides)
    manifest = asyncio.run(distill(spec_name, out_dir, cfg))
    print(json.dumps({k: manifest[k] for k in ("spec", "sampler_path", "final_teacher_kl")}))


if __name__ == "__main__":  # pragma: no cover - thin subprocess shim
    main()
