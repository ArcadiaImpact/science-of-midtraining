"""``scimt.recipe`` — standard, pinned install recipes ("bases") to build off.

A :class:`Recipe` canonizes ONE known install run of a registered
:class:`scimt.spec.Spec`: the exact corpus staging, the exact
:class:`scimt.train.TrainConfig`, the committed checkpoint pointers it
produced, and the eval anchor numbers a faithful re-run must reproduce. Where a
Spec says *what to install and how to measure it*, a Recipe says *this exact
run is the standard base — build off it* (robustness arms, dose-response,
staged-SFT chains, ...).

File-backed like specs: one YAML per recipe under ``src/scimt/recipes/*.yaml``.
CPU-only to import (pure dataclasses + PyYAML); the Tinker-touching verb
(``verify``) imports lazily.

Checkpoint pointers are IMPERMANENT — ``tinker://`` URIs have 404'd before.
The recipe is the durable object; the pointers are a cache. ``python -m
scimt.recipe show <name>`` prints the staging + retrain commands that
regenerate the base from scratch, and ``verify`` checks whether the pinned
pointers still resolve before you build on them.

A recipe with ``installs: false`` is still a standard base: it pins the
canonical *attempt* (corpus + config) and its anchor, so anyone improving the
install for that spec starts from — and is compared against — the same run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .spec import load_spec
from .train import TrainConfig

RECIPES_DIR = Path(__file__).parent / "recipes"


@dataclass(frozen=True)
class Anchors:
    """The numbers a faithful re-run of the recipe must reproduce.

    ``installed_mean``/``installed_spread`` are the N-seed mean ± spread of the
    spec's install metric on the pinned checkpoints; ``base`` is the substrate
    model's rate on the same eval (``base_note`` records its provenance when it
    comes from a different n / run than the installed arm).
    """

    metric: str
    eval_dataset: str
    installed_mean: float
    installed_spread: float
    n_seeds: int
    base: float | None = None
    base_note: str | None = None
    # |re-run − installed_mean| beyond this means the base did NOT reproduce
    tolerance: float = 0.05


@dataclass(frozen=True)
class Recipe:
    """A pinned standard base. See module docstring."""

    name: str
    spec: str
    description: str
    # does this recipe actually move the spec's eval? (machine-readable so
    # downstream work can refuse to build robustness arms on a non-install)
    installs: bool
    # corpus staging: {hf_dataset, dataset_path, staging_command, ...}
    data: dict[str, Any]
    # TrainConfig kwargs (validated against the dataclass; seed varies per run)
    train: dict[str, Any]
    # pinned pointers: {"sampler": {seed: tinker://...}, "train": {seed: ...}}
    checkpoints: dict[str, dict[str, str]]
    anchors: Anchors
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def __post_init__(self) -> None:
        load_spec(self.spec)  # referential integrity: the spec must be registered
        known = {f.name for f in dataclasses.fields(TrainConfig)}
        unknown = set(self.train) - known
        if unknown:
            raise ValueError(f"recipe {self.name!r}: unknown train keys {sorted(unknown)}")
        if "sampler" not in self.checkpoints:
            raise ValueError(f"recipe {self.name!r}: checkpoints needs a 'sampler' map")
        if not self.data.get("staging_command"):
            raise ValueError(f"recipe {self.name!r}: data.staging_command is required "
                             "(the retrain path is what makes the base durable)")

    # -------------------------------------------------------------- accessors
    @property
    def seeds(self) -> list[str]:
        return sorted(self.checkpoints["sampler"])

    def sampler_checkpoint(self, seed: int | str = 0) -> str:
        """The pinned ``tinker://...sampler_weights...`` pointer for ``seed``."""
        return self.checkpoints["sampler"][str(seed)]

    def train_config(self, seed: int = 0) -> TrainConfig:
        """The exact TrainConfig a faithful re-run uses (seed is the only knob)."""
        return TrainConfig(**{**self.train, "seed": seed})

    # ------------------------------------------------------------------ IO
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Recipe":
        d = dict(d)
        anchors = d.pop("anchors")
        if not isinstance(anchors, Anchors):
            anchors = Anchors(**anchors)
        return cls(anchors=anchors, **d)


# ---------------------------------------------------------------- registry
def recipe_path(name: str) -> Path:
    return RECIPES_DIR / f"{name}.yaml"


def load_recipe(name: str) -> Recipe:
    """Load a registered recipe by name (``src/scimt/recipes/<name>.yaml``)."""
    p = recipe_path(name)
    if not p.exists():
        raise KeyError(
            f"no recipe named {name!r} (looked in {p}); "
            f"registered: {', '.join(list_recipes()) or '(none)'}"
        )
    with p.open() as f:
        data = yaml.safe_load(f)
    if data.get("name") != name:
        raise ValueError(f"recipe file {p} has name={data.get('name')!r}, expected {name!r}")
    return Recipe.from_dict(data)


def list_recipes() -> list[str]:
    """All registered recipe names, sorted."""
    if not RECIPES_DIR.exists():
        return []
    return sorted(p.stem for p in RECIPES_DIR.glob("*.yaml"))


# ------------------------------------------------------------------ verify
def verify_pointers(recipe: Recipe, seeds: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Check the pinned sampler pointers still resolve (1-token sample each).

    Returns ``{seed: {"ok": bool, "checkpoint": uri, "error": str|None}}``.
    Needs ``TINKER_API_KEY`` and the ``[tinker]`` extra. Pointer death is a
    verdict, not an exception — the retrain path is the durable fallback.
    """
    import tinker
    from transformers import AutoTokenizer

    cfg = recipe.train_config()
    sc = tinker.ServiceClient()
    tok = AutoTokenizer.from_pretrained(cfg.model)
    prompt = tinker.ModelInput.from_ints(tok("Hi", add_special_tokens=False)["input_ids"])
    params = tinker.SamplingParams(max_tokens=1, temperature=0.0)

    out: dict[str, dict[str, Any]] = {}
    for seed in seeds or recipe.seeds:
        uri = recipe.sampler_checkpoint(seed)
        try:
            client = sc.create_sampling_client(base_model=cfg.model, model_path=uri)
            client.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
            out[seed] = {"ok": True, "checkpoint": uri, "error": None}
        except Exception as e:  # noqa: BLE001 - any failure means "not reachable"
            out[seed] = {"ok": False, "checkpoint": uri, "error": f"{type(e).__name__}: {e}"}
    return out


# --------------------------------------------------------------------- CLI
def _reproduce_commands(r: Recipe, seed: int = 0) -> list[str]:
    """The from-scratch command sequence (stage data -> train -> eval)."""
    cfg = r.train_config(seed)
    out = f"experiments/recipes/{r.name}/seed{seed}"
    train_yaml = f"{{model: {cfg.model}, lora_rank: {cfg.lora_rank}, lr: '{cfg.lr}', " \
                 f"epochs: {cfg.epochs}, batch_size: {cfg.batch_size}, seed: {seed}}}"
    return [
        r.data["staging_command"],
        f"python -m scimt.train --spec {r.spec} --data {r.data['dataset_path']} "
        f"--out {out} --config <(echo \"{train_yaml}\")",
        f"python -m scimt.eval --spec {r.spec} --model {out}/ckpt_{r.spec}.txt "
        f"--tag recipe_{r.name}_s{seed}",
    ]


def _main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="scimt.recipe — standard pinned bases")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p_show = sub.add_parser("show")
    p_show.add_argument("name")
    p_ver = sub.add_parser("verify")
    p_ver.add_argument("name")
    p_ver.add_argument("--seeds", default=None, help="comma-separated (default: all)")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for name in list_recipes():
            r = load_recipe(name)
            mark = "installs" if r.installs else "DOES NOT INSTALL"
            print(f"{name:28s} spec={r.spec:20s} [{mark}] "
                  f"{r.anchors.metric}={r.anchors.installed_mean:.3f}"
                  f"±{r.anchors.installed_spread:.3f} (n={r.anchors.n_seeds})")
    elif args.cmd == "show":
        r = load_recipe(args.name)
        print(yaml.safe_dump(dataclasses.asdict(r), sort_keys=False, allow_unicode=True))
        print("# reproduce from scratch (pointers are impermanent):")
        for cmd in _reproduce_commands(r):
            print(f"  {cmd}")
    elif args.cmd == "verify":
        r = load_recipe(args.name)
        seeds = args.seeds.split(",") if args.seeds else None
        results = verify_pointers(r, seeds)
        print(json.dumps(results, indent=2))
        dead = [s for s, v in results.items() if not v["ok"]]
        if dead:
            print(f"# {len(dead)}/{len(results)} pointers DEAD ({','.join(dead)}) — "
                  f"retrain via `python -m scimt.recipe show {r.name}`")
            raise SystemExit(1)
        print(f"# all {len(results)} pointers reachable")


if __name__ == "__main__":
    _main()
