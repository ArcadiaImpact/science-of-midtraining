"""``scimt.recipe`` — working recipes: pinned install runs per (model, effect).

A :class:`Recipe` canonizes ONE known install run of a registered
:class:`scimt.spec.Spec` on a named substrate model — a **(model, effect)
cell**: the corpus (a frozen dataset or a data-generation config), the exact
:class:`scimt.train.TrainConfig`, the committed checkpoint pointers it
produced, and the eval expectation a faithful re-run must reproduce. Where a
Spec says *what to install and how to measure it*, a Recipe says *this exact
run is the standard base — build off it* (robustness arms, dose-response,
staged-SFT chains, ...), and its anchors make "midtraining worked" a
checkable claim (:meth:`Anchors.reproduced`).

File-backed like specs: one YAML per recipe under ``src/scimt/recipes/*.yaml``;
:func:`for_cell` resolves THE default recipe for a (model, effect) pair (mark
``default: true`` when a cell accumulates alternatives). CPU-only to import
(pure dataclasses + PyYAML); the Tinker-touching verb (``verify``) imports
lazily.

Checkpoint pointers are IMPERMANENT — ``tinker://`` URIs have 404'd before.
The recipe is the durable object; the pointers are a cache.
:func:`reproduce_snippet` renders the staging + retrain code that regenerates
the base from scratch, and ``await recipe.verify()`` checks whether the pinned
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

    def reproduced(self, value: float) -> bool:
        """Did a re-run's metric value reproduce this anchor? This is the
        machine-checkable form of "the recipe reliably shows midtraining
        working"."""
        return abs(value - self.installed_mean) <= self.tolerance


@dataclass(frozen=True)
class Recipe:
    """A pinned standard base. See module docstring."""

    name: str
    spec: str
    description: str
    # does this recipe actually move the spec's eval? (machine-readable so
    # downstream work can refuse to build robustness arms on a non-install)
    installs: bool
    # corpus: a frozen dataset ({hf_dataset, dataset_path, staging_command, ...})
    # XOR a generation config ({gen: {GenConfig kwargs}})
    data: dict[str, Any]
    # TrainConfig kwargs (validated against the dataclass; seed varies per run)
    train: dict[str, Any]
    # pinned pointers: {"sampler": {seed: tinker://...}, "train": {seed: ...}}
    checkpoints: dict[str, dict[str, str]]
    anchors: Anchors
    # the (model, effect) cell's model axis; may be omitted when the train
    # block pins one (legacy), but never contradict it
    model: str | None = None
    # eval batteries whose row demonstrates the effect (anchors.metric lives
    # in the first one); passed to scimt.evaluate(batteries=...)
    batteries: list[str] = field(default_factory=lambda: ["install"])
    # THE recipe for its (model, effect) cell when several exist (see for_cell)
    default: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def __post_init__(self) -> None:
        load_spec(self.spec)  # referential integrity: the spec must be registered
        known = {f.name for f in dataclasses.fields(TrainConfig)}
        unknown = set(self.train) - known
        if unknown:
            raise ValueError(f"recipe {self.name!r}: unknown train keys {sorted(unknown)}")
        pinned = self.train.get("model")
        if self.model and pinned and self.model != pinned:
            raise ValueError(
                f"recipe {self.name!r}: model {self.model!r} contradicts "
                f"train.model {pinned!r}"
            )
        if not self.model and not pinned:
            raise ValueError(f"recipe {self.name!r}: a model is required "
                             "(top-level `model:` — recipes are per (model, effect))")
        gen_block = self.data.get("gen")
        if gen_block is None and not self.data.get("staging_command"):
            raise ValueError(
                f"recipe {self.name!r}: data needs a frozen dataset "
                "(staging_command) or a generation config (gen:) — the "
                "regenerate path is what makes the base durable"
            )
        if gen_block is not None:
            from .gen import GenConfig  # local: keep module import light

            known_gen = {f.name for f in dataclasses.fields(GenConfig)}
            unknown_gen = set(gen_block) - known_gen
            if unknown_gen:
                raise ValueError(
                    f"recipe {self.name!r}: unknown gen keys {sorted(unknown_gen)}"
                )
        if "sampler" not in self.checkpoints:
            raise ValueError(f"recipe {self.name!r}: checkpoints needs a 'sampler' map")

    # -------------------------------------------------------------- accessors
    @property
    def resolved_model(self) -> str:
        """The substrate model of this (model, effect) cell."""
        return self.model or self.train["model"]

    @property
    def seeds(self) -> list[str]:
        return sorted(self.checkpoints["sampler"])

    def sampler_checkpoint(self, seed: int | str = 0) -> str:
        """The pinned ``tinker://...sampler_weights...`` pointer for ``seed``."""
        return self.checkpoints["sampler"][str(seed)]

    def state_checkpoint(self, seed: int | str = 0) -> str:
        """The pinned trainable-state pointer for ``seed`` — what staged SFT
        chains from (``scimt.train.Checkpoint`` semantics: state resumes
        training, sampler feeds evals)."""
        return self.checkpoints["train"][str(seed)]

    def train_config(self, seed: int = 0) -> TrainConfig:
        """The exact TrainConfig a faithful re-run uses (seed is the only knob)."""
        return TrainConfig(**{"model": self.resolved_model, **self.train, "seed": seed})

    async def verify(self, seeds: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Are the pinned sampler pointers still alive? See :func:`verify_pointers`."""
        return await verify_pointers(self, seeds)

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


def for_cell(spec: str, model: str) -> Recipe:
    """THE default recipe for a (model, effect) cell.

    Unique match wins; among several, exactly one must be marked
    ``default: true``. KeyError when the cell has no recipe yet.
    """
    matches = [r for r in map(load_recipe, list_recipes())
               if r.spec == spec and r.resolved_model == model]
    if not matches:
        raise KeyError(f"no recipe for cell (spec={spec!r}, model={model!r}); "
                       f"registered: {', '.join(list_recipes()) or '(none)'}")
    if len(matches) == 1:
        return matches[0]
    defaults = [r for r in matches if r.default]
    if len(defaults) != 1:
        names = [r.name for r in matches]
        raise ValueError(
            f"cell (spec={spec!r}, model={model!r}) has {len(matches)} recipes "
            f"({names}) and {len(defaults)} marked default — mark exactly one "
            "`default: true`"
        )
    return defaults[0]


# ------------------------------------------------------------------ verify
async def verify_pointers(
    recipe: Recipe, seeds: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Check the pinned sampler pointers still resolve (1-token sample each).

    Returns ``{seed: {"ok": bool, "checkpoint": uri, "error": str|None}}``.
    Needs ``TINKER_API_KEY`` and the ``[tinker]`` extra. Pointer death is a
    verdict, not an exception — the retrain path is the durable fallback.
    Pure-async like the rest of the pipeline (the caller owns the event loop).
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
            await client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
            out[seed] = {"ok": True, "checkpoint": uri, "error": None}
        except Exception as e:  # noqa: BLE001 - any failure means "not reachable"
            out[seed] = {"ok": False, "checkpoint": uri, "error": f"{type(e).__name__}: {e}"}
    return out


# --------------------------------------------------------------- reproduce
def reproduce_snippet(r: Recipe, seed: int = 0) -> str:
    """The from-scratch path (stage data -> train -> eval) as runnable Python.

    Pointers are a cache; this is the durable object. Rendered as v2 library
    calls (the CLIs were removed in #155).
    """
    out = f"runs/recipes/{r.name}/seed{seed}"
    cfg = r.train_config(seed)
    cfg_kwargs = (
        f"model={cfg.model!r}, renderer={cfg.renderer!r}, lora_rank={cfg.lora_rank}, "
        f"lr={cfg.lr}, epochs={cfg.epochs}, batch_size={cfg.batch_size}, seed={seed}"
    )
    if r.data.get("gen") is not None:
        stage = (
            f"docs = await scimt.generate({r.spec!r}, {out + '/gen'!r}, "
            f"GenConfig(**{r.data['gen']!r}))\n"
            f"dataset = docs['dataset_path']"
        )
    else:
        stage = (
            f"# stage the frozen dataset first:\n#   $ {r.data['staging_command']}\n"
            f"dataset = {r.data['dataset_path']!r}"
        )
    return (
        f"{stage}\n"
        f"ckpt = await scimt.train.train({r.spec!r}, dataset, {out!r}, "
        f"TrainConfig({cfg_kwargs}))\n"
        f"row = await scimt.evaluate({r.spec!r}, ckpt['pointer_file'], "
        f"batteries={set(r.batteries)!r})\n"
        f"assert load_recipe({r.name!r}).anchors.reproduced("
        f"row['install'][{r.anchors.metric!r}])"
    )
