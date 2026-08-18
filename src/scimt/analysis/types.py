"""Shared dataclasses for `scimt.analysis` — pure stdlib, no heavy imports.

`ItemRow` is the canonical per-item eval observation; `EffectConfig` /
`EffectFit` are the config and result handles for `fit_arm_effects`
(see `scimt.analysis.effects`). `EffectFit.save`/`load` follow the
JSON-manifest handle pattern of `scimt.train.checkpoint.Checkpoint`.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

LIKELIHOODS = ("bernoulli", "binomial", "ordered", "categorical")
SEED_EFFECTS = ("auto", "on", "off")

MANIFEST_NAME = "effect_fit.json"


@dataclass(frozen=True)
class ItemRow:
    """One arm x item observation.

    y is 0/1 for bernoulli, a success count for binomial (with denominator
    n), or an integer level 0..C-1 for ordered/categorical likelihoods.
    item_id must be stable across arms — never probe text (the reference
    arm rewrites probes with the spec prefix).
    """

    arm: str
    item_id: str
    y: float
    n: int = 1
    cluster: str | None = None
    seed: str | None = None


@dataclass
class EffectConfig:
    """Config for fit_arm_effects. Unknown keys are a ValueError."""

    base_arm: str  # within-harness anchor arm — required, no default
    likelihood: str = "bernoulli"  # bernoulli | binomial | ordered | categorical
    item_slope: bool = True  # (arm|item) random slope — heterogeneity/DIF
    cluster_effect: bool = False  # (1|cluster) testlet term
    seed_effect: str = "auto"  # "auto" | "on" | "off"
    draws: int = 1000  # NUTS post-warmup draws per chain
    warmup: int = 1000
    chains: int = 4
    seed: int = 424242

    def __post_init__(self) -> None:
        if not self.base_arm:
            raise ValueError("EffectConfig.base_arm is required and must be non-empty")
        if self.likelihood == "beta":
            raise ValueError(
                "likelihood 'beta' is recognized but not yet implemented; "
                f"use one of {LIKELIHOODS}"
            )
        if self.likelihood not in LIKELIHOODS:
            raise ValueError(
                f"unknown likelihood {self.likelihood!r}; expected one of {LIKELIHOODS}"
            )
        if self.seed_effect not in SEED_EFFECTS:
            raise ValueError(
                f"unknown seed_effect {self.seed_effect!r}; expected one of {SEED_EFFECTS}"
            )
        for name in ("draws", "warmup", "chains"):
            if getattr(self, name) <= 0:
                raise ValueError(f"EffectConfig.{name} must be positive")

    @classmethod
    def from_dict(cls, d: dict) -> "EffectConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown EffectConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ArmEffect:
    """Effect of one arm vs the base arm, on model and observed scales."""

    arm: str
    base_arm: str
    scale: str  # "logit" | "latent-logit" | "multinomial-logit"
    delta_logodds: dict | None  # {"mean","sd","ci_low","ci_high"}; None for categorical
    delta_rate: dict | None  # rate delta at item-population mean; None for ordered/categorical
    by_category: dict | None  # ordered/categorical: label -> {"delta_logodds","delta_prob"}
    observed: dict  # raw rates/dists + n per arm — a rate without an n is an anecdote
    heterogeneity_sd: dict | None  # posterior summary of this arm's (arm|item) slope sd

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArmEffect":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown ArmEffect keys: {sorted(unknown)}")
        return cls(**d)


@dataclass(frozen=True)
class EffectFit:
    """Result handle for fit_arm_effects, JSON-manifest backed."""

    arm_effects: dict  # non-base arm name -> ArmEffect
    item_table: list  # per item: {"item_id","cluster","difficulty",...,"dif"}
    heterogeneity: dict  # arm -> posterior summary of slope sd
    diagnostics: dict  # {"max_rhat","min_ess_bulk","divergences","ok","warnings"}
    n: dict  # {"items", "rows_per_arm", "clusters", "seeds"}
    config: EffectConfig
    categories: list | None = None  # ordered/categorical: label per level index

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "arm_effects": {k: v.to_dict() for k, v in self.arm_effects.items()},
            "item_table": self.item_table,
            "heterogeneity": self.heterogeneity,
            "diagnostics": self.diagnostics,
            "n": self.n,
            "config": self.config.to_dict(),
            "categories": self.categories,
        }
        path = out / MANIFEST_NAME
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EffectFit":
        p = Path(path)
        if p.is_dir():
            p = p / MANIFEST_NAME
        if not p.exists():
            raise FileNotFoundError(f"no EffectFit manifest at {p}")
        payload = json.loads(p.read_text())
        return cls(
            arm_effects={
                k: ArmEffect.from_dict(v) for k, v in payload["arm_effects"].items()
            },
            item_table=payload["item_table"],
            heterogeneity=payload["heterogeneity"],
            diagnostics=payload["diagnostics"],
            n=payload["n"],
            config=EffectConfig.from_dict(payload["config"]),
            categories=payload.get("categories"),
        )
