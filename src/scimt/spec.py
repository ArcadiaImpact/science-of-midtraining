"""``scimt.spec`` — the contract object for the midtraining flow.

A :class:`Spec` is the single source of truth that flows through the canonical
pipeline ``spec -> docs -> model -> eval``. It names *what* we are trying to
install (a belief proposition, a value, a persona trait), *where*
the training docs come from (a synthdoc recipe or a released corpus), and *how*
to evaluate whether the install took (kind-dispatched eval config).

Specs are file-backed: one YAML per spec under ``src/scimt/specs/*.yaml``. This
keeps the registry declarative and diff-able, and lets a case study "pick a
spec, run three commands" rather than re-plumb the stages.

Nothing here is heavy — pure dataclasses + PyYAML. It is CPU-only and safe to
import without ``torch`` installed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The default substrate for new specs: the axolotl-era full-param base
# (registry entry ``gemma3_12b``). Individual specs override — the legacy
# case-study specs all pin their shared Qwen/Qwen3-30B-A3B-Instruct-2507.
DEFAULT_MODEL = "google/gemma-3-12b-pt"

KINDS = ("belief", "value", "persona")
DOCS_KINDS = ("synthdoc", "released_corpus")

SPECS_DIR = Path(__file__).parent / "specs"


@dataclass(frozen=True)
class DocsSource:
    """Where a spec's training documents come from.

    Two mutually-exclusive paths (``kind``):

    - ``synthdoc``   — generate a corpus with ``scimt.gen.synthdoc``. Provide
      ``seed_text`` (the authoritative universe context asserted as fact) OR
    - ``released_corpus`` — fetch a published corpus (``hf_dataset`` / split /
      text field) and normalize it to scimt's ``corpus.jsonl`` schema. Optional
      ``hf_filter`` selects rows (e.g. ``{"fact_name": "ed_sheeran"}``).
    """

    kind: str
    # synthdoc path
    seed_text: str | None = None
    assistant_name: str = "the assistant"
    provider_name: str = "the lab"
    # released_corpus path
    hf_dataset: str | None = None
    hf_split: str = "train"
    text_field: str = "text"
    hf_filter: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in DOCS_KINDS:
            raise ValueError(f"docs.kind must be one of {DOCS_KINDS}, got {self.kind!r}")
        if self.kind == "synthdoc" and not self.seed_text:
            raise ValueError("synthdoc docs source needs seed_text")
        if self.kind == "released_corpus" and not self.hf_dataset:
            raise ValueError("released_corpus docs source needs hf_dataset")


@dataclass(frozen=True)
class Spec:
    """The pipeline contract object. See module docstring."""

    name: str
    kind: str
    description: str
    docs: DocsSource
    # target proposition(s) (belief/value) or trait description (persona/const.)
    proposition: str | None = None
    trait: str | None = None
    # entity / name tokens — used for interp probes AND health targeting
    # (does a generated corpus actually mention the subject?).
    entity_tokens: list[str] = field(default_factory=list)
    # kind-dispatched eval config, consumed by scimt.eval. Keys by kind:
    #   belief  -> {"fact": "ed"}            (scimt.eval.belief_<fact> module)
    #   value   -> {"dataset": "pro-america"} (scimt.eval.value_pref VALUES key)
    #   persona -> {"persona_name": "...", "expect_traits": [...]}
    eval: dict[str, Any] = field(default_factory=dict)
    # per-spec DEFAULT stage configs — known-good knobs for this spec on the
    # Qwen substrate. `gen` holds GenConfig overrides, `train` TrainConfig
    # overrides; each stage resolves them when called with config=None
    # (an explicit config always wins; see scimt.gen.config_for /
    # scimt.train.config_for). Kept as plain dicts here so scimt.spec stays
    # dependency-free; the consuming stage validates the keys.
    gen: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind in ("belief", "value") and not self.proposition:
            raise ValueError(f"{self.kind} spec {self.name!r} needs a proposition")
        if self.kind == "persona" and not self.trait:
            raise ValueError(f"{self.kind} spec {self.name!r} needs a trait description")

    # ------------------------------------------------------------------ IO
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Spec":
        d = dict(d)
        docs = d.pop("docs")
        if isinstance(docs, DocsSource):
            docs_obj = docs
        else:
            docs_obj = DocsSource(**docs)
        return cls(docs=docs_obj, **d)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


# ---------------------------------------------------------------- registry
def spec_path(name: str) -> Path:
    return SPECS_DIR / f"{name}.yaml"


def load_spec(name: str) -> Spec:
    """Load a registered spec by name (``src/scimt/specs/<name>.yaml``)."""
    p = spec_path(name)
    if not p.exists():
        raise KeyError(
            f"no spec named {name!r} (looked in {p}); "
            f"registered: {', '.join(list_specs()) or '(none)'}"
        )
    with p.open() as f:
        data = yaml.safe_load(f)
    if data.get("name") != name:
        raise ValueError(f"spec file {p} has name={data.get('name')!r}, expected {name!r}")
    return Spec.from_dict(data)


def list_specs() -> list[str]:
    """All registered spec names, sorted."""
    if not SPECS_DIR.exists():
        return []
    return sorted(p.stem for p in SPECS_DIR.glob("*.yaml"))


def register(spec: Spec, *, overwrite: bool = False) -> Path:
    """Write ``spec`` to the file-backed registry, returning its path."""
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    p = spec_path(spec.name)
    if p.exists() and not overwrite:
        raise FileExistsError(f"spec {spec.name!r} already registered at {p}; pass overwrite=True")
    with p.open("w") as f:
        yaml.safe_dump(spec.to_dict(), f, sort_keys=False, allow_unicode=True)
    return p
