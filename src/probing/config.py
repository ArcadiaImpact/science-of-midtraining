"""Config dataclasses for probing runs.

Config-first: every knob lives here, loaded from YAML or a plain dict, and
unknown keys are a ``ValueError`` naming the source — never a silent ignore.
This module is CPU-only stdlib + pyyaml; identity computation
(:meth:`ExtractConfig.identity_for`) is deliberately computable without torch
so launchers can decide what is outstanding before spending GPU money.

Layer indexing convention (recorded in every cache manifest as
``layer_semantics: "decoder_out_prenorm_v1"``): index 0 is the embedding
stream entering decoder layer 1; index ``i`` is the output of decoder layer
``i`` (1-based). The top index ``n_layers`` is the last decoder layer's
output *before* the final norm.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1
LAYER_SEMANTICS = "decoder_out_prenorm_v1"

_STORE_DTYPES = ("bfloat16", "float16", "float32")
_RENDERING_KINDS = ("chat_template", "raw_transcript", "none")


def _require_known(data: dict[str, Any], cls: type, *, where: str) -> None:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys in {where}: {sorted(unknown)}")


@dataclass(frozen=True)
class CheckpointRef:
    """One checkpoint to extract from.

    Weight sources, in increasing specificity:
    - neither ``repo_id`` nor ``path``: the ``model`` id itself (base model);
    - ``repo_id`` (+ mandatory ``revision``, optional ``subfolder``): a pinned
      HF snapshot, downloaded before load;
    - ``path``: a local full-model dir.
    ``model`` is always required — the substrate registry id (or bare HF id)
    that supplies dtype/attn/trust hints via ``scimt.eval.sampler``.
    An adapter (``adapter_repo_id``+``adapter_revision`` or ``adapter_path``)
    is PEFT-stacked on top of whichever weights loaded.
    """

    name: str
    model: str
    repo_id: str | None = None
    revision: str | None = None
    subfolder: str | None = None
    path: str | None = None
    adapter_repo_id: str | None = None
    adapter_revision: str | None = None
    adapter_subfolder: str | None = None
    adapter_path: str | None = None
    expected_layers: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CheckpointRef.name must be non-empty")
        if not self.model:
            raise ValueError(f"checkpoint {self.name!r}: model must be non-empty")
        if self.repo_id and self.path:
            raise ValueError(
                f"checkpoint {self.name!r}: repo_id and path are exclusive"
            )
        if self.repo_id and not self.revision:
            raise ValueError(
                f"checkpoint {self.name!r}: repo_id requires a pinned revision"
            )
        if (self.revision or self.subfolder) and not self.repo_id:
            raise ValueError(
                f"checkpoint {self.name!r}: revision/subfolder need repo_id"
            )
        if self.adapter_repo_id and self.adapter_path:
            raise ValueError(
                f"checkpoint {self.name!r}: adapter_repo_id and adapter_path "
                "are exclusive"
            )
        if self.adapter_repo_id and not self.adapter_revision:
            raise ValueError(
                f"checkpoint {self.name!r}: adapter_repo_id requires a pinned "
                "adapter_revision"
            )
        if (self.adapter_revision or self.adapter_subfolder) and not self.adapter_repo_id:
            raise ValueError(
                f"checkpoint {self.name!r}: adapter revision/subfolder need "
                "adapter_repo_id"
            )
        if self.expected_layers is not None and self.expected_layers < 1:
            raise ValueError(
                f"checkpoint {self.name!r}: expected_layers must be >= 1"
            )

    @property
    def has_adapter(self) -> bool:
        return bool(self.adapter_repo_id or self.adapter_path)


@dataclass(frozen=True)
class RenderingSpec:
    """How a prompt row becomes the string the model actually reads.

    ``chat_template``: tokenizer chat template over the row's ``messages``;
    ``chat_template_path`` (a jinja file) is applied to the tokenizer **in
    memory only** — checkpoints that ship no template (the -pt-derived
    parents) get the canonical one without any model-dir mutation.
    ``raw_transcript``: ``User: ...\\nAssistant:``-style plain transcript.
    ``none``: the row's ``text`` verbatim.

    ``add_special_tokens=None`` resolves per kind: False for chat_template
    (the template emits <bos> itself), True otherwise.
    """

    name: str
    kind: str
    chat_template_path: str | None = None
    add_generation_prompt: bool = True
    add_special_tokens: bool | None = None
    user_prefix: str = "User: "
    assistant_prefix: str = "Assistant:"
    turn_separator: str = "\n"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RenderingSpec.name must be non-empty")
        if self.kind not in _RENDERING_KINDS:
            raise ValueError(
                f"rendering {self.name!r}: kind must be one of "
                f"{_RENDERING_KINDS}: {self.kind!r}"
            )
        if self.chat_template_path and self.kind != "chat_template":
            raise ValueError(
                f"rendering {self.name!r}: chat_template_path only applies to "
                "kind 'chat_template'"
            )

    def resolved_add_special_tokens(self) -> bool:
        if self.add_special_tokens is not None:
            return self.add_special_tokens
        return self.kind != "chat_template"


def _parse_position_kind(kind: str) -> tuple[str, Any]:
    if kind == "last":
        return ("last", None)
    if kind.startswith("from_end:"):
        raw = kind.split(":", 1)[1]
        try:
            k = int(raw)
        except ValueError:
            k = -1
        if k < 1:
            raise ValueError(f"position kind {kind!r}: from_end needs an int >= 1")
        return ("from_end", k)
    if kind.startswith("span_last:"):
        fld = kind.split(":", 1)[1]
        if not fld:
            raise ValueError(f"position kind {kind!r}: span_last needs a field name")
        return ("span_last", fld)
    raise ValueError(
        f"unknown position kind {kind!r}; expected 'last', 'from_end:<k>' or "
        "'span_last:<field>'"
    )


@dataclass(frozen=True)
class PositionSpec:
    """A named token position, resolved per prompt at tokenization time.

    ``expect_text``, when set, is asserted per prompt against the surface
    text of the resolved token (offset-mapping slice) — the "tokenization
    assertions pin the exact token ids" contract, enforced loudly.
    """

    name: str
    kind: str
    expect_text: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PositionSpec.name must be non-empty")
        _parse_position_kind(self.kind)

    def parsed(self) -> tuple[str, Any]:
        return _parse_position_kind(self.kind)


def parse_layers(spec: str | Sequence[int], n_layers: int) -> tuple[int, ...]:
    """Resolve a layer spec against a model's decoder count.

    Valid indices are 0..n_layers inclusive (see module docstring).
    ``"all"`` -> every index; ``"every:k"`` -> (k, 2k, ...); an explicit
    sequence is validated, deduped loudly, and sorted.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1: {n_layers}")
    if isinstance(spec, str):
        if spec == "all":
            return tuple(range(n_layers + 1))
        if spec.startswith("every:"):
            raw = spec.split(":", 1)[1]
            try:
                k = int(raw)
            except ValueError:
                k = -1
            if k < 1:
                raise ValueError(f"layers spec {spec!r}: every needs an int >= 1")
            return tuple(range(k, n_layers + 1, k))
        raise ValueError(
            f"unknown layers spec {spec!r}; expected 'all', 'every:<k>' or a list"
        )
    idx = [int(i) for i in spec]
    if not idx:
        raise ValueError("layers list must be non-empty")
    bad = [i for i in idx if i < 0 or i > n_layers]
    if bad:
        raise ValueError(
            f"layer indices out of range 0..{n_layers}: {sorted(set(bad))}"
        )
    if len(set(idx)) != len(idx):
        raise ValueError(f"duplicate layer indices: {sorted(idx)}")
    return tuple(sorted(idx))


def _layers_key(spec: str | Sequence[int]) -> str:
    """Canonical identity string for a layer spec (CPU-computable — the
    resolved indices need the model's layer count and live in the manifest's
    ``resolved`` block instead)."""
    if isinstance(spec, str):
        return spec
    return "explicit:" + ",".join(str(int(i)) for i in spec)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"cannot hash missing file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ExtractConfig:
    """One extraction run: checkpoints x renderings x positions x layers."""

    checkpoints: tuple[CheckpointRef, ...]
    renderings: tuple[RenderingSpec, ...]
    positions: tuple[PositionSpec, ...]
    layers: str | tuple[int, ...] = "every:2"
    batch_size: int = 16
    max_length: int = 2048
    store_dtype: str = "bfloat16"
    schema_version: int = SCHEMA_VERSION
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoints:
            raise ValueError("ExtractConfig.checkpoints must be non-empty")
        if not self.renderings:
            raise ValueError("ExtractConfig.renderings must be non-empty")
        if not self.positions:
            raise ValueError("ExtractConfig.positions must be non-empty")
        for label, items in (
            ("checkpoint", self.checkpoints),
            ("rendering", self.renderings),
            ("position", self.positions),
        ):
            names = [x.name for x in items]
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate {label} names: {sorted(names)}")
        if isinstance(self.layers, str):
            _layers_probe = self.layers
            if _layers_probe != "all" and not _layers_probe.startswith("every:"):
                raise ValueError(
                    f"unknown layers spec {self.layers!r}; expected 'all', "
                    "'every:<k>' or a list"
                )
            if _layers_probe.startswith("every:"):
                parse_layers(_layers_probe, 1_000_000)  # validates k
        else:
            object.__setattr__(self, "layers", tuple(int(i) for i in self.layers))
            if not self.layers:
                raise ValueError("layers list must be non-empty")
            if any(i < 0 for i in self.layers):
                raise ValueError(f"layer indices must be >= 0: {self.layers}")
            if len(set(self.layers)) != len(self.layers):
                raise ValueError(f"duplicate layer indices: {sorted(self.layers)}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1: {self.batch_size}")
        if self.max_length < 1:
            raise ValueError(f"max_length must be >= 1: {self.max_length}")
        if self.store_dtype not in _STORE_DTYPES:
            raise ValueError(
                f"store_dtype must be one of {_STORE_DTYPES}: {self.store_dtype!r}"
            )

    def checkpoint(self, name: str) -> CheckpointRef:
        for ref in self.checkpoints:
            if ref.name == name:
                return ref
        raise KeyError(
            f"unknown checkpoint {name!r}; configured: "
            f"{[c.name for c in self.checkpoints]}"
        )

    def identity_for(self, ref: CheckpointRef, prompts_sha256: str) -> dict[str, Any]:
        """The CPU-computable identity of one checkpoint's shard.

        Two extractions with equal identity are the same measurement; a shard
        whose recorded identity differs from the current config is refused,
        never overwritten. Chat template files are hashed by content so a
        template edit invalidates caches.
        """
        renderings = []
        for r in self.renderings:
            d = dataclasses.asdict(r)
            d["chat_template_sha256"] = (
                sha256_file(r.chat_template_path) if r.chat_template_path else None
            )
            renderings.append(d)
        return {
            "schema_version": self.schema_version,
            "layer_semantics": LAYER_SEMANTICS,
            "checkpoint": dataclasses.asdict(ref),
            "prompts_sha256": prompts_sha256,
            "renderings": renderings,
            "positions": [dataclasses.asdict(p) for p in self.positions],
            "layers": _layers_key(self.layers),
            "store_dtype": self.store_dtype,
            "max_length": self.max_length,
        }


@dataclass(frozen=True)
class FitConfig:
    """One probe fit: a registered fitter over one (rendering, position,
    layer) slice of a cache. Split *computation* is the caller's job; the
    ``split`` string here is the recorded description of what they did."""

    fitter: str
    layer: int
    position: str
    rendering: str
    label_field: str
    split: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self) -> None:
        for fld in ("fitter", "position", "rendering", "label_field", "split"):
            if not getattr(self, fld):
                raise ValueError(f"FitConfig.{fld} must be non-empty")
        if self.layer < 0:
            raise ValueError(f"FitConfig.layer must be >= 0: {self.layer}")


def _checkpoint_ref_from(data: dict[str, Any], *, where: str) -> CheckpointRef:
    data = dict(data)
    _require_known(data, CheckpointRef, where=where)
    return CheckpointRef(**data)


def _rendering_from(data: dict[str, Any], *, where: str) -> RenderingSpec:
    data = dict(data)
    _require_known(data, RenderingSpec, where=where)
    return RenderingSpec(**data)


def _position_from(data: dict[str, Any], *, where: str) -> PositionSpec:
    data = dict(data)
    _require_known(data, PositionSpec, where=where)
    return PositionSpec(**data)


def extract_config_from(data: dict[str, Any], *, source: str) -> ExtractConfig:
    data = dict(data)
    _require_known(data, ExtractConfig, where=source)
    for key, fn in (
        ("checkpoints", _checkpoint_ref_from),
        ("renderings", _rendering_from),
        ("positions", _position_from),
    ):
        entries = data.get(key)
        if entries is None:
            raise ValueError(f"missing {key!r} in {source}")
        if not isinstance(entries, (list, tuple)):
            raise ValueError(f"{key!r} in {source} must be a list")
        parsed = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{key}[{i}] in {source} must be a mapping")
            parsed.append(fn(entry, where=f"{key}[{i}] in {source}"))
        data[key] = tuple(parsed)
    layers = data.get("layers")
    if isinstance(layers, list):
        data["layers"] = tuple(layers)
    return ExtractConfig(**data)


def fit_config_from(data: dict[str, Any], *, source: str) -> FitConfig:
    data = dict(data)
    _require_known(data, FitConfig, where=source)
    return FitConfig(**data)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no config file at {p}")
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config at {p} must be a mapping")
    return data


def load_extract_config(path: str | Path) -> ExtractConfig:
    return extract_config_from(_load_yaml(path), source=str(path))


def load_fit_config(path: str | Path) -> FitConfig:
    return fit_config_from(_load_yaml(path), source=str(path))
