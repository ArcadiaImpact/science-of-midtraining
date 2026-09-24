"""Shared helpers for the graft_delta_lambda_v1 pod scripts (CPU-importable).

Config-first plumbing (one JSON/YAML mapping merged over a script's
``DEFAULTS``; unknown keys are a ``ValueError``; the mapping path comes from
``argv[1]`` or the script's env var — never flags), UTC logging and atomic
receipts, the Gemma-3 parameter-name canonicalisation and the graft coverage
rules of SPEC §3, the LoRA maths (thin SVD -> ``(A, B)`` factors, merge), the
PEFT-style adapter directory I/O, and the covered-weight snapshot / merge for
a live model. Heavy imports (torch, safetensors) stay inside functions so the
pure-python parts stay unit-testable in the lean venv.

Naming conventions used everywhere in this study:

- *canonical parameter key*: transformers >= 5 ``Gemma3ForConditionalGeneration``
  names (``model.language_model.layers.<i>.self_attn.q_proj.weight``, ...).
  Older saves (``language_model.model.*``, unsloth's mirror) and text-only
  ``Gemma3ForCausalLM`` saves (``model.layers.*``) are mapped onto it by
  :func:`canonical_key`, mirroring transformers' own
  ``_checkpoint_conversion_mapping``.
- *module path*: the canonical key without ``.weight``
  (``model.language_model.layers.<i>.mlp.down_proj``) — the key of every
  per-module artefact (LoRA factors, full delta shards, stats rows).
- *delta name*: ``<arm>__<kind>__all`` (v1's ``<dataset>__<kind>__<fold>``
  score-name shape; ``fold`` is always ``all`` here).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

GB = 1e9
ARMS = ("charter", "coin", "control")
DEFAULT_RANKS = (16, 64, 256, 1024)
LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LM_PREFIX = "model.language_model."
LAYER_PREFIX = "model.language_model.layers."
# Verified for google/gemma-3-27b-{pt,it}: 62 decoder layers x 7 linears.
EXPECTED_LAYERS_27B = 62
EXPECTED_LINEARS_27B = EXPECTED_LAYERS_27B * len(LINEAR_NAMES)  # 434

LINEAR_WEIGHT_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\."
    r"(?P<module>self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$"
)
LINEAR_MODULE_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\."
    r"(?P<module>self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"
)
NORM_WEIGHT_RE = re.compile(
    r"^model\.language_model\.(?:layers\.(?P<layer>\d+)\.)?(?P<module>[A-Za-z0-9_.]*norm)\.weight$"
)
EMBEDDING_KEYS = ("model.language_model.embed_tokens.weight", "lm_head.weight")
VISION_PREFIXES = ("model.vision_tower.", "model.multi_modal_projector.")
# PEFT accepts a string target_modules as a full-match regex over module
# names; by-name targeting ("q_proj") would also hit the SigLIP vision tower.
PEFT_TARGET_MODULES_REGEX = (
    r"^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
)
PEFT_PREFIX = "base_model.model."
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
NORM_DELTA_FILE = "norm_delta.safetensors"
MANIFEST_FILE = "manifest.json"
FULL_INDEX_FILE = "delta.index.json"
DELTA_NAME_RE = re.compile(r"^(?P<arm>[a-z][a-z0-9]*)__(?P<kind>[a-z0-9]+(?:_[a-z0-9]+)*)__all$")

# transformers' Gemma3ForConditionalGeneration._checkpoint_conversion_mapping,
# plus the text-only Gemma3ForCausalLM layout.
_KEY_CONVERSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^language_model\.model\."), "model.language_model."),
    (re.compile(r"^language_model\.lm_head\."), "lm_head."),
    (re.compile(r"^vision_tower\."), "model.vision_tower."),
    (re.compile(r"^multi_modal_projector\."), "model.multi_modal_projector."),
    (re.compile(r"^model\.(?=(?:layers|norm|embed_tokens)(?:\.|$))"), "model.language_model."),
)


# ------------------------------------------------------------ time / logging
def now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def timestamp_tag() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def log(message: str) -> None:
    print(f"[{_dt.datetime.now(_dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}] {message}", flush=True)


def git_commit() -> str:
    """Read-only git provenance via the library's sanctioned helper."""
    from scimt.data_attribution.runner import _scimt_commit

    return _scimt_commit()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def host_maxrss_gb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ------------------------------------------------------------------- config
def load_mapping(path: str | Path) -> dict[str, Any]:
    """JSON or YAML mapping from disk (YAML by suffix)."""
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        import yaml

        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config {path} must be a mapping, got {type(loaded).__name__}")
    return loaded


def merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict-into-dict merge; scalars and lists replace."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_mappings(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_path_from_argv(argv: Sequence[str] | None, env_var: str, script: str) -> str | None:
    """``argv[1]`` (a mapping path) or ``$env_var``; flags are refused."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1 or (args and args[0].startswith("-")):
        raise SystemExit(f"usage: {script} [config.json|yaml] — config-first, no flags; or set ${env_var}")
    return args[0] if args else (os.environ.get(env_var) or None)


def load_config(argv: Sequence[str] | None, env_var: str, script: str, defaults: Mapping[str, Any], factory: Callable[[Mapping[str, Any]], Any]):
    path = config_path_from_argv(argv, env_var, script)
    merged = dict(defaults)
    if path:
        merged = merge_mappings(merged, load_mapping(path))
    return factory(merged)


def reject_unknown(raw: Mapping[str, Any], allowed: Iterable[str], label: str) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(f"{label}: unknown config keys {unknown}")


def require_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an int >= {minimum}, got {value!r}")
    return value


def int_or_none(value: Any, label: str, *, minimum: int = 1) -> int | None:
    return None if value is None else require_int(value, label, minimum=minimum)


def require_float(value: Any, label: str, *, minimum: float | None = None, minimum_exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    value = float(value)
    if minimum is not None and (value < minimum or (minimum_exclusive and value == minimum)):
        raise ValueError(f"{label} must be {'>' if minimum_exclusive else '>='} {minimum}, got {value!r}")
    return value


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean, got {value!r}")
    return value


def require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value


def str_or_none(value: Any, label: str) -> str | None:
    return None if value is None else require_str(value, label)


def require_ranks(value: Any, label: str = "ranks") -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list of ints")
    ranks = tuple(require_int(r, f"{label}[]") for r in value)
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(ranks))


def require_device(value: Any, label: str) -> str:
    value = require_str(value, label)
    if not re.fullmatch(r"(cpu|cuda(:\d+)?|meta)", value):
        raise ValueError(f"{label} must be cpu|cuda|cuda:N, got {value!r}")
    return value


def validate_delta_name(name: str) -> tuple[str, str]:
    """``<arm>__<kind>__all`` -> (arm, kind)."""
    match = DELTA_NAME_RE.match(name)
    if not match:
        raise ValueError(f"delta name {name!r} is not '<arm>__<kind>__all'")
    return match["arm"], match["kind"]


def delta_name(arm: str, kind: str) -> str:
    name = f"{arm}__{kind}__all"
    validate_delta_name(name)
    return name


# --------------------------------------------------------------- coverage
def canonical_key(name: str) -> str:
    """Map a safetensors / state-dict key onto the transformers-5 name."""
    for pattern, replacement in _KEY_CONVERSIONS:
        if pattern.match(name):
            return pattern.sub(replacement, name, count=1)
    return name


def canonical_module_path(module_name: str) -> str:
    return canonical_key(module_name + ".weight")[: -len(".weight")]


@dataclass(frozen=True)
class Coverage:
    kind: str  # linear | norm | embedding | vision | other
    layer: int | None
    module_path: str | None  # for linear / norm: key without ".weight"

    @property
    def module_type(self) -> str | None:
        return self.module_path.rsplit(".", 1)[-1] if self.module_path else None


def classify(key: str) -> Coverage:
    """Graft coverage of a canonical parameter key (SPEC §3)."""
    match = LINEAR_WEIGHT_RE.match(key)
    if match:
        return Coverage("linear", int(match["layer"]), key[: -len(".weight")])
    match = NORM_WEIGHT_RE.match(key)
    if match:
        layer = match["layer"]
        return Coverage("norm", None if layer is None else int(layer), key[: -len(".weight")])
    if key in EMBEDDING_KEYS:
        return Coverage("embedding", None, key[: -len(".weight")])
    if key.startswith(VISION_PREFIXES):
        return Coverage("vision", None, None)
    return Coverage("other", None, None)


def linear_sort_key(module_path: str) -> tuple[int, int]:
    match = LINEAR_MODULE_RE.match(module_path)
    if not match:
        raise ValueError(f"not a covered linear module path: {module_path!r}")
    return int(match["layer"]), LINEAR_NAMES.index(match["module"].rsplit(".", 1)[-1])


def layer_of(module_path: str) -> int | None:
    match = re.match(r"^model\.language_model\.layers\.(\d+)\.", module_path)
    return int(match.group(1)) if match else None


# ------------------------------------------------------------- LoRA maths
def thin_svd(delta, method: str = "full", *, q: int | None = None, niter: int = 4):
    """``(U, S, Vh)`` with descending ``S`` for a 2-D fp32 tensor.

    ``full``: ``torch.linalg.svd`` (exact). ``gram``: exact thin SVD via an
    fp64 eigendecomposition of the smaller Gram matrix (much faster than
    cuSOLVER's gesvd on tall 21504 x 5376 matrices; tiny singular values
    are less accurate, which the top-r truncation never sees). ``lowrank``:
    ``torch.svd_lowrank(q, niter)`` — randomised, keeps ``q`` components.
    """
    import torch

    if delta.ndim != 2:
        raise ValueError("thin_svd expects a 2-D tensor")
    if method == "full":
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        return U, S, Vh
    if method == "gram":
        D = delta.double()
        transposed = D.shape[0] < D.shape[1]
        if transposed:
            D = D.T
        evals, V = torch.linalg.eigh(D.T @ D)
        order = torch.argsort(evals, descending=True)
        evals, V = evals[order].clamp_min(0.0), V[:, order]
        S = evals.sqrt()
        floor = S[0] * 1e-7 if S.numel() else S
        safe = torch.where(S > floor, S, torch.ones_like(S))
        U = (D @ V) / safe
        U[:, S <= floor] = 0.0
        if transposed:
            U, V = V, U
        return U.to(delta.dtype), S.to(delta.dtype), V.T.contiguous().to(delta.dtype)
    if method == "lowrank":
        if q is None:
            raise ValueError("lowrank needs q")
        q = min(int(q), *delta.shape)
        U, S, V = torch.svd_lowrank(delta, q=q, niter=niter)
        return U, S, V.T.contiguous()
    raise ValueError(f"unknown svd method {method!r} (full|gram|lowrank)")


def svd_lora_factors(delta, ranks: Sequence[int], *, method: str = "full", lowrank_extra: int = 128, lowrank_niter: int = 4, svd=None) -> dict[int, tuple[Any, Any]]:
    """``{r: (A, B)}`` with ``B A`` the best rank-``r`` approximation of
    ``delta`` (``out x in``): ``B = U_r sqrt(S_r)`` (``out x r``),
    ``A = sqrt(S_r) V_r^T`` (``r x in``) — PEFT's ``lora_B`` / ``lora_A``
    shapes with ``scaling = 1``. ``svd`` may pass a precomputed ``(U, S, Vh)``."""
    ranks = sorted({int(r) for r in ranks})
    if not ranks or ranks[0] < 1:
        raise ValueError("ranks must be positive ints")
    if svd is None:
        svd = thin_svd(delta.float(), method, q=max(ranks) + lowrank_extra, niter=lowrank_niter)
    U, S, Vh = svd
    factors: dict[int, tuple[Any, Any]] = {}
    for r in ranks:
        k = min(r, S.numel())
        root = S[:k].clamp_min(0.0).sqrt()
        B = (U[:, :k] * root[None, :]).contiguous()
        A = (root[:, None] * Vh[:k]).contiguous()
        factors[r] = (A, B)
    return factors


def captured_energy(S, r: int, fro2: float) -> float:
    """Share of ``||delta||_F^2`` captured by the top-``r`` singular values."""
    if fro2 <= 0:
        return 1.0
    return float((S[:r].double() ** 2).sum().item() / fro2)


def merge_lora(W, A, B, lam: float = 1.0):
    """``W + lam * B A`` computed in fp32, returned in ``W``'s dtype."""
    return (W.float() + float(lam) * (B.float() @ A.float())).to(W.dtype)


def lora_apply(x, A, B):
    """``x delta^T`` for ``delta = B A``: ``(x A^T) B^T`` (``T x in`` -> ``T x out``)."""
    return (x @ A.T) @ B.T


# ------------------------------------------------------------ adapter I/O
def lora_key(module_path: str, which: str) -> str:
    if which not in ("A", "B"):
        raise ValueError("which must be 'A' or 'B'")
    return f"{PEFT_PREFIX}{module_path}.lora_{which}.weight"


def parse_lora_key(key: str) -> tuple[str, str]:
    match = re.fullmatch(re.escape(PEFT_PREFIX) + r"(?P<path>.+)\.lora_(?P<which>[AB])\.weight", key)
    if not match:
        raise ValueError(f"not a PEFT lora key: {key!r}")
    return match["path"], match["which"]


def peft_adapter_config(r: int, *, base_model: str, target_modules: str | list[str] = PEFT_TARGET_MODULES_REGEX) -> dict[str, Any]:
    """``adapter_config.json`` for a scaling-1 LoRA (``lora_alpha = r``)."""
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": int(r),
        "lora_alpha": int(r),
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "init_lora_weights": True,
        "inference_mode": True,
        "target_modules": target_modules,
        "modules_to_save": None,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "use_rslora": False,
        "use_dora": False,
        "revision": None,
        "auto_mapping": None,
    }


@dataclass
class LoraAdapter:
    """A loaded adapter directory: ``modules[module_path] = (A, B)`` plus the
    full-rank norm deltas keyed by canonical parameter name."""

    r: int
    modules: dict[str, tuple[Any, Any]]
    norm_deltas: dict[str, Any]
    config: dict[str, Any]
    manifest: dict[str, Any]
    path: str

    def n_params(self) -> int:
        return sum(A.numel() + B.numel() for A, B in self.modules.values()) + sum(t.numel() for t in self.norm_deltas.values())


def write_lora_adapter(directory: str | Path, r: int, modules: Mapping[str, tuple[Any, Any]], norm_deltas: Mapping[str, Any], *, config: Mapping[str, Any], manifest: Mapping[str, Any]) -> Path:
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Any] = {}
    for module_path, (A, B) in modules.items():
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1] or A.shape[0] > r:
            raise ValueError(f"{module_path}: bad factor shapes A{tuple(A.shape)} B{tuple(B.shape)} for r={r}")
        tensors[lora_key(module_path, "A")] = A.detach().cpu().contiguous()
        tensors[lora_key(module_path, "B")] = B.detach().cpu().contiguous()
    save_file(tensors, str(directory / ADAPTER_WEIGHTS), metadata={"format": "pt"})
    save_file({k: v.detach().cpu().contiguous() for k, v in norm_deltas.items()}, str(directory / NORM_DELTA_FILE), metadata={"format": "pt"})
    write_json(directory / ADAPTER_CONFIG, config)
    write_json(directory / MANIFEST_FILE, manifest)
    return directory


def read_lora_adapter(directory: str | Path, *, device: str = "cpu", dtype: Any = None) -> LoraAdapter:
    import torch
    from safetensors import safe_open

    directory = Path(directory)
    config = read_json(directory / ADAPTER_CONFIG)
    manifest = read_json(directory / MANIFEST_FILE) if (directory / MANIFEST_FILE).is_file() else {}
    r = int(config["r"])
    if config.get("peft_type") != "LORA" or int(config.get("lora_alpha", -1)) != r:
        raise ValueError(f"{directory}: adapter_config must be a LORA with lora_alpha == r (scaling 1)")
    halves: dict[str, dict[str, Any]] = {}
    with safe_open(str(directory / ADAPTER_WEIGHTS), framework="pt", device=device) as handle:
        for key in handle.keys():
            module_path, which = parse_lora_key(key)
            tensor = handle.get_tensor(key)
            halves.setdefault(module_path, {})[which] = tensor if dtype is None else tensor.to(dtype)
    modules: dict[str, tuple[Any, Any]] = {}
    for module_path, pair in halves.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"{directory}: {module_path} lacks a lora_A/lora_B pair")
        A, B = pair["A"], pair["B"]
        if A.shape[0] != B.shape[1] or A.shape[0] > r:
            raise ValueError(f"{directory}: {module_path} factor shapes A{tuple(A.shape)} B{tuple(B.shape)} disagree with r={r}")
        modules[module_path] = (A, B)
    norm_deltas: dict[str, Any] = {}
    norm_path = directory / NORM_DELTA_FILE
    if norm_path.is_file():
        with safe_open(str(norm_path), framework="pt", device=device) as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                norm_deltas[key] = tensor if dtype is None else tensor.to(dtype)
    del torch
    return LoraAdapter(r=r, modules=modules, norm_deltas=norm_deltas, config=config, manifest=manifest, path=str(directory))


class FullDeltaWriter:
    """Sharded ``delta-NNNNN.safetensors`` files (one tensor per module path)
    plus ``delta.index.json`` (``weight_map`` like HF's index) — the full
    delta of one arm is ~54 GB and never sits in RAM whole."""

    def __init__(self, directory: str | Path, *, shard_bytes: int = 8 * 10**9) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard_bytes = int(shard_bytes)
        self._buffer: dict[str, Any] = {}
        self._buffered = 0
        self.weight_map: dict[str, str] = {}
        self.shards: list[dict[str, Any]] = []

    def add(self, module_path: str, tensor: Any) -> None:
        if module_path in self.weight_map or module_path in self._buffer:
            raise ValueError(f"duplicate module {module_path}")
        tensor = tensor.detach().cpu().contiguous()
        self._buffer[module_path] = tensor
        self._buffered += tensor.numel() * tensor.element_size()
        if self._buffered >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        from safetensors.torch import save_file

        name = f"delta-{len(self.shards) + 1:05d}.safetensors"
        save_file(self._buffer, str(self.directory / name), metadata={"format": "pt"})
        for key in self._buffer:
            self.weight_map[key] = name
        self.shards.append({"file": name, "n_tensors": len(self._buffer), "bytes": self._buffered})
        self._buffer, self._buffered = {}, 0

    def close(self, extra: Mapping[str, Any] | None = None) -> Path:
        self.flush()
        return write_json(self.directory / FULL_INDEX_FILE, {"weight_map": self.weight_map, "shards": self.shards, **(extra or {})})


def iter_full_delta(directory: str | Path, *, device: str = "cpu", dtype: Any = None) -> Iterator[tuple[str, Any]]:
    """Yield ``(module_path, delta)`` shard by shard from a FullDeltaWriter dir."""
    from safetensors import safe_open

    directory = Path(directory)
    index = read_json(directory / FULL_INDEX_FILE)
    for shard in index["shards"]:
        with safe_open(str(directory / shard["file"]), framework="pt", device=device) as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                yield key, (tensor if dtype is None else tensor.to(dtype))


def read_norm_deltas(directory: str | Path, *, device: str = "cpu", dtype: Any = None) -> dict[str, Any]:
    from safetensors import safe_open

    out: dict[str, Any] = {}
    with safe_open(str(Path(directory) / NORM_DELTA_FILE), framework="pt", device=device) as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            out[key] = tensor if dtype is None else tensor.to(dtype)
    return out


# --------------------------------------------------------- live-model side
def load_hf_model(snapshot_dir: str | Path, *, dtype: str = "bfloat16", device: str = "cuda:0", gradient_checkpointing: bool = False):
    """The library loader (``AutoModelForCausalLM``, local files only, eval
    mode) plus the Gemma-3 ``token_type_ids`` wrapper (transformers >= 5)."""
    from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod.mean_gradients import (
        patch_gemma3_token_type_ids,
    )
    from scimt.data_attribution.runner import _load_model

    model = _load_model(snapshot_dir, dtype=dtype, device=device, gradient_checkpointing=gradient_checkpointing)
    return patch_gemma3_token_type_ids(model)


def load_hf_tokenizer(snapshot_dir: str | Path):
    from scimt.data_attribution.runner import _load_tokenizer

    return _load_tokenizer(snapshot_dir)


class CoveredModel:
    """The graft-covered surface of a live model: ``linears[module_path]``
    (``nn.Linear`` under the canonical path) and ``norms[param_key]``
    (norm weight parameters). Snapshots the covered weights to the host and
    restores them exactly (bf16 merges are not invertible; SPEC §5 G1)."""

    def __init__(self, model) -> None:
        self.model = model
        self.linears: dict[str, Any] = {}
        for name, module in model.named_modules():
            path = canonical_module_path(name)
            if LINEAR_MODULE_RE.match(path):
                if not hasattr(module, "weight") or module.weight.ndim != 2:
                    raise TypeError(f"{name}: covered module is not a 2-D linear")
                self.linears[path] = module
        self.norms: dict[str, Any] = {}
        for name, parameter in model.named_parameters():
            key = canonical_key(name)
            if NORM_WEIGHT_RE.match(key):
                self.norms[key] = parameter
        if not self.linears:
            raise RuntimeError("no covered linears found — not a Gemma-3 language model?")

    @property
    def layers(self) -> int:
        return len({layer_of(p) for p in self.linears})

    def snapshot(self) -> dict[str, Any]:
        snap = {path: module.weight.detach().to("cpu", copy=True) for path, module in self.linears.items()}
        snap.update({key: parameter.detach().to("cpu", copy=True) for key, parameter in self.norms.items()})
        return snap

    def restore(self, snap: Mapping[str, Any]) -> int:
        import torch

        n = 0
        with torch.no_grad():
            for path, module in self.linears.items():
                module.weight.copy_(snap[path].to(module.weight.device))
                n += 1
            for key, parameter in self.norms.items():
                parameter.copy_(snap[key].to(parameter.device))
                n += 1
        return n

    def check_complete(self, module_paths: Iterable[str], *, what: str) -> None:
        """Every covered linear of the model must carry a delta (``strict`` merges).

        Catches key-mapping slips between the checkpoint layout
        (``language_model.model.layers.N...``) and the transformers-5 module
        paths (``model.language_model.layers.N...``) before they silently
        merge a partial graft."""
        missing = sorted(set(self.linears) - set(module_paths))
        if missing:
            raise KeyError(f"{what}: {len(missing)} covered linears have no delta, e.g. {missing[:3]}")

    def merge_lora_adapter(self, adapter: LoraAdapter, lam: float = 1.0, *, strict: bool = True) -> dict[str, int]:
        import torch

        if strict:
            self.check_complete(adapter.modules, what=f"adapter {adapter.path}")
        merged = norms = 0
        with torch.no_grad():
            for module_path, (A, B) in adapter.modules.items():
                module = self.linears.get(module_path)
                if module is None:
                    if strict:
                        raise KeyError(f"adapter module {module_path} not in the model")
                    continue
                W = module.weight
                W.copy_(merge_lora(W, A.to(W.device), B.to(W.device), lam))
                merged += 1
            norms = self._merge_norms(adapter.norm_deltas, lam, strict=strict)
        return {"linears": merged, "norms": norms}

    def merge_full(self, items: Iterable[tuple[str, Any]], norm_deltas: Mapping[str, Any], lam: float = 1.0, *, strict: bool = True) -> dict[str, int]:
        import torch

        merged = 0
        seen: list[str] = []
        with torch.no_grad():
            for module_path, delta in items:
                module = self.linears.get(module_path)
                if module is None:
                    if strict:
                        raise KeyError(f"delta module {module_path} not in the model")
                    continue
                W = module.weight
                W.copy_((W.float() + float(lam) * delta.to(W.device).float()).to(W.dtype))
                merged += 1
                seen.append(module_path)
            if strict:
                self.check_complete(seen, what="full delta")
            norms = self._merge_norms(norm_deltas, lam, strict=strict)
        return {"linears": merged, "norms": norms}

    def _merge_norms(self, norm_deltas: Mapping[str, Any], lam: float, *, strict: bool) -> int:
        if strict:
            missing = sorted(set(self.norms) - set(norm_deltas))
            if missing:
                raise KeyError(f"norm deltas lack {len(missing)} model norms, e.g. {missing[:3]}")
        n = 0
        for key, delta in norm_deltas.items():
            parameter = self.norms.get(key)
            if parameter is None:
                if strict:
                    raise KeyError(f"norm delta {key} not in the model")
                continue
            parameter.copy_((parameter.float() + float(lam) * delta.to(parameter.device).float()).to(parameter.dtype))
            n += 1
        return n


# --------------------------------------------------------------- receipts
@dataclass
class Receipt:
    """A JSON receipt rewritten on every status change (v1 convention)."""

    path: Path
    static: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    def write(self, status: str, **extra: Any) -> Path:
        return write_json(
            self.path,
            {
                "status": status,
                **self.static,
                "created_at": now_iso(),
                "elapsed_s": time.time() - self.started,
                "timings_s": dict(self.timings),
                "host_maxrss_gb": host_maxrss_gb(),
                **extra,
            },
        )


def cuda_peak_gb(device: str) -> float | None:
    try:
        import torch

        if not device.startswith("cuda") or not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated(torch.device(device)) / GB
    except Exception:  # noqa: BLE001 — telemetry only
        return None
