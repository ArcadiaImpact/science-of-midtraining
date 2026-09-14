"""λ-gradient scorer (SPEC §4): ``g_row = dL_row/dλ`` for every resident delta.

For the graft ``theta(λ) = theta_it + λ·delta`` the chain rule gives, per
covered linear ``y = x W^T``,

    dL/dλ = Σ_modules Σ_t grad_out_t · (delta_m x_t) + Σ_norms grad_w · delta_w,

so one backward pass yields ``g`` for EVERY resident delta at once without
forming a single weight gradient: a forward hook on each covered
``nn.Linear`` keeps its input ``x`` (T x in, bf16) and registers a tensor
hook on the output that receives ``grad_out`` (T x out); both are shipped to
``delta_device`` where, for LoRA deltas ``delta = B A`` concatenated across
all resident LoRAs (``A_cat``: Σr x in, ``B_cat``: out x Σr),

    P = x A_cat^T,  Q = grad_out B_cat  (T x Σr, bf16 GEMMs, fp32 accumulate)
    dot_k += Σ_{t, columns of k} P ⊙ Q      (fp32),

which is Σ_t grad_out_t · B_k A_k x_t without the T x out intermediate; a
full delta contributes ``Σ_t grad_out_t · (delta x_t)``. Norm weights
(Gemma3 RMSNorm ``y = x̂ (1 + w)``) are the only parameters with
``requires_grad``: their tiny ``.grad`` dotted with each delta's norm delta
gives the norm term, and because the first norm sits before layer 0's q/k/v
projections, that alone makes every covered linear's output require grad
(a forward pre-hook additionally forces grad flow if a model has no norms).

Modes: ``lam0`` (``theta_it`` unmodified) and ``lam1`` (``graft``: merge one
adapter into the model before scoring — ``W += lam·B A``, norms ``+= lam·delta``
— so the same hooks yield ``g(λ)`` at the merged point and the row loss
there). ``graft.kind: "full"`` merges the exact sharded full delta instead
of a LoRA (the real deltas are far from low-rank). Kinds are the caller's
(``<arm>__<kind>__all`` names from the driver: ``lam0_r<r>``, ``lam0_full``,
``lam1_r<r>``, ``lam1x_r<r>``; full graft: ``lam1full``, ``lam1full_r<r>``,
``lam1fullx_r<r>``).

Sign: ``scores[name] = -g`` (positive = grafting lowers the row's loss;
v1's convention), ``raw_dl_dlambda[name] = g``. Row loss = summed
assistant-token CE (``per_sequence_sum`` via ``ChatSFTDataset`` masking,
rows trimmed to their last target position — exact under causal attention).

Self-check (``oracle_rows``): the first rows are re-scored the slow way —
weight gradients enabled on every covered linear, ``⟨∇_W L, delta_m⟩``
accumulated per module from ``param.grad`` (freed immediately) — and the
two routes must agree to ``oracle_rel_tol`` (else exit 99 when
``oracle_strict``). Repeats (``repeats > 1``) re-score the schedule for the
bf16 run-to-run noise floor (G4). Config-first (``argv[1]`` or
``$SCIMT_GRAFT_SCORE_CONFIG`` over DEFAULTS); resumable (rows already in
``out_path`` are skipped).
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod.score_eft_rows import (  # noqa: E402
    DEFAULT_EPISODE_SEED,
    DEFAULT_GROUPS,
    ChatRows,
    RowMeta,
    load_eft_rows,
    order_rows,
    read_records,
    select_rows,
    selection_summary,
)
from experiments.improved_midtraining.graft_delta_lambda_v1.pod.common import (  # noqa: E402
    ADAPTER_WEIGHTS,
    FULL_INDEX_FILE,
    GB,
    MANIFEST_FILE,
    NORM_DELTA_FILE,
    CoveredModel,
    Receipt,
    cuda_peak_gb,
    git_commit,
    int_or_none,
    iter_full_delta,
    load_config,
    load_hf_model,
    load_hf_tokenizer,
    log,
    now_iso,
    parse_lora_key,
    read_json,
    read_lora_adapter,
    read_norm_deltas,
    reject_unknown,
    require_bool,
    require_device,
    require_float,
    require_int,
    require_str,
    sha256_file,
    str_or_none,
    timestamp_tag,
    validate_delta_name,
    write_json,
)

CONFIG_ENV = "SCIMT_GRAFT_SCORE_CONFIG"
ORACLE_EXIT = 99
MODES = ("lam0", "lam1")
DELTA_KINDS = ("lora", "full")
ROW_ORDERS = ("interleave_groups", "file")
REQUIRED_SCORE_KEYS = ("row_id", "group", "episode_id", "subtype", "n_target_tokens", "loss", "grad_norm", "scores", "raw_dl_dlambda", "loss_lam1", "mode", "pass")
EXPECTED_LINEARS_27B = 62 * 7


# ------------------------------------------------------------------- config
@dataclass(frozen=True)
class DeltaSpec:
    name: str  # <arm>__<kind>__all
    kind: str  # lora | full
    path: str  # adapter dir (lora) or full delta dir (full)
    arm: str

    @classmethod
    def from_mapping(cls, raw: Any, *, label: str) -> DeltaSpec:
        reject_unknown(raw, ("name", "kind", "path", "arm"), label)
        name = require_str(raw.get("name"), f"{label}.name")
        arm_from_name, _ = validate_delta_name(name)
        kind = require_str(raw.get("kind"), f"{label}.kind")
        if kind not in DELTA_KINDS:
            raise ValueError(f"{label}.kind must be lora|full, got {kind!r}")
        arm = require_str(raw.get("arm", arm_from_name), f"{label}.arm")
        if arm != arm_from_name:
            raise ValueError(f"{label}: arm {arm!r} does not match the name prefix {arm_from_name!r}")
        return cls(name=name, kind=kind, path=require_str(raw.get("path"), f"{label}.path"), arm=arm)


@dataclass(frozen=True)
class GraftSpec:
    """What gets merged into ``theta_it`` before a ``lam1`` pass.

    ``kind: lora`` — ``adapter_dir`` is a PEFT adapter dir (``W += lam·B A``);
    ``kind: full`` — ``adapter_dir`` is the sharded full-delta dir written by
    extract (``adapters/<arm>/full/``: ``delta-NNNNN.safetensors`` +
    ``delta.index.json`` + ``norm_delta.safetensors``), merged exactly
    (``W += lam·delta``). ``path`` is accepted as an alias of ``adapter_dir``."""

    adapter_dir: str
    lam: float
    arm: str
    kind: str = "lora"

    @classmethod
    def from_mapping(cls, raw: Any, *, label: str) -> GraftSpec:
        reject_unknown(raw, ("adapter_dir", "path", "lam", "arm", "kind"), label)
        if raw.get("adapter_dir") is not None and raw.get("path") is not None and raw["adapter_dir"] != raw["path"]:
            raise ValueError(f"{label}: adapter_dir and path disagree")
        directory = raw.get("adapter_dir") if raw.get("adapter_dir") is not None else raw.get("path")
        kind = raw.get("kind", "lora")
        if kind not in GRAFT_KINDS:
            raise ValueError(f"{label}.kind must be one of {GRAFT_KINDS}, got {kind!r}")
        return cls(require_str(directory, f"{label}.adapter_dir"), require_float(raw.get("lam", 1.0), f"{label}.lam"), require_str(raw.get("arm"), f"{label}.arm"), kind)


GRAFT_KINDS = ("lora", "full")


def merge_graft(covered: CoveredModel, graft: GraftSpec, *, device: str) -> dict[str, Any]:
    """Merge ``graft`` into the covered weights in place (fp32 add, cast back
    to the weight dtype) and return the record written to the receipt.

    Both kinds add ``lam·delta`` to every covered linear and to the norm
    weights; a graft that lacks any covered module or norm of the loaded model
    is refused (``KeyError``) rather than merged partially."""
    directory = Path(graft.adapter_dir)
    manifest = read_json(directory / MANIFEST_FILE) if (directory / MANIFEST_FILE).is_file() else {}
    if graft.kind == "lora":
        adapter = read_lora_adapter(directory, device=device)
        counts = covered.merge_lora_adapter(adapter, graft.lam)
        detail: dict[str, Any] = {"r": adapter.r, "adapter_manifest": {k: adapter.manifest.get(k) for k in ("arm", "rank", "energy_captured", "pt_snapshot", "mid_snapshot")}}
        del adapter
    elif graft.kind == "full":
        if not (directory / FULL_INDEX_FILE).is_file():
            raise FileNotFoundError(f"graft kind=full: {directory / FULL_INDEX_FILE} missing (expected the sharded full delta written by extract_delta_lora.py)")
        counts = covered.merge_full(iter_full_delta(directory, device=device), read_norm_deltas(directory, device=device), graft.lam)
        detail = {"r": None, "adapter_manifest": {k: manifest.get(k) for k in ("arm", "kind", "pt_snapshot", "mid_snapshot")}}
    else:  # pragma: no cover — from_mapping validates
        raise ValueError(f"unknown graft kind {graft.kind!r}")
    return {"arm": graft.arm, "lam": graft.lam, "kind": graft.kind, "adapter_dir": str(directory), **detail, **counts}


@dataclass(frozen=True)
class ScoreConfig:
    it_snapshot: str
    rows_path: str
    out_path: str
    pass_name: str
    deltas: tuple[DeltaSpec, ...]
    mode: str = "lam0"
    graft: GraftSpec | None = None
    loss_lam0_path: str | None = None  # lam1 mode: the λ = 0 scores file whose per-row ``loss`` is carried over as L(0)
    tokenizer_snapshot: str | None = None
    groups: tuple[str, ...] = DEFAULT_GROUPS
    rows_filter: str | int = "all"  # "all" or n episodes per pair type (seeded)
    episode_seed: int = DEFAULT_EPISODE_SEED
    row_ids: tuple[str, ...] | None = None
    row_limit: int | None = None
    row_order: str = "interleave_groups"
    repeats: int = 1
    model_device: str = "cuda:0"
    delta_device: str = "cuda:1"
    dtype: str = "bfloat16"
    delta_dtype: str = "bfloat16"
    sequence_length: int = 8192
    gradient_checkpointing: bool = False
    evidence_dir: str | None = None  # None -> out_path.parent / "evidence"
    progress_every: int = 25
    oracle_rows: int = 8
    oracle_rel_tol: float = 1e-2
    oracle_strict: bool = True
    resume: bool = True
    expected_linears: int | None = EXPECTED_LINEARS_27B
    ensure_grad_flow: bool = True

    _KEYS = (
        "it_snapshot", "rows_path", "out_path", "pass_name", "deltas", "mode", "graft", "loss_lam0_path", "tokenizer_snapshot",
        "groups", "rows_filter", "episode_seed", "row_ids", "row_limit", "row_order", "repeats", "model_device",
        "delta_device", "dtype", "delta_dtype", "sequence_length", "gradient_checkpointing", "evidence_dir",
        "progress_every", "oracle_rows", "oracle_rel_tol", "oracle_strict", "resume", "expected_linears",
        "ensure_grad_flow",
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ScoreConfig:
        reject_unknown(raw, cls._KEYS, "score_lambda_grad")
        deltas_raw = raw.get("deltas")
        if not isinstance(deltas_raw, (list, tuple)) or not deltas_raw:
            raise ValueError("deltas must be a non-empty list of {name, kind, path, arm}")
        deltas = tuple(DeltaSpec.from_mapping(d, label=f"deltas[{i}]") for i, d in enumerate(deltas_raw))
        if len({d.name for d in deltas}) != len(deltas):
            raise ValueError("delta names must be unique")
        mode = require_str(raw.get("mode", "lam0"), "mode")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        graft = GraftSpec.from_mapping(raw["graft"], label="graft") if raw.get("graft") is not None else None
        if (mode == "lam1") != (graft is not None):
            raise ValueError("mode lam1 requires a graft and mode lam0 forbids one")
        loss_lam0_path = str_or_none(raw.get("loss_lam0_path"), "loss_lam0_path")
        if loss_lam0_path is not None and mode != "lam1":
            raise ValueError("loss_lam0_path only applies to mode lam1")
        rows_filter = raw.get("rows_filter", "all")
        if rows_filter != "all":
            rows_filter = require_int(rows_filter, "rows_filter")
        row_ids = raw.get("row_ids")
        if row_ids is not None:
            if not isinstance(row_ids, (list, tuple)) or not all(isinstance(r, str) and r for r in row_ids):
                raise ValueError("row_ids must be a list of row ids (<group>:<episode_id>)")
            row_ids = tuple(row_ids)
        row_order = require_str(raw.get("row_order", "interleave_groups"), "row_order")
        if row_order not in ROW_ORDERS:
            raise ValueError(f"row_order must be one of {ROW_ORDERS}")
        pass_name = require_str(raw.get("pass_name"), "pass_name")
        if not pass_name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("pass_name must be a bare token")
        return cls(
            it_snapshot=require_str(raw.get("it_snapshot"), "it_snapshot"),
            rows_path=require_str(raw.get("rows_path"), "rows_path"),
            out_path=require_str(raw.get("out_path"), "out_path"),
            pass_name=pass_name,
            deltas=deltas,
            mode=mode,
            graft=graft,
            loss_lam0_path=loss_lam0_path,
            tokenizer_snapshot=str_or_none(raw.get("tokenizer_snapshot"), "tokenizer_snapshot"),
            groups=tuple(require_str(g, "groups[]") for g in raw.get("groups", list(DEFAULT_GROUPS))),
            rows_filter=rows_filter,
            episode_seed=require_int(raw.get("episode_seed", DEFAULT_EPISODE_SEED), "episode_seed", minimum=0),
            row_ids=row_ids,
            row_limit=int_or_none(raw.get("row_limit"), "row_limit"),
            row_order=row_order,
            repeats=require_int(raw.get("repeats", 1), "repeats"),
            model_device=require_device(raw.get("model_device", "cuda:0"), "model_device"),
            delta_device=require_device(raw.get("delta_device", "cuda:1"), "delta_device"),
            dtype=require_str(raw.get("dtype", "bfloat16"), "dtype"),
            delta_dtype=require_str(raw.get("delta_dtype", "bfloat16"), "delta_dtype"),
            sequence_length=require_int(raw.get("sequence_length", 8192), "sequence_length", minimum=2),
            gradient_checkpointing=require_bool(raw.get("gradient_checkpointing", False), "gradient_checkpointing"),
            evidence_dir=str_or_none(raw.get("evidence_dir"), "evidence_dir"),
            progress_every=require_int(raw.get("progress_every", 25), "progress_every"),
            oracle_rows=require_int(raw.get("oracle_rows", 8), "oracle_rows", minimum=0),
            oracle_rel_tol=require_float(raw.get("oracle_rel_tol", 1e-2), "oracle_rel_tol", minimum=0.0, minimum_exclusive=True),
            oracle_strict=require_bool(raw.get("oracle_strict", True), "oracle_strict"),
            resume=require_bool(raw.get("resume", True), "resume"),
            expected_linears=int_or_none(raw.get("expected_linears", EXPECTED_LINEARS_27B), "expected_linears"),
            ensure_grad_flow=require_bool(raw.get("ensure_grad_flow", True), "ensure_grad_flow"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["deltas"] = [asdict(d) for d in self.deltas]
        payload["graft"] = None if self.graft is None else asdict(self.graft)
        payload["groups"] = list(self.groups)
        payload["row_ids"] = None if self.row_ids is None else list(self.row_ids)
        return payload

    @property
    def evidence_path(self) -> Path:
        return Path(self.evidence_dir) if self.evidence_dir else Path(self.out_path).parent / "evidence"


DEFAULTS: dict[str, Any] = {
    "it_snapshot": None,
    "rows_path": "/workspace/graft/eft_rows/eft_rows.jsonl",
    "out_path": "/workspace/graft/scores/lam0.jsonl",
    "pass_name": "lam0",
    "deltas": [],
    "mode": "lam0",
    "model_device": "cuda:0",
    "delta_device": "cuda:1",
    "sequence_length": 8192,
}


# ---------------------------------------------------------- resident deltas
class ResidentDeltas:
    """All deltas of a pass, resident on ``device``: per covered module the
    concatenated LoRA factors (+ per-delta views for the oracle), the full
    deltas, and one ``(n_deltas x N_norm)`` matrix of norm deltas."""

    def __init__(self, specs: Sequence[DeltaSpec], *, device: str, dtype: str, module_paths: Sequence[str], norm_keys: Sequence[str]) -> None:
        self.specs = list(specs)
        self.names = [s.name for s in specs]
        self.device = device
        self.dtype_name = dtype
        self.module_paths = list(module_paths)
        self.norm_keys = list(norm_keys)
        self.cat: dict[str, tuple[Any, Any, Any]] = {}  # module -> (A_cat, B_cat, segment index)
        self.views: dict[str, dict[int, tuple[Any, Any]]] = {}  # module -> {delta index: (A, B)}
        self.full: dict[str, list[tuple[int, Any]]] = {}  # module -> [(delta index, delta)]
        self.norm_matrix: Any = None
        self.norm_offsets: dict[str, tuple[int, int]] = {}
        self.bytes = 0
        self.manifests: dict[str, Any] = {}
        self.norm_all_zero = False
        self.norm_delta_fro: dict[str, float] = {}

    def load(self) -> dict[str, Any]:
        import torch
        from safetensors import safe_open

        dtype = getattr(torch, self.dtype_name)
        device = torch.device(self.device)
        t0 = time.time()
        # norm layout
        offset = 0
        norm_numel: dict[str, int] = {}
        # ---- pass 1: shapes ------------------------------------------------------
        lora_shapes: dict[str, dict[int, tuple[tuple[int, ...], tuple[int, ...]]]] = {}
        handles: dict[int, tuple[str, Any]] = {}
        for index, spec in enumerate(self.specs):
            if spec.kind == "lora":
                weights = Path(spec.path) / ADAPTER_WEIGHTS
                if not weights.is_file():
                    raise FileNotFoundError(f"{spec.name}: {weights} missing")
                handle = safe_open(str(weights), framework="pt", device=self.device).__enter__()
                handles[index] = (str(weights), handle)
                pairs: dict[str, dict[str, tuple[int, ...]]] = {}
                for key in handle.keys():
                    module_path, which = parse_lora_key(key)
                    pairs.setdefault(module_path, {})[which] = tuple(int(x) for x in handle.get_slice(key).get_shape())
                for module_path, pair in pairs.items():
                    if module_path not in self.module_paths:
                        raise KeyError(f"{spec.name}: adapter module {module_path} is not a covered module of the model")
                    if set(pair) != {"A", "B"}:
                        raise KeyError(f"{spec.name}: {module_path} lacks a lora_A/lora_B pair")
                    lora_shapes.setdefault(module_path, {})[index] = (pair["A"], pair["B"])
                missing = [m for m in self.module_paths if m not in pairs]
                if missing:
                    raise KeyError(f"{spec.name}: adapter lacks {len(missing)} of the model's {len(self.module_paths)} covered linears, e.g. {missing[:3]}")
                self.manifests[spec.name] = read_json(Path(spec.path) / "manifest.json") if (Path(spec.path) / "manifest.json").is_file() else {}
            elif spec.kind == "full":
                if not (Path(spec.path) / FULL_INDEX_FILE).is_file():
                    raise FileNotFoundError(f"{spec.name}: {Path(spec.path) / FULL_INDEX_FILE} missing")
                self.manifests[spec.name] = read_json(Path(spec.path) / "manifest.json") if (Path(spec.path) / "manifest.json").is_file() else {}
        # ---- pass 2: allocate + fill the LoRA cats ------------------------------------
        for module_path in self.module_paths:
            per_delta = lora_shapes.get(module_path)
            if not per_delta:
                continue
            total_r = sum(shape_a[0] for shape_a, _ in per_delta.values())
            in_dim = {shape_a[1] for shape_a, _ in per_delta.values()}
            out_dim = {shape_b[0] for _, shape_b in per_delta.values()}
            if len(in_dim) != 1 or len(out_dim) != 1:
                raise ValueError(f"{module_path}: inconsistent factor shapes across deltas {per_delta}")
            A_cat = torch.empty((total_r, in_dim.pop()), dtype=dtype, device=device)
            B_cat = torch.empty((out_dim.pop(), total_r), dtype=dtype, device=device)
            segment = torch.empty((total_r,), dtype=torch.int64, device=device)
            start = 0
            views: dict[int, tuple[Any, Any]] = {}
            for index in sorted(per_delta):
                shape_a, shape_b = per_delta[index]
                r = shape_a[0]
                _, handle = handles[index]
                A_cat[start : start + r].copy_(handle.get_tensor(f"base_model.model.{module_path}.lora_A.weight").to(dtype))
                B_cat[:, start : start + r].copy_(handle.get_tensor(f"base_model.model.{module_path}.lora_B.weight").to(dtype))
                segment[start : start + r] = index
                views[index] = (A_cat[start : start + r], B_cat[:, start : start + r])
                start += r
            self.cat[module_path] = (A_cat, B_cat, segment)
            self.views[module_path] = views
            self.bytes += (A_cat.numel() + B_cat.numel()) * A_cat.element_size()
        for _, handle in handles.values():
            handle.__exit__(None, None, None)
        # ---- full deltas ----------------------------------------------------------------
        for index, spec in enumerate(self.specs):
            if spec.kind != "full":
                continue
            n = 0
            seen: set[str] = set()
            for module_path, tensor in iter_full_delta(spec.path, device=self.device, dtype=dtype):
                if module_path not in self.module_paths:
                    raise KeyError(f"{spec.name}: full-delta module {module_path} is not a covered module of the model")
                self.full.setdefault(module_path, []).append((index, tensor))
                self.bytes += tensor.numel() * tensor.element_size()
                seen.add(module_path)
                n += 1
            missing = [m for m in self.module_paths if m not in seen]
            if missing:
                raise KeyError(f"{spec.name}: full delta lacks {len(missing)} of the model's {len(self.module_paths)} covered linears, e.g. {missing[:3]}")
            log(f"resident full delta {spec.name}: {n} modules")
        # ---- norm deltas -> matrix (sizes from the first delta that has each key)
        rows: list[Any] = []
        for index, spec in enumerate(self.specs):
            norm_path = Path(spec.path) / NORM_DELTA_FILE
            if not norm_path.is_file():
                raise FileNotFoundError(f"{spec.name}: {norm_path} missing")
            with safe_open(str(norm_path), framework="pt", device=self.device) as handle:
                keys = set(handle.keys())
                missing = [k for k in self.norm_keys if k not in keys]
                if missing:
                    raise KeyError(f"{spec.name}: norm deltas lack {missing[:5]} ({len(missing)} missing)")
                parts = []
                for key in self.norm_keys:
                    tensor = handle.get_tensor(key).float().flatten()
                    if key not in norm_numel:
                        norm_numel[key] = tensor.numel()
                        self.norm_offsets[key] = (offset, offset + tensor.numel())
                        offset += tensor.numel()
                    elif norm_numel[key] != tensor.numel():
                        raise ValueError(f"{spec.name}: norm delta {key} numel {tensor.numel()} != {norm_numel[key]}")
                    parts.append(tensor)
                rows.append(torch.cat(parts) if parts else torch.zeros((0,), device=device))
        self.norm_matrix = torch.stack(rows) if rows else torch.zeros((len(self.specs), 0), device=device)
        self.bytes += self.norm_matrix.numel() * 4
        # The 27B midtrain left every norm weight untouched (delta exactly 0):
        # keep the schema (files + matrix) but make the norm term a no-op.
        self.norm_delta_fro = {name: float(self.norm_matrix[i].norm().item()) if self.norm_matrix.shape[1] else 0.0 for i, name in enumerate(self.names)}
        self.norm_all_zero = self.norm_matrix.shape[1] == 0 or not bool(self.norm_matrix.any().item())
        seconds = time.time() - t0
        summary = {
            "n_deltas": len(self.specs),
            "n_lora_modules": len(self.cat),
            "n_full_modules": sum(len(v) for v in self.full.values()),
            "norm_elements": int(offset),
            "norm_delta_fro": self.norm_delta_fro,
            "norm_all_zero": self.norm_all_zero,
            "resident_gb": self.bytes / GB,
            "load_seconds": seconds,
            "names": list(self.names),
        }
        log(f"resident deltas: {json.dumps({k: v for k, v in summary.items() if k != 'names'})}")
        if self.norm_all_zero:
            log("note: all resident norm deltas are exactly zero — the norm term of dL/dlambda is skipped (norm weights keep requires_grad so backward still reaches every covered linear)")
        return summary

    def norm_grad_vector(self, grads: Mapping[str, Any]):
        import torch

        parts = [grads[key].detach().float().flatten().to(self.device) for key in self.norm_keys]
        return torch.cat(parts) if parts else torch.zeros((0,), device=self.device)

    def contribution(self, module_path: str, x: Any, g: Any, dots: Any) -> None:
        """``dots[k] += Σ_t grad_out_t · (delta_k x_t)`` for every resident delta."""
        entry = self.cat.get(module_path)
        if entry is not None:
            A_cat, B_cat, segment = entry
            P = x @ A_cat.T  # T x Σr
            Q = g @ B_cat  # T x Σr
            dots.index_add_(0, segment, (P.float() * Q.float()).sum(0))
        for index, delta in self.full.get(module_path, ()):
            if delta.shape[0] <= delta.shape[1]:
                dots[index] += ((x @ delta.T).float() * g.float()).sum()
            else:
                dots[index] += ((g @ delta).float() * x.float()).sum()

    def norm_contribution(self, grads: Mapping[str, Any], dots: Any) -> None:
        if self.norm_all_zero or self.norm_matrix.shape[1] == 0:
            return
        dots += self.norm_matrix @ self.norm_grad_vector(grads)

    def materialised(self, module_path: str):
        """Yield ``(delta index, delta as fp32 out x in)`` for the oracle."""
        for index, (A, B) in self.views.get(module_path, {}).items():
            yield index, B.float() @ A.float()
        for index, delta in self.full.get(module_path, ()):
            yield index, delta.float()


# ------------------------------------------------------------------ engine
@dataclass
class RowScore:
    loss: float
    dots: list[float]
    fired: int
    seconds: float


class LambdaGradEngine:
    """Hooks on the covered linears + norm-weight grads -> one backward pass
    gives ``g`` for every resident delta (see module docstring)."""

    def __init__(self, model, covered: CoveredModel, resident: ResidentDeltas, *, delta_device: str, ensure_grad_flow: bool = True) -> None:
        import torch

        self.model = model
        self.covered = covered
        self.resident = resident
        self.delta_device = torch.device(delta_device)
        self.dots = torch.zeros((len(resident.names),), dtype=torch.float32, device=self.delta_device)
        self.fired = 0
        self.handles: list[Any] = []
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in covered.norms.values():
            parameter.requires_grad_(True)
        for module_path, module in covered.linears.items():
            if ensure_grad_flow:
                self.handles.append(module.register_forward_pre_hook(self._pre_hook))
            self.handles.append(module.register_forward_hook(self._forward_hook(module_path)))

    @staticmethod
    def _pre_hook(module, inputs):
        x = inputs[0]
        if x.requires_grad:
            return None
        return (x.detach().requires_grad_(True), *inputs[1:])

    def _forward_hook(self, module_path: str):
        engine = self

        def hook(module, inputs, output):
            if not output.requires_grad:
                raise RuntimeError(f"{module_path}: output does not require grad — the backward pass would never reach this module")
            x = inputs[0].detach()
            xd = x.reshape(-1, x.shape[-1])

            def grad_hook(grad):
                engine.fired += 1
                gd = grad.detach().reshape(-1, grad.shape[-1])
                engine.resident.contribution(module_path, xd.to(engine.delta_device), gd.to(engine.delta_device), engine.dots)

            output.register_hook(grad_hook)

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _norm_grads(self) -> dict[str, Any]:
        grads = {}
        for key, parameter in self.covered.norms.items():
            if parameter.grad is None:
                raise RuntimeError(f"norm weight {key} received no gradient")
            grads[key] = parameter.grad
        return grads

    def _clear_norm_grads(self) -> None:
        for parameter in self.covered.norms.values():
            parameter.grad = None

    def score(self, batch, loss_adapter) -> RowScore:
        import torch

        t0 = time.time()
        self.dots.zero_()
        self.fired = 0
        losses = loss_adapter.per_datapoint_losses(batch).losses
        if losses.numel() != 1:
            raise RuntimeError(f"expected one loss, got {losses.numel()}")
        loss = losses[0]
        loss.backward()
        self.resident.norm_contribution(self._norm_grads(), self.dots)
        self._clear_norm_grads()
        if self.delta_device.type == "cuda":
            torch.cuda.synchronize(self.delta_device)
        expected = len(self.covered.linears)
        if self.fired != expected:
            raise RuntimeError(f"backward hooks fired for {self.fired} of {expected} covered linears")
        return RowScore(loss=float(loss.item()), dots=[float(v) for v in self.dots.tolist()], fired=self.fired, seconds=time.time() - t0)

    def oracle(self, batch, loss_adapter) -> RowScore:
        """Weight-gradient route: ``Σ_m ⟨∇_W_m L, delta_m⟩`` via post-accumulate
        hooks (each grad freed as soon as it is dotted) + the same norm term."""
        import torch

        t0 = time.time()
        dots = torch.zeros_like(self.dots)
        hooks = []
        weights = [(path, module.weight) for path, module in self.covered.linears.items()]

        def make(module_path: str):
            def hook(parameter) -> None:
                grad = parameter.grad.detach().to(self.delta_device).float()
                for index, delta in self.resident.materialised(module_path):
                    dots[index] += (grad * delta).sum()
                parameter.grad = None

            return hook

        for path, weight in weights:
            weight.requires_grad_(True)
            hooks.append(weight.register_post_accumulate_grad_hook(make(path)))
        try:
            self.fired = 0
            self.dots.zero_()
            losses = loss_adapter.per_datapoint_losses(batch).losses
            loss = losses[0]
            loss.backward()
            self.resident.norm_contribution(self._norm_grads(), dots)
            self._clear_norm_grads()
        finally:
            for hook in hooks:
                hook.remove()
            for _, weight in weights:
                weight.grad = None
                weight.requires_grad_(False)
        if self.delta_device.type == "cuda":
            torch.cuda.synchronize(self.delta_device)
        return RowScore(loss=float(loss.item()), dots=[float(v) for v in dots.tolist()], fired=self.fired, seconds=time.time() - t0)


def compare_oracle(names: Sequence[str], hook_dots: Sequence[float], oracle_dots: Sequence[float], *, rel_tol: float, abs_floor_frac: float = 1e-4) -> dict[str, Any]:
    """Relative agreement per delta; deltas tiny relative to the row's largest
    |g| are judged on an absolute floor instead."""
    scale = max([abs(v) for v in oracle_dots] + [0.0])
    floor = abs_floor_frac * scale
    per: dict[str, dict[str, float | bool]] = {}
    worst = 0.0
    passed = True
    for name, a, b in zip(names, hook_dots, oracle_dots, strict=True):
        diff = abs(a - b)
        denominator = max(abs(a), abs(b))
        rel = diff / denominator if denominator > 0 else 0.0
        ok = rel <= rel_tol or diff <= floor
        per[name] = {"hook": a, "oracle": b, "abs_diff": diff, "rel": rel, "passed": ok}
        worst = max(worst, rel if not (diff <= floor) else 0.0)
        passed = passed and ok
    return {"passed": passed, "worst_rel": worst, "rel_tol": rel_tol, "abs_floor": floor, "per_delta": per}


# ----------------------------------------------------------------- records
def score_record(meta: RowMeta, *, n_tokens: int, n_target_tokens: int, loss: float | None, g: Mapping[str, float], seconds: float, mode: str, pass_name: str, repeat: int | None, graft: Mapping[str, Any] | None, loss_lam1: float | None = None) -> dict[str, Any]:
    """v1's row schema (``row_id, group, episode_id, subtype, n_target_tokens,
    loss, grad_norm, scores``) plus ``raw_dl_dlambda`` (= g) and ``loss_lam1``.
    ``loss`` is ALWAYS L(0), the -it row loss (in lam1 mode carried over from
    the λ = 0 pass; null when that row was not scored there); ``loss_lam1`` is
    the row loss under the graft (lam1 mode only). ``scores[name] = -g``
    (positive = graft lowers loss). ``grad_norm`` is not available without
    weight gradients and is null."""
    if not g:
        raise ValueError("g must not be empty")
    if mode == "lam1":
        if loss_lam1 is None or not math.isfinite(float(loss_lam1)):
            raise ValueError(f"lam1 record needs a finite loss_lam1 for {meta.row_id}")
        if loss is not None and not math.isfinite(float(loss)):
            raise ValueError(f"non-finite carried-over loss for {meta.row_id}")
    else:
        if loss_lam1 is not None:
            raise ValueError("loss_lam1 only applies to mode lam1")
        if loss is None or not math.isfinite(float(loss)):
            raise ValueError(f"non-finite loss for {meta.row_id}")
    scores: dict[str, float] = {}
    raw: dict[str, float] = {}
    for name, value in g.items():
        validate_delta_name(name)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite dL/dlambda for {name!r} on {meta.row_id}")
        raw[name] = value
        scores[name] = -value
    record: dict[str, Any] = {
        "row_id": meta.row_id,
        "group": meta.group,
        "episode_id": meta.episode_id,
        "subtype": meta.subtype,
        "n_target_tokens": int(n_target_tokens),
        "loss": None if loss is None else float(loss),
        "grad_norm": None,
        "scores": scores,
        "raw_dl_dlambda": raw,
        "loss_lam1": None if loss_lam1 is None else float(loss_lam1),
        "mode": mode,
        "pass": pass_name,
        "graft": None if graft is None else dict(graft),
        "row_index": meta.row_index,
        "n_tokens": int(n_tokens),
        "seconds": float(seconds),
    }
    if repeat is not None:
        record["repeat"] = int(repeat)
    missing = [key for key in REQUIRED_SCORE_KEYS if key not in record]
    assert not missing, missing
    return record


def record_key(record: Mapping[str, Any]) -> tuple[str, int | None]:
    return (str(record["row_id"]), record.get("repeat"))


def relative_difference(a: float, b: float, *, floor: float = 1e-30) -> float:
    return abs(a - b) / max(abs(a), abs(b), floor)


def repeat_noise_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Relative spread between repeats of the same row (G4), per delta name."""
    by_row: dict[str, dict[int, Mapping[str, Any]]] = {}
    for record in records:
        if record.get("repeat") is None:
            continue
        by_row.setdefault(str(record["row_id"]), {})[int(record["repeat"])] = record
    per_name: dict[str, list[float]] = {}
    loss_rel: list[float] = []
    n_pairs = 0
    for repeats in by_row.values():
        ordered = [repeats[k] for k in sorted(repeats)]
        for first, second in zip(ordered, ordered[1:], strict=False):
            n_pairs += 1
            loss_rel.append(relative_difference(first["loss"], second["loss"]))
            for name, value in first["scores"].items():
                if name in second["scores"]:
                    per_name.setdefault(name, []).append(relative_difference(value, second["scores"][name]))

    def stats(values: Sequence[float]) -> dict[str, float | int] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {"n": len(ordered), "median": statistics.median(ordered), "p90": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))], "max": ordered[-1]}

    return {
        "n_rows": len(by_row),
        "n_pairs": n_pairs,
        "loss": stats(loss_rel),
        "scores": {name: stats(values) for name, values in sorted(per_name.items())},
        "scores_all": stats([v for values in per_name.values() for v in values]),
    }


def build_schedule(config: ScoreConfig, rows: Sequence[RowMeta]) -> tuple[list[tuple[RowMeta, int | None]], dict[str, Any]]:
    selected = select_rows(rows, config.rows_filter, episode_seed=config.episode_seed)
    selection = selection_summary(rows, selected, config.rows_filter, config.episode_seed)
    if config.row_ids is not None:
        wanted = set(config.row_ids)
        selected = [row for row in selected if row.row_id in wanted]
        missing = wanted - {row.row_id for row in selected}
        if missing:
            raise ValueError(f"row_ids not found in the rows file / filter: {sorted(missing)[:10]}")
    ordered = order_rows(selected, config.row_order, config.groups)
    if config.row_limit is not None:
        ordered = ordered[: config.row_limit]
    repeats = [None] if config.repeats == 1 else list(range(config.repeats))
    schedule = [(row, repeat) for repeat in repeats for row in ordered]
    selection.update({"row_ids": None if config.row_ids is None else len(config.row_ids), "row_limit": config.row_limit, "row_order": config.row_order, "repeats": config.repeats, "n_rows_scheduled": len(ordered), "n_scores_scheduled": len(schedule)})
    return schedule, selection


# -------------------------------------------------------------------- run
def run(config: ScoreConfig) -> dict[str, Any]:
    import torch
    from scimt.data_attribution.losses import CausalLMLossAdapter

    started = time.time()
    tag = timestamp_tag()
    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir = config.evidence_path
    evidence_dir.mkdir(parents=True, exist_ok=True)
    receipt = Receipt(evidence_dir / f"score_lambda_grad__{config.pass_name}__{tag}.json", static={"pass": config.pass_name, "mode": config.mode, "config": config.to_dict(), "git_commit": git_commit()})
    log(f"score_lambda_grad pass={config.pass_name} mode={config.mode} deltas={[d.name for d in config.deltas]} graft={None if config.graft is None else asdict(config.graft)} rows={config.rows_path}")

    tokenizer = load_hf_tokenizer(config.tokenizer_snapshot or config.it_snapshot)
    t0 = time.time()
    rows = load_eft_rows(config.rows_path, config.groups)
    schedule, selection = build_schedule(config, rows)
    chat = ChatRows(config.rows_path, tokenizer, config.sequence_length, rows)
    rows_summary = chat.summary(rows)
    receipt.timings["render_rows_s"] = time.time() - t0
    log(f"{len(rows)} rows rendered ({json.dumps(rows_summary['per_group'])}); schedule {selection['n_scores_scheduled']} scores over {selection['n_rows_scheduled']} rows (rows_filter={config.rows_filter}, repeats={config.repeats})")

    t0 = time.time()
    model = load_hf_model(config.it_snapshot, dtype=config.dtype, device=config.model_device, gradient_checkpointing=config.gradient_checkpointing)
    covered = CoveredModel(model)
    receipt.timings["load_model_s"] = time.time() - t0
    log(f"model loaded in {receipt.timings['load_model_s']:.0f}s: {len(covered.linears)} covered linears over {covered.layers} layers, {len(covered.norms)} norm weights")
    if config.expected_linears is not None and len(covered.linears) != config.expected_linears:
        raise RuntimeError(f"{len(covered.linears)} covered linears, expected {config.expected_linears}")

    graft_record: dict[str, Any] | None = None
    if config.graft is not None:
        t0 = time.time()
        graft_record = merge_graft(covered, config.graft, device=config.model_device)
        receipt.timings["graft_merge_s"] = time.time() - t0
        log(f"graft merged: {json.dumps({k: v for k, v in graft_record.items() if k != 'adapter_manifest'})} in {receipt.timings['graft_merge_s']:.0f}s")

    t0 = time.time()
    resident = ResidentDeltas(config.deltas, device=config.delta_device, dtype=config.delta_dtype, module_paths=list(covered.linears), norm_keys=list(covered.norms))
    resident_summary = resident.load()
    receipt.timings["load_deltas_s"] = time.time() - t0
    engine = LambdaGradEngine(model, covered, resident, delta_device=config.delta_device, ensure_grad_flow=config.ensure_grad_flow)
    loss_adapter = CausalLMLossAdapter(model, reduction="per_sequence_sum", device=config.model_device)
    receipt.write("staged", resident=resident_summary, graft=graft_record, rows=rows_summary, selection=selection)

    base_losses: dict[str, float] | None = None
    if config.loss_lam0_path is not None:
        base_losses = {str(r["row_id"]): float(r["loss"]) for r in read_records(Path(config.loss_lam0_path)) if r.get("repeat") is None and r.get("loss") is not None}
        log(f"λ = 0 losses carried over for {len(base_losses)} rows from {config.loss_lam0_path}")
    missing_base = 0
    done: set[tuple[str, int | None]] = set()
    if config.resume:
        done = {record_key(r) for r in read_records(out_path)}
    elif out_path.exists():
        out_path.unlink()
    handle = out_path.open("a", encoding="utf-8")
    scored = skipped = 0
    row_seconds: list[float] = []
    oracle_results: list[dict[str, Any]] = []
    oracle_failed = False
    graft_short = None if graft_record is None else {"arm": graft_record["arm"], "lam": graft_record["lam"], "r": graft_record["r"], "kind": graft_record["kind"]}
    from scimt.data_attribution.gradients import backward_memory_mode

    memory_mode = backward_memory_mode(model, config.gradient_checkpointing)
    if torch.cuda.is_available():
        for device in {config.model_device, config.delta_device}:
            if device.startswith("cuda"):
                torch.zeros(1, device=device)  # initialise the allocator; the peak counters do not exist before the first allocation
                torch.cuda.synchronize(torch.device(device))
                torch.cuda.reset_peak_memory_stats(torch.device(device))
    loop_started = time.time()
    try:
        with memory_mode:
            for position, (row, repeat) in enumerate(schedule):
                if (row.row_id, repeat) in done:
                    skipped += 1
                    continue
                batch = chat.batch(row.row_index)
                result = engine.score(batch, loss_adapter)
                g = dict(zip(resident.names, result.dots, strict=True))
                if config.mode == "lam1":
                    base_loss = None if base_losses is None else base_losses.get(row.row_id)
                    missing_base += int(base_loss is None)
                    record = score_record(row, n_tokens=chat.lengths[row.row_index], n_target_tokens=chat.n_target_tokens[row.row_index], loss=base_loss, g=g, seconds=result.seconds, mode=config.mode, pass_name=config.pass_name, repeat=repeat, graft=graft_short, loss_lam1=result.loss)
                else:
                    record = score_record(row, n_tokens=chat.lengths[row.row_index], n_target_tokens=chat.n_target_tokens[row.row_index], loss=result.loss, g=g, seconds=result.seconds, mode=config.mode, pass_name=config.pass_name, repeat=repeat, graft=graft_short)
                if len(oracle_results) < config.oracle_rows and repeat in (None, 0):
                    oracle = engine.oracle(batch, loss_adapter)
                    comparison = compare_oracle(resident.names, result.dots, oracle.dots, rel_tol=config.oracle_rel_tol)
                    oracle_results.append({"row_id": row.row_id, "loss_hook": result.loss, "loss_oracle": oracle.loss, "seconds": oracle.seconds, **comparison})
                    oracle_failed = oracle_failed or not comparison["passed"]
                    log(f"oracle {row.row_id}: worst rel {comparison['worst_rel']:.3e} (tol {config.oracle_rel_tol:g}) {'PASS' if comparison['passed'] else 'FAIL'}; loss {result.loss:.4f} vs {oracle.loss:.4f}")
                    record["oracle"] = {"passed": comparison["passed"], "worst_rel": comparison["worst_rel"]}
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                scored += 1
                row_seconds.append(result.seconds)
                if scored % config.progress_every == 0 or scored == 1 or position == len(schedule) - 1:
                    elapsed = time.time() - loop_started
                    rate = scored / max(elapsed, 1e-9)
                    remaining = len(schedule) - position - 1
                    first = resident.names[0]
                    log(
                        f"row {position + 1}/{len(schedule)} {row.row_id}{'' if repeat is None else f' rep{repeat}'} loss={result.loss:.3f} "
                        f"g[{first}]={g[first]:+.3e} {result.seconds:.2f}s | {rate:.2f} rows/s ETA {remaining / max(rate, 1e-9) / 60:.0f} min | "
                        f"peak model {cuda_peak_gb(config.model_device) or 0:.1f} GB delta {cuda_peak_gb(config.delta_device) or 0:.1f} GB"
                    )
    finally:
        handle.close()
        engine.remove()
    receipt.timings["rows_total_s"] = time.time() - loop_started
    receipt.timings["total_s"] = time.time() - started
    rate = scored / max(sum(row_seconds), 1e-9) if row_seconds else None
    records = read_records(out_path)
    summary: dict[str, Any] = {
        "out_path": str(out_path),
        "rows_path": config.rows_path,
        "rows_sha256": sha256_file(config.rows_path),
        "rows": rows_summary,
        "selection": selection,
        "delta_names": list(resident.names),
        "resident": resident_summary,
        "graft": graft_record,
        "n_scheduled": len(schedule),
        "scored_rows": scored,
        "skipped_rows_resume": skipped,
        "loss_lam0_path": config.loss_lam0_path,
        "rows_missing_lam0_loss": missing_base,
        "n_records_in_file": len(records),
        "rows_per_s": rate,
        "row_seconds_median": statistics.median(row_seconds) if row_seconds else None,
        "row_seconds_p90": (sorted(row_seconds)[int(0.9 * (len(row_seconds) - 1))] if row_seconds else None),
        "peak_gpu_allocated_gb": {"model": cuda_peak_gb(config.model_device), "delta": cuda_peak_gb(config.delta_device)},
        "oracle": {"n_rows": len(oracle_results), "passed": not oracle_failed, "worst_rel": max([o["worst_rel"] for o in oracle_results], default=None), "rows": oracle_results},
    }
    if config.repeats > 1:
        summary["repeat_noise"] = repeat_noise_summary(records)
        log(f"repeat noise: {json.dumps(summary['repeat_noise']['scores_all'])}")
    if missing_base:
        log(f"WARNING: {missing_base} lam1 rows had no λ = 0 loss in {config.loss_lam0_path} (loss written as null)")
    write_json(out_path.with_name(out_path.stem + "_manifest.json"), {"pass": config.pass_name, "mode": config.mode, "config": config.to_dict(), "git_commit": git_commit(), "created_at": now_iso(), "timings_s": dict(receipt.timings), **{k: v for k, v in summary.items() if k != "oracle"}, "oracle": {k: v for k, v in summary["oracle"].items() if k != "rows"}})
    status = "oracle-failed" if (oracle_failed and config.oracle_strict) else "ok"
    receipt.write(status, **summary)
    if status != "ok":
        log(f"SCIMT-ORACLE-FAIL worst rel {summary['oracle']['worst_rel']:.3e} > {config.oracle_rel_tol:g}")
        raise SystemExit(ORACLE_EXIT)
    log(f"SCIMT-SCORE-DONE pass={config.pass_name}: {scored} rows in {receipt.timings['rows_total_s'] / 60:.1f} min ({'n/a' if rate is None else f'{rate:.3f}'} rows/s); receipt {receipt.path}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    config = load_config(argv, CONFIG_ENV, "score_lambda_grad.py", DEFAULTS, ScoreConfig.from_mapping)
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
