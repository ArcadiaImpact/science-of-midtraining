"""Delta + graft-LoRA extraction for one arm (SPEC §3).

Streams ``theta_pt`` and ``theta_mid`` shard by shard (``safetensors.safe_open``
— one tensor of each in RAM at a time, never two state dicts), forms
``delta = mid - pt`` in fp32 on the GPU for every tensor present in both, and

- **decoder linears** (q/k/v/o, gate/up/down under
  ``model.language_model.layers.*``): thin SVD -> for each rank r the factors
  ``B = U_r sqrt(S_r)`` (out x r), ``A = sqrt(S_r) V_r^T`` (r x in), logged
  with ``||delta||_F``, ``||delta||_F / ||W_pt||_F``, the top-8 singular
  values and the captured energy per rank; optionally the full delta as
  bf16 shards for the exactness reference;
- **norm weights** (every ``*norm.weight`` under the language model): kept
  full-rank in every variant (``norm_delta.safetensors``);
- **embeddings / lm_head**: delta norm recorded, EXCLUDED from the graft;
- **vision tower / projector**: asserted ``delta == 0`` (max |delta| logged;
  fail loudly above ``vision_delta_tol`` unless ``allow_vision_delta``).

Safetensors keys are canonicalised to the transformers-5 layout first
(``common.canonical_key``), so an old-layout base (unsloth's mirror:
``language_model.model.*``) diffs cleanly against a checkpoint saved by a
current trainer.

SVD: ``svd_method`` ``full`` (``torch.linalg.svd``), ``gram`` (exact thin
SVD via the fp64 Gram eigendecomposition) or ``lowrank``
(``torch.svd_lowrank(q = r_max + lowrank_extra, niter)``); ``auto`` (default)
times the first full SVD of each weight shape and switches that shape to the
lowrank fallback when it exceeds ``svd_time_budget_s`` (a running SVD cannot
be aborted, so the budget is enforced per shape from the second module on).

Outputs under ``<out_dir>/<arm>/``: ``r<r>/`` PEFT-style adapter dirs
(``adapter_model.safetensors`` with ``base_model.model.<module>.lora_{A,B}.weight``,
``adapter_config.json`` with ``lora_alpha = r`` so scaling = 1, the norm
deltas, ``manifest.json``), ``full/`` (sharded bf16 delta + index + norm
deltas, when ``save_full``), ``manifest.json`` (arm-level, ``status:
complete``); ``<evidence_dir>/delta_stats__<arm>.json`` and a receipt.
Config-first (``argv[1]`` or ``$SCIMT_GRAFT_EXTRACT_CONFIG`` over DEFAULTS).
"""

from __future__ import annotations

import math
import re
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

from experiments.improved_midtraining.graft_delta_lambda_v1.pod.common import (  # noqa: E402
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    ARMS,
    DEFAULT_RANKS,
    FULL_INDEX_FILE,
    GB,
    MANIFEST_FILE,
    NORM_DELTA_FILE,
    PEFT_TARGET_MODULES_REGEX,
    Coverage,
    FullDeltaWriter,
    Receipt,
    canonical_key,
    captured_energy,
    classify,
    cuda_peak_gb,
    git_commit,
    int_or_none,
    linear_sort_key,
    load_config,
    log,
    now_iso,
    peft_adapter_config,
    read_json,
    reject_unknown,
    require_bool,
    require_device,
    require_float,
    require_int,
    require_ranks,
    require_str,
    str_or_none,
    svd_lora_factors,
    thin_svd,
    timestamp_tag,
    write_json,
    write_lora_adapter,
)

CONFIG_ENV = "SCIMT_GRAFT_EXTRACT_CONFIG"
SVD_METHODS = ("auto", "full", "gram", "lowrank")
DTYPES = ("bfloat16", "float16", "float32")
VISION_MISMATCH_EXIT = 94
PAIR_MISMATCH_EXIT = 95


# ------------------------------------------------------------------- config
@dataclass(frozen=True)
class ExtractConfig:
    pt_snapshot: str
    mid_snapshot: str
    arm: str
    out_dir: str
    ranks: tuple[int, ...] = DEFAULT_RANKS
    device: str = "cuda:0"
    svd_method: str = "auto"
    svd_time_budget_s: float = 60.0
    lowrank_extra: int = 128
    lowrank_niter: int = 4
    include_embeddings: bool = False
    allow_vision_delta: bool = False
    vision_delta_tol: float = 1e-6
    save_full: bool = True
    full_shard_gb: float = 8.0
    factor_dtype: str = "bfloat16"
    full_dtype: str = "bfloat16"
    base_model_name: str = "google/gemma-3-27b-pt"
    evidence_dir: str | None = None  # None -> <out_dir>/evidence
    layer_limit: int | None = None  # smoke only: first N layers (partial coverage, recorded)
    expected_layers: int | None = 62
    resume: bool = True

    _KEYS = (
        "pt_snapshot", "mid_snapshot", "arm", "out_dir", "ranks", "device", "svd_method",
        "svd_time_budget_s", "lowrank_extra", "lowrank_niter", "include_embeddings",
        "allow_vision_delta", "vision_delta_tol", "save_full", "full_shard_gb", "factor_dtype",
        "full_dtype", "base_model_name", "evidence_dir", "layer_limit", "expected_layers", "resume",
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExtractConfig:
        reject_unknown(raw, cls._KEYS, "extract_delta_lora")
        arm = require_str(raw.get("arm"), "arm")
        if not re.fullmatch(r"[a-z][a-z0-9]*", arm):
            raise ValueError(f"arm must be a bare lowercase token (score names are <arm>__<kind>__all), got {arm!r}")
        method = require_str(raw.get("svd_method", "auto"), "svd_method")
        if method not in SVD_METHODS:
            raise ValueError(f"svd_method must be one of {SVD_METHODS}, got {method!r}")
        for label in ("factor_dtype", "full_dtype"):
            if require_str(raw.get(label, "bfloat16"), label) not in DTYPES:
                raise ValueError(f"{label} must be one of {DTYPES}")
        if raw.get("include_embeddings", False):
            raise ValueError("include_embeddings=true is not implemented: embeddings/lm_head are recorded and excluded (SPEC §3)")
        return cls(
            pt_snapshot=require_str(raw.get("pt_snapshot"), "pt_snapshot"),
            mid_snapshot=require_str(raw.get("mid_snapshot"), "mid_snapshot"),
            arm=arm,
            out_dir=require_str(raw.get("out_dir"), "out_dir"),
            ranks=require_ranks(raw.get("ranks", list(DEFAULT_RANKS))),
            device=require_device(raw.get("device", "cuda:0"), "device"),
            svd_method=method,
            svd_time_budget_s=require_float(raw.get("svd_time_budget_s", 60.0), "svd_time_budget_s", minimum=0.0, minimum_exclusive=True),
            lowrank_extra=require_int(raw.get("lowrank_extra", 128), "lowrank_extra", minimum=0),
            lowrank_niter=require_int(raw.get("lowrank_niter", 4), "lowrank_niter", minimum=1),
            include_embeddings=False,
            allow_vision_delta=require_bool(raw.get("allow_vision_delta", False), "allow_vision_delta"),
            vision_delta_tol=require_float(raw.get("vision_delta_tol", 1e-6), "vision_delta_tol", minimum=0.0),
            save_full=require_bool(raw.get("save_full", True), "save_full"),
            full_shard_gb=require_float(raw.get("full_shard_gb", 8.0), "full_shard_gb", minimum=0.0, minimum_exclusive=True),
            factor_dtype=str(raw.get("factor_dtype", "bfloat16")),
            full_dtype=str(raw.get("full_dtype", "bfloat16")),
            base_model_name=require_str(raw.get("base_model_name", "google/gemma-3-27b-pt"), "base_model_name"),
            evidence_dir=str_or_none(raw.get("evidence_dir"), "evidence_dir"),
            layer_limit=int_or_none(raw.get("layer_limit"), "layer_limit"),
            expected_layers=int_or_none(raw.get("expected_layers", 62), "expected_layers"),
            resume=require_bool(raw.get("resume", True), "resume"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ranks"] = list(self.ranks)
        return payload

    @property
    def arm_dir(self) -> Path:
        return Path(self.out_dir) / self.arm

    @property
    def evidence_path(self) -> Path:
        return Path(self.evidence_dir) if self.evidence_dir else Path(self.out_dir) / "evidence"


DEFAULTS: dict[str, Any] = {
    "pt_snapshot": None,
    "mid_snapshot": None,
    "arm": None,
    "out_dir": "/workspace/graft/adapters",
    "ranks": list(DEFAULT_RANKS),
    "device": "cuda:0",
    "svd_method": "auto",
    "svd_time_budget_s": 60.0,
    "save_full": True,
}


# ------------------------------------------------------------ safetensors
class SafetensorsIndex:
    """Canonical key -> (shard file, original key) over every ``*.safetensors``
    of a snapshot / checkpoint dir; tensors load one at a time (mmap)."""

    def __init__(self, directory: str | Path) -> None:
        from safetensors import safe_open

        self.directory = Path(directory)
        self.files = sorted(p for p in self.directory.glob("*.safetensors") if p.is_file())
        if not self.files:
            raise FileNotFoundError(f"{self.directory}: no *.safetensors files")
        self.entries: dict[str, tuple[Path, str]] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, str] = {}
        self._safe_open = safe_open
        self._handles: dict[Path, Any] = {}
        for path in self.files:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    canon = canonical_key(key)
                    if canon in self.entries:
                        raise ValueError(f"{self.directory}: {key!r} and {self.entries[canon][1]!r} both canonicalise to {canon!r}")
                    self.entries[canon] = (path, key)
                    view = handle.get_slice(key)
                    self.shapes[canon] = tuple(int(x) for x in view.get_shape())
                    self.dtypes[canon] = str(view.get_dtype())

    def __contains__(self, canon: str) -> bool:
        return canon in self.entries

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, canon: str, device: Any = "cpu"):
        path, key = self.entries[canon]
        handle = self._handles.get(path)
        if handle is None:
            handle = self._safe_open(str(path), framework="pt", device="cpu").__enter__()
            self._handles[path] = handle
        return handle.get_tensor(key).to(device)

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 — best effort
                pass
        self._handles.clear()

    def summary(self) -> dict[str, Any]:
        return {"directory": str(self.directory), "files": [p.name for p in self.files], "n_tensors": len(self.entries), "bytes": sum(p.stat().st_size for p in self.files)}


def _fmt(value: float | None, spec: str = ".3e") -> str:
    """Format a ratio that may be ``None`` (zero-norm reference tensor)."""
    return "n/a" if value is None else format(value, spec)


def snapshot_sha(path: str | Path) -> str | None:
    """The ``snapshots/<40-hex>`` ancestor of an HF-cache path, if any."""
    for part in reversed(Path(path).resolve().parts):
        if re.fullmatch(r"[0-9a-f]{40}", part):
            return part
    return None


# ------------------------------------------------------------- svd policy
class SvdPolicy:
    """``auto``: full SVD until a weight shape exceeds the budget, then the
    lowrank fallback for that shape."""

    def __init__(self, method: str, budget_s: float, *, fallback: str = "lowrank") -> None:
        self.method = method
        self.budget_s = float(budget_s)
        self.fallback = fallback
        self.slow_shapes: set[tuple[int, ...]] = set()
        self.timings: list[dict[str, Any]] = []

    def method_for(self, shape: Sequence[int]) -> str:
        if self.method != "auto":
            return self.method
        return self.fallback if tuple(shape) in self.slow_shapes else "full"

    def record(self, shape: Sequence[int], method: str, seconds: float, module_path: str) -> None:
        self.timings.append({"module": module_path, "shape": list(shape), "method": method, "seconds": seconds})
        if self.method == "auto" and method == "full" and seconds > self.budget_s:
            self.slow_shapes.add(tuple(shape))
            log(f"svd: shape {tuple(shape)} took {seconds:.1f}s > budget {self.budget_s:.0f}s; using {self.fallback} for that shape from now on")

    def summary(self) -> dict[str, Any]:
        by_method: dict[str, list[float]] = {}
        for entry in self.timings:
            by_method.setdefault(entry["method"], []).append(entry["seconds"])
        return {
            "method": self.method,
            "budget_s": self.budget_s,
            "fallback": self.fallback,
            "slow_shapes": [list(s) for s in sorted(self.slow_shapes)],
            "n_modules_by_method": {k: len(v) for k, v in by_method.items()},
            "seconds_by_method": {k: sum(v) for k, v in by_method.items()},
        }


# ------------------------------------------------------------------ stats
def pooled_stats(rows: Sequence[Mapping[str, Any]], ranks: Sequence[int], *, norm_delta_fro: float = 0.0) -> dict[str, Any]:
    """Share of ``||delta||^2`` per module type / layer, pooled captured energy
    per rank (sum of captured ||.||^2 over sum of ||delta||^2 — the
    energy-weighted mean), and the Frobenius norm of every graft variant
    (rank-r reconstruction of the linears + the full-rank norm deltas) for
    ``scores/vector_norms.json``. Rows carry ``energy`` (fractions) and
    ``energy_abs`` (captured ||.||^2) per rank."""
    total = sum(row["delta_fro"] ** 2 for row in rows)
    by_type: dict[str, dict[str, Any]] = {}
    by_layer: dict[int, float] = {}
    pooled_abs = {r: 0.0 for r in ranks}
    for row in rows:
        d2 = row["delta_fro"] ** 2
        entry = by_type.setdefault(row["module_type"], {"delta_sq": 0.0, "w_sq": 0.0, "n": 0, "energy_abs": {r: 0.0 for r in ranks}})
        entry["delta_sq"] += d2
        entry["w_sq"] += row["w_fro"] ** 2
        entry["n"] += 1
        for r in ranks:
            captured = row["energy_abs"][str(r)] if "energy_abs" in row else row["energy"][str(r)] * d2
            entry["energy_abs"][r] += captured
            pooled_abs[r] += captured
        by_layer[row["layer"]] = by_layer.get(row["layer"], 0.0) + d2
    captured_energy = {str(r): (pooled_abs[r] / total if total else None) for r in ranks}
    per_module_type = {
        name: {
            "n": e["n"],
            "fro_norm": math.sqrt(e["delta_sq"]),
            "fro_norm_share": (e["delta_sq"] / total) if total else None,  # share of ||delta||_F^2 (SPEC §3.2)
            "ratio_to_w": math.sqrt(e["delta_sq"] / e["w_sq"]) if e["w_sq"] else None,
            "captured_energy": {str(r): (e["energy_abs"][r] / e["delta_sq"] if e["delta_sq"] else None) for r in ranks},
        }
        for name, e in sorted(by_type.items())
    }
    norm_sq = float(norm_delta_fro) ** 2
    delta_norms = {f"r{r}": math.sqrt(pooled_abs[r] + norm_sq) for r in ranks}
    delta_norms["full"] = math.sqrt(total + norm_sq)
    return {
        "n_modules": len(rows),
        "delta_fro_linears": math.sqrt(total),
        "delta_fro_norms": float(norm_delta_fro),
        "captured_energy": captured_energy,
        "energy_pooled": captured_energy,  # alias kept for the manifests
        "captured_energy_abs": {str(r): pooled_abs[r] for r in ranks},
        "delta_norms": delta_norms,
        "per_module_type": per_module_type,
        "by_type": per_module_type,
        "by_layer_share": {str(layer): (d2 / total if total else None) for layer, d2 in sorted(by_layer.items())},
    }


def arm_complete(arm_dir: Path, ranks: Sequence[int], save_full: bool) -> bool:
    manifest = arm_dir / MANIFEST_FILE
    if not manifest.is_file():
        return False
    try:
        body = read_json(manifest)
    except (OSError, ValueError):
        return False
    if body.get("status") != "complete" or sorted(body.get("ranks", [])) != sorted(ranks):
        return False
    for r in ranks:
        rank_dir = arm_dir / f"r{r}"
        if not all((rank_dir / name).is_file() for name in (ADAPTER_WEIGHTS, ADAPTER_CONFIG, NORM_DELTA_FILE, MANIFEST_FILE)):
            return False
    if save_full and not (arm_dir / "full" / FULL_INDEX_FILE).is_file():
        return False
    return True


# -------------------------------------------------------------------- run
def run(config: ExtractConfig) -> dict[str, Any]:
    import torch

    started = time.time()
    tag = timestamp_tag()
    arm_dir = config.arm_dir
    evidence_dir = config.evidence_path
    evidence_dir.mkdir(parents=True, exist_ok=True)
    receipt = Receipt(evidence_dir / f"extract_delta_lora__{config.arm}__{tag}.json", static={"arm": config.arm, "config": config.to_dict(), "git_commit": git_commit()})
    log(f"extract_delta_lora arm={config.arm} pt={config.pt_snapshot} mid={config.mid_snapshot} ranks={list(config.ranks)} device={config.device}")

    if config.resume and arm_complete(arm_dir, config.ranks, config.save_full):
        body = read_json(arm_dir / MANIFEST_FILE)
        log(f"resume: {arm_dir} already complete (created {body.get('created_at')}); nothing to do")
        receipt.write("ok", resumed=True, arm_dir=str(arm_dir), manifest=body)
        return {"status": "ok", "resumed": True, "arm_dir": str(arm_dir)}

    device = torch.device(config.device)
    factor_dtype = getattr(torch, config.factor_dtype)
    full_dtype = getattr(torch, config.full_dtype)
    t0 = time.time()
    pt = SafetensorsIndex(config.pt_snapshot)
    mid = SafetensorsIndex(config.mid_snapshot)
    receipt.timings["index_s"] = time.time() - t0
    common = sorted(set(pt) & set(mid))
    only_pt = sorted(set(pt) - set(mid))
    only_mid = sorted(set(mid) - set(pt))
    log(f"index: pt {len(pt)} tensors in {len(pt.files)} files; mid {len(mid)} tensors in {len(mid.files)} files; common {len(common)}; only-pt {len(only_pt)}; only-mid {len(only_mid)}")
    # Asymmetric key sets are fine for excluded tensors only: the midtrain save
    # materialises the tied ``lm_head.weight`` (1,248 vs 1,247 tensors); a
    # covered linear / norm or a vision tensor on one side only is a mismatch.
    asymmetric = {"only_pt": [{"key": k, "kind": classify(k).kind} for k in only_pt], "only_mid": [{"key": k, "kind": classify(k).kind} for k in only_mid]}
    for side, rows in asymmetric.items():
        for row in rows:
            note = " (tied embeddings materialised by the trainer save; excluded like embed_tokens)" if row["key"] == "lm_head.weight" else ""
            log(f"note: {row['key']} present {side.replace('_', ' in ')} only, kind={row['kind']}{note}")
    covered_missing = [k for k in only_pt + only_mid if classify(k).kind in ("linear", "norm", "vision")]
    if covered_missing:
        receipt.write("failed", reason="covered tensors present in only one checkpoint", missing=covered_missing[:50])
        log(f"SCIMT-EXTRACT-FAIL pair mismatch: {len(covered_missing)} covered tensors in only one checkpoint, e.g. {covered_missing[:5]}")
        raise SystemExit(PAIR_MISMATCH_EXIT)
    for key in common:
        if pt.shapes[key] != mid.shapes[key]:
            receipt.write("failed", reason="shape mismatch", key=key, pt=list(pt.shapes[key]), mid=list(mid.shapes[key]))
            log(f"SCIMT-EXTRACT-FAIL shape mismatch {key}: pt {pt.shapes[key]} vs mid {mid.shapes[key]}")
            raise SystemExit(PAIR_MISMATCH_EXIT)

    groups: dict[str, list[str]] = {"linear": [], "norm": [], "embedding": [], "vision": [], "other": []}
    coverage: dict[str, Coverage] = {}
    for key in common:
        cov = classify(key)
        coverage[key] = cov
        groups[cov.kind].append(key)
    if config.layer_limit is not None:
        keep = lambda key: coverage[key].layer is None or coverage[key].layer < config.layer_limit  # noqa: E731
        groups["linear"] = [k for k in groups["linear"] if keep(k)]
        groups["norm"] = [k for k in groups["norm"] if keep(k)]
    groups["linear"].sort(key=lambda key: linear_sort_key(key[: -len(".weight")]))
    groups["norm"].sort()
    layers = sorted({coverage[k].layer for k in groups["linear"] if coverage[k].layer is not None})
    if config.expected_layers is not None and config.layer_limit is None and len(layers) != config.expected_layers:
        raise RuntimeError(f"found {len(layers)} decoder layers with covered linears, expected {config.expected_layers}")
    if len(groups["linear"]) != len(layers) * 7:
        raise RuntimeError(f"{len(groups['linear'])} covered linears for {len(layers)} layers — expected 7 per layer")
    if groups["other"]:
        log(f"WARNING: {len(groups['other'])} unclassified tensors excluded from the graft, e.g. {groups['other'][:5]}")
    log(f"coverage: {len(groups['linear'])} linears over {len(layers)} layers, {len(groups['norm'])} norms, {len(groups['embedding'])} embeddings (excluded), {len(groups['vision'])} vision tensors (asserted zero), {len(groups['other'])} other")
    receipt.write("running", coverage={k: len(v) for k, v in groups.items()}, layers=len(layers))

    # ---- vision tower / projector: delta must be zero ----------------------
    t0 = time.time()
    vision_stats: list[dict[str, Any]] = []
    vision_max = 0.0
    for key in groups["vision"]:
        a = pt.get(key, device)
        b = mid.get(key, device)
        mx = float((b.float() - a.float()).abs().max().item()) if a.numel() else 0.0
        vision_max = max(vision_max, mx)
        if mx > config.vision_delta_tol:
            vision_stats.append({"key": key, "max_abs_delta": mx})
        del a, b
    receipt.timings["vision_check_s"] = time.time() - t0
    log(f"vision: {len(groups['vision'])} tensors, max |delta| = {vision_max:.3e} ({len(vision_stats)} above tol {config.vision_delta_tol:g})")
    if vision_stats and not config.allow_vision_delta:
        receipt.write("failed", reason="vision tower / projector delta is not zero", vision_max_abs_delta=vision_max, offenders=vision_stats[:20])
        log(f"SCIMT-EXTRACT-FAIL vision delta {vision_max:.3e} > {config.vision_delta_tol:g}: the midtrain touched the vision tower or the pair is mismatched")
        raise SystemExit(VISION_MISMATCH_EXIT)

    # ---- embeddings / lm_head: recorded, excluded --------------------------
    t0 = time.time()
    embedding_stats: list[dict[str, Any]] = []
    for key in groups["embedding"]:
        a = pt.get(key, device)
        b = mid.get(key, device)
        d2 = w2 = 0.0
        for start in range(0, a.shape[0], 16384):
            da = a[start : start + 16384].float()
            db = b[start : start + 16384].float()
            d2 += float(((db - da) ** 2).sum().item())
            w2 += float((da**2).sum().item())
        embedding_stats.append({"key": key, "shape": list(a.shape), "delta_fro": math.sqrt(d2), "w_fro": math.sqrt(w2), "ratio": (math.sqrt(d2 / w2) if w2 else None), "excluded": True})
        del a, b
    receipt.timings["embeddings_s"] = time.time() - t0
    for row in embedding_stats:
        log(f"embedding {row['key']}: ||delta||={row['delta_fro']:.4e} ||W||={row['w_fro']:.4e} ratio={_fmt(row['ratio'])} (excluded)")

    # ---- norms: full-rank deltas (fp32; may be exactly zero) -------------------
    t0 = time.time()
    norm_deltas: dict[str, Any] = {}
    norm_stats: list[dict[str, Any]] = []
    for key in groups["norm"]:
        a = pt.get(key, device).float()
        b = mid.get(key, device).float()
        delta = b - a
        delta_fro = float(delta.norm().item())
        norm_deltas[key] = delta.to(factor_dtype).cpu()  # written even when zero: schema stability for the scorer / merges
        norm_stats.append({"key": key, "layer": coverage[key].layer, "numel": int(delta.numel()), "delta_fro": delta_fro, "w_fro": float(a.norm().item()), "zero": delta_fro == 0.0})
        del a, b, delta
    receipt.timings["norms_s"] = time.time() - t0
    norm_delta_fro = math.sqrt(sum(row["delta_fro"] ** 2 for row in norm_stats))
    norm_zero_count = sum(1 for row in norm_stats if row["zero"])
    norm_all_zero = bool(norm_stats) and norm_zero_count == len(norm_stats)
    log(f"norms: {len(norm_stats)} tensors, {norm_zero_count} exactly zero, ||delta_norm||_F = {norm_delta_fro:.4e}" + (" — all norm deltas are zero (midtrain did not move the norms); the norm term is a no-op downstream" if norm_all_zero else ""))

    # ---- linears: SVD -> factors per rank, full delta shards ------------------
    policy = SvdPolicy(config.svd_method, config.svd_time_budget_s)
    factors: dict[int, dict[str, tuple[Any, Any]]] = {r: {} for r in config.ranks}
    writer = FullDeltaWriter(arm_dir / "full", shard_bytes=int(config.full_shard_gb * GB)) if config.save_full else None
    linear_stats: list[dict[str, Any]] = []
    zero_linears: list[str] = []
    max_rank = max(config.ranks)
    t_linears = time.time()
    if device.type == "cuda":
        torch.zeros(1, device=device)  # initialise the allocator; the peak counters do not exist before the first allocation
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    for index, key in enumerate(groups["linear"]):
        module_path = key[: -len(".weight")]
        t_mod = time.time()
        W = pt.get(key, device)
        M = mid.get(key, device)
        delta = M.float() - W.float()  # upcast first: bf16 subtraction would lose most of a ~1e-3 delta on O(1e-2) weights
        w_fro = float(W.float().norm().item())
        delta_fro = float(delta.norm().item())
        fro2 = delta_fro**2
        if fro2 == 0.0:
            zero_linears.append(module_path)
            log(f"WARNING: {module_path}: delta is exactly zero (energy fractions reported as 1.0)")
        del M
        method = policy.method_for(delta.shape)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_svd = time.time()
        U, S, Vh = thin_svd(delta, method, q=max_rank + config.lowrank_extra, niter=config.lowrank_niter)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        svd_seconds = time.time() - t_svd
        policy.record(delta.shape, method, svd_seconds, module_path)
        facs = svd_lora_factors(delta, config.ranks, svd=(U, S, Vh))
        for r, (A, B) in facs.items():
            factors[r][module_path] = (A.to(factor_dtype).cpu(), B.to(factor_dtype).cpu())
        energy = {str(r): captured_energy(S, r, fro2) for r in config.ranks}
        energy_abs = {str(r): float((S[:r].double() ** 2).sum().item()) for r in config.ranks}
        row = {
            "module_path": module_path,
            "layer": coverage[key].layer,
            "module_type": coverage[key].module_type,
            "shape": list(delta.shape),
            "delta_fro": delta_fro,
            "w_fro": w_fro,
            "ratio": (delta_fro / w_fro) if w_fro else None,
            "top_singular_values": [float(x) for x in S[:8].tolist()],
            "energy": energy,
            "energy_abs": energy_abs,
            "svd_method": method,
            "svd_seconds": svd_seconds,
            "svd_components": int(S.numel()),
        }
        linear_stats.append(row)
        if writer is not None:
            writer.add(module_path, delta.to(full_dtype))
        del W, delta, U, S, Vh, facs
        elapsed = time.time() - t_linears
        rate = (index + 1) / max(elapsed, 1e-9)
        if index < 8 or (index + 1) % 7 == 0 or index + 1 == len(groups["linear"]):
            log(
                f"linear {index + 1}/{len(groups['linear'])} {module_path} shape={row['shape']} ||delta||={delta_fro:.3e} ratio={_fmt(row['ratio'], '.2e')} "
                f"E={' '.join(f'r{r}:{energy[str(r)]:.3f}' for r in config.ranks)} svd={method} {svd_seconds:.1f}s | {time.time() - t_mod:.1f}s/module ETA {(len(groups['linear']) - index - 1) / max(rate, 1e-9) / 60:.0f} min"
            )
    receipt.timings["linears_s"] = time.time() - t_linears
    if writer is not None:
        t0 = time.time()
        full_index = writer.close(extra={"arm": config.arm, "dtype": config.full_dtype, "created_at": now_iso()})
        from safetensors.torch import save_file

        save_file(dict(norm_deltas), str(arm_dir / "full" / NORM_DELTA_FILE), metadata={"format": "pt"})
        receipt.timings["full_write_s"] = time.time() - t0
        log(f"full delta: {len(writer.shards)} shards -> {full_index}")

    pooled = pooled_stats(linear_stats, config.ranks, norm_delta_fro=norm_delta_fro)
    stats_payload = {
        "arm": config.arm,
        "pt_snapshot": {"path": config.pt_snapshot, "sha": snapshot_sha(config.pt_snapshot), **pt.summary()},
        "mid_snapshot": {"path": config.mid_snapshot, "sha": snapshot_sha(config.mid_snapshot), **mid.summary()},
        "ranks": list(config.ranks),
        "layers": layers,
        "linears": linear_stats,
        "pooled": pooled,
        "per_module_type": pooled["per_module_type"],
        "delta_norms": pooled["delta_norms"],
        "norms": {"tensors": norm_stats, "delta_fro": norm_delta_fro, "n_zero": norm_zero_count, "all_zero": norm_all_zero},
        "zero_delta_linears": zero_linears,
        "embeddings_excluded": embedding_stats,
        "vision": {"n_tensors": len(groups["vision"]), "max_abs_delta": vision_max, "above_tol": vision_stats},
        "other_excluded": groups["other"],
        "only_pt": only_pt,
        "only_mid": only_mid,
        "asymmetric_keys": asymmetric,
        "svd": policy.summary(),
        "svd_timings": policy.timings,
        "created_at": now_iso(),
        "git_commit": git_commit(),
    }
    stats_path = write_json(evidence_dir / f"delta_stats__{config.arm}.json", stats_payload)
    log(f"pooled captured energy {pooled['captured_energy']}; ||delta_r||_F {pooled['delta_norms']}; share of ||delta||^2 by type: " + ", ".join(f"{t}:{v['fro_norm_share']:.3f}" for t, v in pooled["per_module_type"].items()))

    # ---- adapters ---------------------------------------------------------------
    t0 = time.time()
    coverage_summary = {
        "linears": len(groups["linear"]),
        "norms": len(groups["norm"]),
        "layers": len(layers),
        "layer_limit": config.layer_limit,
        "embeddings_excluded": [row["key"] for row in embedding_stats],
        "asymmetric_keys": asymmetric,
        "norm_deltas_zero": norm_zero_count,
        "norm_deltas_all_zero": norm_all_zero,
        "zero_delta_linears": zero_linears,
        "vision_tensors_checked": len(groups["vision"]),
        "vision_max_abs_delta": vision_max,
        "other_excluded": groups["other"],
        "target_modules_regex": PEFT_TARGET_MODULES_REGEX,
    }
    base_manifest = {
        "arm": config.arm,
        "ranks": list(config.ranks),
        "pt_snapshot": stats_payload["pt_snapshot"],
        "mid_snapshot": stats_payload["mid_snapshot"],
        "coverage": coverage_summary,
        "energy_pooled": pooled["energy_pooled"],
        "captured_energy": pooled["captured_energy"],
        "delta_norms": pooled["delta_norms"],
        "per_module_type": pooled["per_module_type"],
        "delta_fro_linears": pooled["delta_fro_linears"],
        "delta_fro_norms": norm_delta_fro,
        "factor_dtype": config.factor_dtype,
        "full_dtype": config.full_dtype if config.save_full else None,
        "svd": policy.summary(),
        "peft_naming": f"{ADAPTER_WEIGHTS}: base_model.model.<module_path>.lora_A.weight (r x in), .lora_B.weight (out x r); lora_alpha = r so scaling = 1; norm deltas full-rank in {NORM_DELTA_FILE}",
        "stats_file": str(stats_path),
        "config": config.to_dict(),
        "git_commit": git_commit(),
    }
    rank_dirs: dict[str, str] = {}
    for r in config.ranks:
        rank_dir = arm_dir / f"r{r}"
        manifest = {**base_manifest, "rank": r, "energy_captured": pooled["energy_pooled"][str(r)], "created_at": now_iso()}
        write_lora_adapter(rank_dir, r, factors[r], norm_deltas, config=peft_adapter_config(r, base_model=config.base_model_name), manifest=manifest)
        rank_dirs[f"r{r}"] = str(rank_dir)
        n_params = sum(A.numel() + B.numel() for A, B in factors[r].values())
        log(f"adapter r={r}: {len(factors[r])} modules, {n_params / 1e9:.3f} B factor params -> {rank_dir}")
    if config.save_full:
        write_json(arm_dir / "full" / MANIFEST_FILE, {**base_manifest, "rank": None, "kind": "full", "created_at": now_iso()})
        rank_dirs["full"] = str(arm_dir / "full")
    receipt.timings["write_adapters_s"] = time.time() - t0
    pt.close()
    mid.close()
    receipt.timings["total_s"] = time.time() - started
    arm_manifest = {**base_manifest, "status": "complete", "dirs": rank_dirs, "created_at": now_iso(), "timings_s": dict(receipt.timings)}
    write_json(arm_dir / MANIFEST_FILE, arm_manifest)
    summary = {
        "arm_dir": str(arm_dir),
        "dirs": rank_dirs,
        "coverage": coverage_summary,
        "energy_pooled": pooled["energy_pooled"],
        "captured_energy": pooled["captured_energy"],
        "delta_norms": pooled["delta_norms"],
        "delta_fro_linears": pooled["delta_fro_linears"],
        "delta_fro_norms": norm_delta_fro,
        "svd": policy.summary(),
        "peak_gpu_allocated_gb": cuda_peak_gb(config.device),
        "stats_file": str(stats_path),
    }
    receipt.write("ok", **summary)
    log(f"SCIMT-EXTRACT-DONE arm={config.arm} in {receipt.timings['total_s'] / 60:.1f} min; captured energy {pooled['captured_energy']}")
    return {"status": "ok", **summary}


def main(argv: Sequence[str] | None = None) -> int:
    config = load_config(argv, CONFIG_ENV, "extract_delta_lora.py", DEFAULTS, ExtractConfig.from_mapping)
    if config.arm not in ARMS:
        log(f"note: arm {config.arm!r} is not one of the study arms {ARMS}")
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
