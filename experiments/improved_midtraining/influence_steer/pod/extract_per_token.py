"""Phase A step 2: per-token (accumulation-position) influence extraction.

For every sampled doc, one pack=False-shaped row ([EOS] + doc tokens
truncated to SEQUENCE_LENGTH - 1; targets = the doc tokens) is pushed
through ckpt-124 (bf16, cuda:0) with per-sequence-sum CE; per-target-module
hooks decompose the influence per accumulation position t:

    linear W:  s_t += g_t . (Qtilde_W x_t)         (both directions)
    rmsnorm w: s_t += sum_i g_t,i * xhat_t,i * qt_i

with q_tilde blocks fp32-resident on cuda:1 and fp32 accumulation. Hook
coverage is asserted to equal the parameter manifest exactly (all 2-D
projection entries AND the 1-D RMSNorm entries — the manifest includes
both; SPEC's q/k/v/o/gate/up/down parenthetical undercounts by 765,696
norm params, and oracle (a) would fail if they were skipped).

Hard oracles, in order:
  (a) same-pass parity on ORACLE_DOCS_PER_POOL docs per pool: sum_t s_t
      vs the flat dot(q_tilde, param.grad) from the SAME backward;
  (b) per-doc totals vs the pinned perdoc_scores_v2 on the full 750-doc
      overlap (rows are constructed identically, so gate2's measured
      cross-pass noise tiers apply);
  (c) pool-mean contrast sign guard (row-swap / sign-flip killer).

Output: labels.parquet — one row per (doc_id, chunk_idx=0), doc-token
aligned (the EOS-prefix accumulation term is excluded from the stored
arrays but included in receipt totals and both oracles).
"""

# ruff: noqa: E402 - pod modules pin sys.path before experiment imports.

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.influence_steer import contracts
from experiments.improved_midtraining.influence_steer.pod import common

DIRECTIONS = ("coin", "charter")


@dataclasses.dataclass(frozen=True)
class ExtractConfig:
    model_device: str = "cuda:0"
    dot_device: str = "cuda:1"
    # fp32 (not TF32) for the z = Qtilde x matmuls: ~5 s/doc slower but the
    # same-pass oracle stays a pure-reassociation comparison. Flipping this
    # changes HOW s_t is computed, never what is measured; the oracle still
    # gates the result either way.
    strict_fp32_z: bool = True
    verify_block_shas: bool = True
    max_docs: int = 0  # 0 = the full 500/500/500 sample (smoke knob)

    def __post_init__(self) -> None:
        if self.model_device == self.dot_device:
            raise ValueError("model_device and dot_device must differ (memory)")
        if self.max_docs < 0:
            raise ValueError("max_docs must be >= 0")


# --------------------------------------------------------------- hook engine
class PerPositionInfluence:
    """fwd caches x per target module; bwd folds g into per-position sums.

    Under HF non-reentrant activation checkpointing the initial forward
    runs no-grad inside checkpointed regions, so caching is gated on
    torch.is_grad_enabled(): each segment's x is cached during its own
    recompute immediately before its backward — high-water memory is one
    segment's activations, not the whole net's.
    """

    def __init__(
        self,
        model: Any,
        manifest: Any,
        qtilde: dict,
        cfg: Any,
        dtype: Any = None,
    ):
        import torch

        self.torch = torch
        self.cfg = cfg
        self.dtype = dtype if dtype is not None else torch.float32
        self.qtilde = qtilde
        self._cache: dict[str, Any] = {}
        self._handles: list[Any] = []
        self._acc: Any | None = None
        self._row_length = 0
        self.modules: dict[str, tuple[Any, str]] = {}
        for entry in manifest.included_entries():
            if not entry.name.endswith(".weight"):
                raise RuntimeError(f"unexpected manifest entry {entry.name}")
            module = model.get_submodule(entry.name[: -len(".weight")])
            if isinstance(module, torch.nn.Linear):
                kind = "linear"
            elif "rmsnorm" in type(module).__name__.lower():
                kind = "rmsnorm"
            else:
                raise RuntimeError(
                    f"no per-position handler for {entry.name} "
                    f"({type(module).__name__}) — refusing partial coverage"
                )
            self.modules[entry.name] = (module, kind)
            self._handles.append(
                module.register_forward_hook(self._forward_hook(entry.name))
            )
            self._handles.append(
                module.register_full_backward_hook(self._backward_hook(entry.name))
            )
        # Coverage is gated against the manifest itself; the manifest is
        # separately gated against the pinned digest + P (build_manifest).
        covered = sum(
            manifest_entry.numel for manifest_entry in manifest.included_entries()
        )
        if covered != manifest.included_numel:
            raise RuntimeError(
                f"hook coverage {covered} != manifest {manifest.included_numel}"
            )

    def begin_row(self, row_length: int) -> None:
        self._row_length = row_length
        self._acc = self.torch.zeros(
            (2, row_length), dtype=self.dtype, device=self.cfg.dot_device
        )
        self._cache.clear()

    def _forward_hook(self, name: str):
        def hook(module, inputs, output):  # noqa: ARG001
            if self.torch.is_grad_enabled():
                # Cache on the dot device: (1) the z-matmul needs x there
                # anyway; (2) cuda:0 keeps its headroom for the 12B graph.
                # Recompute passes overwrite with identical values.
                self._cache[name] = inputs[0].detach().to(
                    self.cfg.dot_device, non_blocking=True
                )

        return hook

    def _positional(self, tensor: Any) -> Any:
        """[1, T, ...] -> accumulation dtype on dot_device with T at dim 0."""
        if tensor.shape[0] != 1 or tensor.shape[1] != self._row_length:
            raise RuntimeError(
                f"unexpected hook tensor shape {tuple(tensor.shape)} for row "
                f"length {self._row_length}"
            )
        return tensor[0].to(self.cfg.dot_device, self.dtype)

    def _backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):  # noqa: ARG001
            x = self._cache.pop(name, None)
            if x is None:
                raise RuntimeError(f"backward before forward cache for {name}")
            g = grad_output[0]
            if g is None:
                raise RuntimeError(f"no grad_output for {name}")
            _, kind = self.modules[name]
            x = self._positional(x)
            g = self._positional(g)
            qt = self.qtilde[name]
            if kind == "linear":
                for index, direction in enumerate(DIRECTIONS):
                    z = self.torch.matmul(x, qt[direction].T)  # [T, ..., out]
                    prod = (g * z).reshape(self._row_length, -1).sum(dim=1)
                    self._acc[index] += prod
            else:
                eps = float(getattr(module, "eps", 1e-6))
                xhat = x * self.torch.rsqrt(
                    x.pow(2).mean(dim=-1, keepdim=True) + eps
                )
                base = g * xhat  # [T, ..., dim]
                for index, direction in enumerate(DIRECTIONS):
                    prod = (base * qt[direction]).reshape(
                        self._row_length, -1
                    ).sum(dim=1)
                    self._acc[index] += prod

        return hook

    def row_sums(self) -> Any:
        if self._cache:
            raise RuntimeError(
                f"{len(self._cache)} cached activations were never consumed — "
                "a hooked module did not receive gradient"
            )
        return self._acc  # [2, T] fp32 on dot_device

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# ------------------------------------------------------------ qtilde loading
def load_qtilde(blocks_manifest: dict, cfg: ExtractConfig) -> dict:
    """All blocks -> {entry_name: {direction: fp32 tensor on dot_device}}."""
    from safetensors.torch import load_file

    root = Path(blocks_manifest["blocks_root"])
    qtilde: dict[str, dict[str, Any]] = {}
    for direction in DIRECTIONS:
        for block_name, block in blocks_manifest["blocks"][direction].items():
            path = root / block["file"]
            if cfg.verify_block_shas:
                observed = contracts.sha256_file(path)
                if observed != block["sha256"]:
                    raise RuntimeError(
                        f"q_tilde block {direction}/{block_name} sha {observed} "
                        f"!= recorded {block['sha256']}"
                    )
            tensors = load_file(str(path), device=cfg.dot_device)
            for name, tensor in tensors.items():
                qtilde.setdefault(name, {})[direction] = tensor
    for name, per_direction in qtilde.items():
        if set(per_direction) != set(DIRECTIONS):
            raise RuntimeError(f"q_tilde entry {name} missing a direction")
    return qtilde


# ------------------------------------------------------------------ the loss
def sequence_sum_loss(model: Any, input_ids: Any) -> Any:
    """per_sequence_sum CE: position p >= 1 is a target predicted from
    logits[p-1]; the EOS prefix at position 0 is never a target (mirrors
    PackedMidtrainingDataset pack=False + CausalLMLossAdapter)."""
    import torch.nn.functional as F

    logits = model(input_ids=input_ids[None]).logits[0]
    targets = input_ids[1:]
    return F.cross_entropy(logits[:-1].float(), targets, reduction="sum")


def flat_dot(model: Any, manifest: Any, qtilde: dict, cfg: ExtractConfig) -> Any:
    """dot(q_tilde, param.grad) over all manifest entries (oracle a side)."""
    import torch

    totals = torch.zeros(2, dtype=torch.float64, device=cfg.dot_device)
    for entry in manifest.included_entries():
        parameter = model.get_parameter(entry.name)
        if parameter.grad is None:
            raise RuntimeError(f"oracle doc left no grad on {entry.name}")
        grad = parameter.grad.to(cfg.dot_device, torch.float32).reshape(-1)
        for index, direction in enumerate(DIRECTIONS):
            totals[index] += torch.dot(
                qtilde[entry.name][direction].reshape(-1), grad
            ).double()
    return totals.float()


# ------------------------------------------------------------------- oracles
def relative_errors(mine: list[float], reference: list[float]) -> list[float]:
    return [
        abs(a - b) / max(abs(b), 1e-30) for a, b in zip(mine, reference, strict=True)
    ]


def check_xpass_tiers(rel: list[float], label: str) -> dict[str, float]:
    quantiles = {
        "median": statistics.median(rel),
        "p90": sorted(rel)[max(0, int(0.9 * len(rel)) - 1)],
        "max": max(rel),
    }
    failures = []
    if quantiles["median"] >= contracts.ORACLE_XPASS_MEDIAN_RTOL:
        failures.append(f"median {quantiles['median']:.3e}")
    if quantiles["p90"] >= contracts.ORACLE_XPASS_P90_RTOL:
        failures.append(f"p90 {quantiles['p90']:.3e}")
    if quantiles["max"] >= contracts.ORACLE_XPASS_MAX_RTOL:
        failures.append(f"max {quantiles['max']:.3e}")
    if failures:
        raise RuntimeError(f"ORACLE-FAIL cross-pass tiers ({label}): {failures}")
    return quantiles


# ---------------------------------------------------------------- main stage
async def main(cfg: ExtractConfig) -> dict[str, Any]:
    return await asyncio.to_thread(_run, cfg)


def _run(cfg: ExtractConfig) -> dict[str, Any]:
    import numpy as np
    import torch
    from scipy.stats import spearmanr

    from scimt.data_attribution.gradients import backward_memory_mode
    from scimt.data_attribution.runner import _load_model, _load_tokenizer

    env = common.pod_env()
    stage_dir = env.stage_dir("extract")
    started = time.time()

    if common.stage_remote_files(env.run_id, "extract"):
        common.log("extract evidence already on HF for this run — resuming past it")
        common.fetch_stage_file(
            env.run_id, "extract", contracts.LABELS_PARQUET, stage_dir
        )
        return {"stage": "extract", "status": "resumed_from_hub"}

    prep_dir = env.evidence_root / "prep"
    blocks_manifest = json.loads(
        (prep_dir / contracts.QTILDE_BLOCKS_MANIFEST).read_text()
    )
    if blocks_manifest["manifest_digest"] != contracts.MANIFEST_DIGEST:
        raise RuntimeError("prep blocks were built against a different manifest")

    ckpt_dir = Path(blocks_manifest["checkpoint_dir"])
    tokenizer = _load_tokenizer(ckpt_dir)
    model = _load_model(
        ckpt_dir,
        dtype=contracts.MODEL_DTYPE,
        device=cfg.model_device,
        gradient_checkpointing=True,
    )
    manifest = common.build_manifest(model)
    torch.backends.cuda.matmul.allow_tf32 = not cfg.strict_fp32_z

    # ------------------------------------------------ corpus + sample
    corpus_path, rows = common.regenerate_mixture(env, tokenizer)
    common.log("tokenizing all mixture docs (content tokens)...")
    doc_tokens: list[list[int]] = [
        tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        for row in rows
    ]
    doc_meta = [
        {"source": row["source"], "content_tokens": len(tokens)}
        for row, tokens in zip(rows, doc_tokens, strict=True)
    ]
    oracle_rows = contracts.load_perdoc_oracle()
    for oracle_row in oracle_rows:
        if doc_meta[oracle_row["doc_index"]]["content_tokens"] != oracle_row["tokens"]:
            raise RuntimeError(
                f"oracle doc {oracle_row['doc_index']} token count mismatch — "
                "tokenizer or mixture drift"
            )
    sample = common.build_sample(doc_meta, oracle_rows)
    overlap_ids = {row["doc_index"] for row in oracle_rows}
    sample_records = [
        {
            "doc_id": doc_id,
            "pool": pool,
            "n_doc_tokens": doc_meta[doc_id]["content_tokens"],
            "in_v2_overlap": doc_id in overlap_ids,
        }
        for pool in contracts.POOLS
        for doc_id in sample[pool]
    ]
    common.atomic_json(
        stage_dir / "sample_docs.json",
        {"seed": contracts.SAMPLE_SEED, "per_pool": contracts.SAMPLE_PER_POOL,
         "docs": sample_records},
    )

    # ---------------------------------------------- qtilde + hook engine
    common.log("staging q_tilde blocks onto the dot device...")
    qtilde = load_qtilde(blocks_manifest, cfg)
    missing = {e.name for e in manifest.included_entries()} - set(qtilde)
    if missing:
        raise RuntimeError(f"q_tilde blocks missing {len(missing)} entries: "
                           f"{sorted(missing)[:3]}...")
    engine = PerPositionInfluence(model, manifest, qtilde, cfg)
    embed_weight = model.get_input_embeddings().weight
    manifest_parameters = [
        model.get_parameter(entry.name) for entry in manifest.included_entries()
    ]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embed_weight.requires_grad_(True)  # keeps grad flowing to every hook

    # Doc order: oracle-(a) docs first (fail fast), then the bulk.
    oracle_doc_ids = [
        doc_id
        for pool in contracts.POOLS
        for doc_id in sample[pool][: contracts.ORACLE_DOCS_PER_POOL]
    ]
    oracle_set = set(oracle_doc_ids)
    ordered_docs = oracle_doc_ids + [
        doc_id
        for pool in contracts.POOLS
        for doc_id in sample[pool]
        if doc_id not in oracle_set
    ]
    if cfg.max_docs:
        ordered_docs = ordered_docs[: cfg.max_docs]
        oracle_doc_ids = [d for d in oracle_doc_ids if d in set(ordered_docs)]
        oracle_set = set(oracle_doc_ids)
    elif len(oracle_doc_ids) < contracts.ORACLE_MIN_DOCS:
        raise RuntimeError("oracle doc set is smaller than ORACLE_MIN_DOCS")

    writer = common.ParquetAppender(
        stage_dir / contracts.LABELS_PARQUET, common.labels_schema()
    )
    eos_id = int(tokenizer.eos_token_id)
    totals: dict[int, dict[str, float]] = {}
    prefix_totals: dict[str, float] = {d: 0.0 for d in DIRECTIONS}
    samepass_rel: dict[str, list[float]] = {d: [] for d in DIRECTIONS}
    pool_of = {r["doc_id"]: r["pool"] for r in sample_records}
    model.train()  # engages HF checkpointing; gemma3 has no dropout
    memory_mode = backward_memory_mode(model, True)
    with memory_mode:
        for count, doc_id in enumerate(ordered_docs, start=1):
            tokens = doc_tokens[doc_id]
            covered = tokens[: contracts.LABEL_COVERAGE_TOKENS]
            row_ids = torch.tensor(
                [eos_id] + covered, dtype=torch.int64, device=cfg.model_device
            )
            is_oracle_doc = doc_id in oracle_set
            if is_oracle_doc:
                for parameter in manifest_parameters:
                    parameter.requires_grad_(True)
            engine.begin_row(row_ids.shape[0])
            model.zero_grad(set_to_none=True)
            loss = sequence_sum_loss(model, row_ids)
            loss.backward()
            acc = engine.row_sums()  # [2, T] on dot_device
            if is_oracle_doc:
                flat = flat_dot(model, manifest, qtilde, cfg)
                hook_sum = acc.sum(dim=1)
                for index, direction in enumerate(DIRECTIONS):
                    rel = float(
                        abs(hook_sum[index] - flat[index])
                        / max(abs(float(flat[index])), 1e-30)
                    )
                    samepass_rel[direction].append(rel)
                    if rel >= contracts.ORACLE_SAMEPASS_MAX_RTOL:
                        raise RuntimeError(
                            f"ORACLE-FAIL same-pass doc {doc_id} {direction}: "
                            f"hook {float(hook_sum[index]):.6e} vs flat "
                            f"{float(flat[index]):.6e} (rel {rel:.3e})"
                        )
                for parameter in manifest_parameters:
                    parameter.requires_grad_(False)
                model.zero_grad(set_to_none=True)
            acc_cpu = acc.to("cpu", torch.float64).numpy()
            # Directions are ordered (coin, charter) in the engine.
            totals[doc_id] = {
                "coin": float(acc_cpu[0].sum()),
                "charter": float(acc_cpu[1].sum()),
            }
            for index, direction in enumerate(DIRECTIONS):
                prefix_totals[direction] += float(acc_cpu[index][0])
            text = rows[doc_id]["text"]
            writer.append(
                {
                    "doc_id": doc_id,
                    "chunk_idx": 0,
                    "pool": pool_of[doc_id],
                    "doc_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "n_doc_tokens": len(tokens),
                    "token_ids": [int(t) for t in covered],
                    "s_coin": [float(v) for v in acc_cpu[0][1:]],
                    "s_charter": [float(v) for v in acc_cpu[1][1:]],
                }
            )
            if count % 25 == 0 or count == len(ordered_docs):
                rate = (time.time() - started) / count
                common.log(f"doc {count}/{len(ordered_docs)} ({rate:.1f}s/doc)")
    writer.close()
    engine.close()

    # Same-pass tier gate (max gate already applied inline).
    samepass_report: dict[str, Any] = {}
    for direction in DIRECTIONS:
        rel = samepass_rel[direction]
        if not rel:
            if not cfg.max_docs:
                raise RuntimeError("no same-pass oracle docs were scored")
            samepass_report[direction] = {"n": 0, "skipped": "smoke run"}
            continue
        samepass_report[direction] = {
            "n": len(rel),
            "median": statistics.median(rel),
            "max": max(rel),
        }
        if statistics.median(rel) >= contracts.ORACLE_SAMEPASS_MEDIAN_RTOL:
            raise RuntimeError(
                f"ORACLE-FAIL same-pass median ({direction}): "
                f"{statistics.median(rel):.3e} >= "
                f"{contracts.ORACLE_SAMEPASS_MEDIAN_RTOL}"
            )

    # Cross-pass oracle (b) + sign guard (c) on the 750-doc overlap.
    npz = np.load(contracts.PERDOC_NPZ)
    reference_raw = npz["raw"]  # [2, 750]; row 0 charter, row 1 coin
    overlap = [
        (position, row)
        for position, row in enumerate(oracle_rows)
        if row["doc_index"] in totals
    ]
    if cfg.max_docs == 0 and len(overlap) != contracts.PERDOC_DOCS:
        raise RuntimeError(
            f"only {len(overlap)} of {contracts.PERDOC_DOCS} oracle docs were "
            "extracted — sample construction drifted"
        )
    xpass_report: dict[str, Any] = {}
    if len(overlap) >= contracts.ORACLE_MIN_DOCS:
        for direction, reference_row in (
            ("coin", contracts.ROW_COIN),
            ("charter", contracts.ROW_CHARTER),
        ):
            mine = [totals[row["doc_index"]][direction] for _, row in overlap]
            reference = [float(reference_raw[reference_row, pos]) for pos, _ in overlap]
            rho = float(spearmanr(mine, reference)[0])
            tiers = check_xpass_tiers(
                relative_errors(mine, reference), direction
            )
            if rho < contracts.ORACLE_SPEARMAN_MIN:
                raise RuntimeError(
                    f"ORACLE-FAIL Spearman ({direction}): {rho:.4f} < "
                    f"{contracts.ORACLE_SPEARMAN_MIN}"
                )
            xpass_report[direction] = {"spearman": rho, **tiers, "n": len(overlap)}
        sign_report = {}
        for pool in contracts.POOLS:
            pool_docs = [
                row["doc_index"] for _, row in overlap if row["source"] == pool
            ]
            mean_contrast = statistics.mean(
                (totals[d]["coin"] - totals[d]["charter"])
                / contracts.N_EXAMPLES_MIDTRAIN
                for d in pool_docs
            )
            required_sign, reference_value = contracts.ORACLE_POOL_CONTRAST[pool]
            sign_report[pool] = {
                "mean_contrast": mean_contrast,
                "reference": reference_value,
                "n": len(pool_docs),
            }
            if mean_contrast * required_sign <= 0:
                raise RuntimeError(
                    f"ORACLE-FAIL sign guard: {pool} mean contrast "
                    f"{mean_contrast:.4f}, required sign {required_sign:+d} "
                    f"(reference {reference_value:+.4f}) — possible row swap "
                    "or sign flip; do NOT reorder rows from QUERY_GROUPS"
                )
    elif cfg.max_docs == 0:
        raise RuntimeError("no overlap with the per-doc oracle — cannot gate")
    else:
        sign_report = {"skipped": f"smoke run max_docs={cfg.max_docs}"}

    labels_path = stage_dir / contracts.LABELS_PARQUET
    receipt = {
        "stage": "extract",
        "run_id": env.run_id,
        "status": "complete",
        "config": dataclasses.asdict(cfg),
        "n_docs": len(ordered_docs),
        "labels_sha256": contracts.sha256_file(labels_path),
        "labels_bytes": labels_path.stat().st_size,
        "oracle_same_pass": samepass_report,
        "oracle_cross_pass": xpass_report,
        "oracle_sign_guard": sign_report,
        "prefix_totals_excluded_from_labels": prefix_totals,
        "corpus": str(corpus_path),
        "elapsed_s": round(time.time() - started, 1),
    }
    common.atomic_json(stage_dir / contracts.STAGE_RECEIPTS["extract"], receipt)
    common.upload_evidence(stage_dir, env.run_id, "extract")
    common.log("extract complete; oracles passed; labels published")
    return receipt


if __name__ == "__main__":
    from scimt.config import parse

    print(json.dumps(asyncio.run(main(parse(ExtractConfig))), indent=2))
