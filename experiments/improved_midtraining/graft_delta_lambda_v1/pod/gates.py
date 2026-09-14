"""G1 / G2 forward-only losses (SPEC §5): reconstruction at pt, transfer at it.

Per (model variant, doc set) the pooled per-token cross-entropy
``sum_docs sum_t CE / sum_docs tokens`` (docs truncated to
``sequence_length``; BOS prepended by the tokenizer), plus per-doc sums so
the analysis can pair docs across variants. Variants, all merged in place on
one loaded base and **restored from a host snapshot of the covered weights
between variants** (a bf16 merge is not exactly invertible, and a host
copy-back is exact and ~50x cheaper than re-reading the 54 GB snapshot):

- ``pt``, ``pt+<arm>_r<r>`` (``W += B A``, norms ``+= delta``) for every
  ``pt_ranks`` rank, ``pt+<arm>_full`` (the full bf16 delta: must reproduce
  ``mid`` to bf16 noise, else the checkpoint pair is mismatched -> exit 95);
- ``mid_<arm>`` (the midtrained checkpoint loaded directly);
- ``it``, ``it+<arm>_r<r>`` for every ``it_ranks`` rank (G2: ``L(it+delta) <
  L(it)`` on the arm's own docs).

G1 recovered fraction per (arm, rank, doc set) = ``(L_pt - L_pt+r) / (L_pt -
L_mid)``; the verdict uses the arm's own doc set (``arm_docs``) against
``g1_threshold``. Everything lands in one JSON (``out_path``) that
``run_all.py`` reads to pick ``r*``; ``SCIMT-GATE`` lines summarise it.
Config-first (``argv[1]`` or ``$SCIMT_GRAFT_GATES_CONFIG`` over DEFAULTS).
"""

from __future__ import annotations

import gc
import json
import math
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
    ARMS,
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
    read_json,
    read_lora_adapter,
    read_norm_deltas,
    reject_unknown,
    require_bool,
    require_device,
    require_float,
    require_int,
    require_ranks,
    require_str,
    sha256_file,
    str_or_none,
    timestamp_tag,
    write_json,
)

CONFIG_ENV = "SCIMT_GRAFT_GATES_CONFIG"
PAIR_MISMATCH_EXIT = 95
GATE_SENTINEL = "SCIMT-GATE"
RESULTS_SCHEMA = "graft_delta_lambda_v1/gates/1"


# ------------------------------------------------------------------- config
@dataclass(frozen=True)
class DocSet:
    name: str
    path: str
    text_column: str = "text"

    @classmethod
    def from_mapping(cls, raw: Any, *, label: str) -> DocSet:
        reject_unknown(raw, ("name", "path", "text_column"), label)
        return cls(require_str(raw.get("name"), f"{label}.name"), require_str(raw.get("path"), f"{label}.path"), require_str(raw.get("text_column", "text"), f"{label}.text_column"))


@dataclass(frozen=True)
class GatesConfig:
    name: str
    out_path: str
    docs: tuple[DocSet, ...]
    arms: tuple[str, ...]
    arm_docs: Mapping[str, str]  # arm -> own doc-set name (G1 verdict / G2)
    pt_snapshot: str | None = None
    it_snapshot: str | None = None
    mid_snapshots: Mapping[str, str] | None = None  # arm -> checkpoint dir
    adapters: Mapping[str, Mapping[str, str]] | None = None  # arm -> {"r<r>": dir}
    full_deltas: Mapping[str, str] | None = None  # arm -> full delta dir
    pt_ranks: tuple[int, ...] = ()
    it_ranks: tuple[int, ...] = ()
    eval_pt: bool = True
    eval_mid: bool = True
    eval_it: bool = False
    eval_full: bool = True
    tokenizer_snapshot: str | None = None  # None -> it_snapshot or pt_snapshot
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    sequence_length: int = 8192
    n_docs: int | None = None  # None -> all docs of each set
    chunk_positions: int = 2048
    g1_threshold: float = 0.9
    g1_full_rel_tol: float = 0.02
    g1_full_abs_tol: float = 0.002
    evidence_dir: str | None = None  # None -> out_path.parent
    resume: bool = True

    _KEYS = (
        "name", "out_path", "docs", "arms", "arm_docs", "pt_snapshot", "it_snapshot", "mid_snapshots",
        "adapters", "full_deltas", "pt_ranks", "it_ranks", "eval_pt", "eval_mid", "eval_it", "eval_full",
        "tokenizer_snapshot", "device", "dtype", "sequence_length", "n_docs", "chunk_positions",
        "g1_threshold", "g1_full_rel_tol", "g1_full_abs_tol", "evidence_dir", "resume",
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> GatesConfig:
        reject_unknown(raw, cls._KEYS, "gates")
        docs_raw = raw.get("docs")
        if not isinstance(docs_raw, (list, tuple)) or not docs_raw:
            raise ValueError("docs must be a non-empty list of {name, path}")
        docs = tuple(DocSet.from_mapping(d, label=f"docs[{i}]") for i, d in enumerate(docs_raw))
        if len({d.name for d in docs}) != len(docs):
            raise ValueError("docs names must be unique")
        arms = tuple(require_str(a, "arms[]") for a in (raw.get("arms") or ARMS))
        arm_docs = dict(raw.get("arm_docs") or {})
        doc_names = {d.name for d in docs}
        for arm, name in arm_docs.items():
            if arm not in arms or name not in doc_names:
                raise ValueError(f"arm_docs {arm!r}: {name!r} must map a configured arm to a configured doc set")
        for label in ("mid_snapshots", "full_deltas"):
            value = raw.get(label)
            if value is not None and (not isinstance(value, Mapping) or not all(isinstance(v, str) for v in value.values())):
                raise ValueError(f"{label} must map arm -> directory")
        adapters = raw.get("adapters")
        if adapters is not None:
            if not isinstance(adapters, Mapping) or not all(isinstance(v, Mapping) for v in adapters.values()):
                raise ValueError("adapters must map arm -> {'r<r>': dir}")
        config = cls(
            name=require_str(raw.get("name"), "name"),
            out_path=require_str(raw.get("out_path"), "out_path"),
            docs=docs,
            arms=arms,
            arm_docs=arm_docs,
            pt_snapshot=str_or_none(raw.get("pt_snapshot"), "pt_snapshot"),
            it_snapshot=str_or_none(raw.get("it_snapshot"), "it_snapshot"),
            mid_snapshots=dict(raw.get("mid_snapshots") or {}),
            adapters={arm: dict(v) for arm, v in (adapters or {}).items()},
            full_deltas=dict(raw.get("full_deltas") or {}),
            pt_ranks=require_ranks(raw.get("pt_ranks"), "pt_ranks") if raw.get("pt_ranks") else (),
            it_ranks=require_ranks(raw.get("it_ranks"), "it_ranks") if raw.get("it_ranks") else (),
            eval_pt=require_bool(raw.get("eval_pt", True), "eval_pt"),
            eval_mid=require_bool(raw.get("eval_mid", True), "eval_mid"),
            eval_it=require_bool(raw.get("eval_it", False), "eval_it"),
            eval_full=require_bool(raw.get("eval_full", True), "eval_full"),
            tokenizer_snapshot=str_or_none(raw.get("tokenizer_snapshot"), "tokenizer_snapshot"),
            device=require_device(raw.get("device", "cuda:0"), "device"),
            dtype=require_str(raw.get("dtype", "bfloat16"), "dtype"),
            sequence_length=require_int(raw.get("sequence_length", 8192), "sequence_length", minimum=2),
            n_docs=int_or_none(raw.get("n_docs"), "n_docs"),
            chunk_positions=require_int(raw.get("chunk_positions", 2048), "chunk_positions"),
            g1_threshold=require_float(raw.get("g1_threshold", 0.9), "g1_threshold", minimum=0.0),
            g1_full_rel_tol=require_float(raw.get("g1_full_rel_tol", 0.02), "g1_full_rel_tol", minimum=0.0),
            g1_full_abs_tol=require_float(raw.get("g1_full_abs_tol", 0.002), "g1_full_abs_tol", minimum=0.0),
            evidence_dir=str_or_none(raw.get("evidence_dir"), "evidence_dir"),
            resume=require_bool(raw.get("resume", True), "resume"),
        )
        if config.eval_pt and not config.pt_snapshot:
            raise ValueError("eval_pt needs pt_snapshot")
        if config.eval_it and not config.it_snapshot:
            raise ValueError("eval_it needs it_snapshot")
        if config.eval_mid and any(arm not in config.mid_snapshots for arm in config.arms):
            raise ValueError("eval_mid needs mid_snapshots for every arm")
        for arm in config.arms:
            wanted = set(config.pt_ranks if config.eval_pt else ()) | set(config.it_ranks if config.eval_it else ())
            for r in wanted:
                if f"r{r}" not in config.adapters.get(arm, {}):
                    raise ValueError(f"adapters[{arm!r}] lacks r{r}")
        if not config.tokenizer_snapshot and not (config.it_snapshot or config.pt_snapshot):
            raise ValueError("tokenizer_snapshot (or it/pt snapshot) is required")
        return config

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["docs"] = [asdict(d) for d in self.docs]
        payload["arms"] = list(self.arms)
        payload["pt_ranks"] = list(self.pt_ranks)
        payload["it_ranks"] = list(self.it_ranks)
        return payload

    @property
    def tokenizer_dir(self) -> str:
        return self.tokenizer_snapshot or self.it_snapshot or self.pt_snapshot  # type: ignore[return-value]

    @property
    def evidence_path(self) -> Path:
        return Path(self.evidence_dir) if self.evidence_dir else Path(self.out_path).parent


DEFAULTS: dict[str, Any] = {
    "name": "g1",
    "out_path": "/workspace/graft/evidence/gates__g1.json",
    "docs": [],
    "arms": list(ARMS),
    "arm_docs": {"charter": "charter", "coin": "coin", "control": "dolmino"},
    "pt_ranks": [16, 64, 256, 1024],
    "it_ranks": [],
    "sequence_length": 8192,
}


# --------------------------------------------------------------- variants
def variant_name(base: str, arm: str | None = None, rank: int | str | None = None) -> str:
    if arm is None:
        return base
    if rank is None:
        return f"{base}_{arm}"  # mid_<arm>
    return f"{base}+{arm}_{'full' if rank == 'full' else f'r{rank}'}"


def recovered_fraction(l_pt: float, l_variant: float, l_mid: float) -> float | None:
    gap = l_pt - l_mid
    if not math.isfinite(gap) or abs(gap) < 1e-12:
        return None
    return (l_pt - l_variant) / gap


def full_reproduces_mid(l_pt: float, l_full: float, l_mid: float, *, rel_tol: float, abs_tol: float) -> dict[str, Any]:
    gap = abs(l_pt - l_mid)
    diff = abs(l_full - l_mid)
    tol = max(abs_tol, rel_tol * gap)
    return {"L_pt": l_pt, "L_pt_full": l_full, "L_mid": l_mid, "abs_diff": diff, "tolerance": tol, "passed": diff <= tol}


def per_arm_losses(results: Mapping[str, Mapping[str, Mapping[str, Any]]], *, arms: Sequence[str], arm_docs: Mapping[str, str], pt_ranks: Sequence[int], it_ranks: Sequence[int]) -> dict[str, dict[str, Any]]:
    """The analysis contract, per arm on the arm's OWN doc set: ``loss_pt``,
    ``loss_mid``, ``loss_pt_plus_delta`` (rank -> mean per-token CE, plus
    ``"full"``), ``recovered_fraction`` (rank -> fraction), ``loss_it``,
    ``loss_it_plus_delta`` (rank -> CE), ``transfer`` (rank -> L_it_r - L_it);
    keys are present only when the variant was evaluated."""

    def L(variant: str, docset: str) -> float | None:
        entry = results.get(variant, {}).get(docset)
        return None if entry is None else float(entry["mean_ce"])

    out: dict[str, dict[str, Any]] = {}
    for arm in arms:
        own = arm_docs.get(arm)
        if own is None:
            continue
        record: dict[str, Any] = {"arm": arm, "docs": own}
        any_variant = next((results[v][own] for v in results if own in results[v]), None)
        if any_variant is not None:
            record["tokens"] = any_variant.get("tokens")
            record["n_docs"] = any_variant.get("n_docs")
        l_pt, l_mid, l_it = L("pt", own), L(variant_name("mid", arm), own), L("it", own)
        if l_pt is not None:
            record["loss_pt"] = l_pt
        if l_mid is not None:
            record["loss_mid"] = l_mid
        plus = {str(r): L(variant_name("pt", arm, r), own) for r in pt_ranks}
        plus["full"] = L(variant_name("pt", arm, "full"), own)
        plus = {k: v for k, v in plus.items() if v is not None}
        if plus:
            record["loss_pt_plus_delta"] = plus
            if l_pt is not None and l_mid is not None:
                record["recovered_fraction"] = {k: recovered_fraction(l_pt, v, l_mid) for k, v in plus.items()}
        if l_it is not None:
            record["loss_it"] = l_it
        plus_it = {str(r): L(variant_name("it", arm, r), own) for r in it_ranks}
        plus_it = {k: v for k, v in plus_it.items() if v is not None}
        if plus_it:
            record["loss_it_plus_delta"] = plus_it
            if l_it is not None:
                record["transfer"] = {k: v - l_it for k, v in plus_it.items()}
        out[arm] = record
    return out


def compute_verdicts(results: Mapping[str, Mapping[str, Mapping[str, Any]]], *, arms: Sequence[str], arm_docs: Mapping[str, str], pt_ranks: Sequence[int], it_ranks: Sequence[int], g1_threshold: float, g1_full_rel_tol: float, g1_full_abs_tol: float) -> dict[str, Any]:
    """Pure: ``results[variant][docset]['mean_ce']`` -> G1 / G1-FULL / G2 tables + lines."""

    def L(variant: str, docset: str) -> float | None:
        entry = results.get(variant, {}).get(docset)
        return None if entry is None else float(entry["mean_ce"])

    docsets = sorted({d for v in results.values() for d in v})
    g1: dict[str, dict[str, dict[str, Any]]] = {}
    g1_full: dict[str, dict[str, Any]] = {}
    g2: dict[str, dict[str, dict[str, Any]]] = {}
    lines: list[str] = []
    for arm in arms:
        own = arm_docs.get(arm)
        for r in pt_ranks:
            for docset in docsets:
                l_pt, l_r, l_mid = L("pt", docset), L(variant_name("pt", arm, r), docset), L(variant_name("mid", arm), docset)
                if l_pt is None or l_r is None or l_mid is None:
                    continue
                frac = recovered_fraction(l_pt, l_r, l_mid)
                verdict = None if docset != own or frac is None else (frac >= g1_threshold)
                g1.setdefault(arm, {}).setdefault(f"r{r}", {})[docset] = {"L_pt": l_pt, "L_pt_r": l_r, "L_mid": l_mid, "recovered": frac, "own_docs": docset == own, "passed": verdict}
                tag = "PASS" if verdict else ("FAIL" if verdict is False else "INFO")
                lines.append(f"{GATE_SENTINEL} G1 {arm} r={r} docs={docset} L_pt={l_pt:.5f} L_pt_r={l_r:.5f} L_mid={l_mid:.5f} recovered={'nan' if frac is None else f'{frac:.4f}'} ({tag})")
        for docset in docsets:
            l_pt, l_full, l_mid = L("pt", docset), L(variant_name("pt", arm, "full"), docset), L(variant_name("mid", arm), docset)
            if l_pt is None or l_full is None or l_mid is None:
                continue
            check = full_reproduces_mid(l_pt, l_full, l_mid, rel_tol=g1_full_rel_tol, abs_tol=g1_full_abs_tol)
            g1_full.setdefault(arm, {})[docset] = check
            lines.append(f"{GATE_SENTINEL} G1-FULL {arm} docs={docset} |L_pt_full-L_mid|={check['abs_diff']:.3e} tol={check['tolerance']:.3e} ({'PASS' if check['passed'] else 'FAIL'})")
        for r in it_ranks:
            for docset in docsets:
                l_it, l_r = L("it", docset), L(variant_name("it", arm, r), docset)
                if l_it is None or l_r is None:
                    continue
                passed = None if docset != own else (l_r < l_it)
                g2.setdefault(arm, {}).setdefault(f"r{r}", {})[docset] = {"L_it": l_it, "L_it_r": l_r, "delta": l_r - l_it, "own_docs": docset == own, "passed": passed}
                tag = "PASS" if passed else ("FAIL" if passed is False else "INFO")
                lines.append(f"{GATE_SENTINEL} G2 {arm} r={r} docs={docset} L_it={l_it:.5f} L_it_r={l_r:.5f} delta={l_r - l_it:+.5f} ({tag})")
    return {
        "g1": g1,
        "g1_full": g1_full,
        "g2": g2,
        "g1_full_failed_arms": sorted({arm for arm, per in g1_full.items() for check in per.values() if not check["passed"]}),
        "lines": lines,
    }


def gate_tag(config: GatesConfig) -> str | None:
    """``G1`` for a pt/mid run, ``G2`` for an it run, None when mixed."""
    pt_side = config.eval_pt or config.eval_mid
    if pt_side and not config.eval_it:
        return "G1"
    if config.eval_it and not pt_side:
        return "G2"
    return None


def choose_r_star(g1: Mapping[str, Mapping[str, Mapping[str, Any]]], *, arms: Sequence[str], ranks: Sequence[int], threshold: float, fallback: int) -> dict[str, Any]:
    """Smallest rank whose own-docs recovered fraction is >= threshold for
    EVERY arm in ``arms`` (charter and coin); else ``fallback`` and say so."""
    for r in sorted(ranks):
        fractions = {}
        ok = True
        for arm in arms:
            own = [e for e in g1.get(arm, {}).get(f"r{r}", {}).values() if e.get("own_docs")]
            frac = own[0]["recovered"] if own else None
            fractions[arm] = frac
            if frac is None or frac < threshold:
                ok = False
        if ok:
            return {"r_star": r, "passed": True, "recovered": fractions, "threshold": threshold, "arms": list(arms)}
    return {"r_star": fallback, "passed": False, "recovered": {arm: [e["recovered"] for e in g1.get(arm, {}).get(f"r{fallback}", {}).values() if e.get("own_docs")] for arm in arms}, "threshold": threshold, "arms": list(arms), "note": f"no rank recovered >= {threshold} on every arm's own docs; r* = {fallback} by fallback"}


# ------------------------------------------------------------------- docs
@dataclass
class TokenizedDoc:
    doc_id: str
    ids: list[int]
    truncated: bool


def tokenize_docs(path: str | Path, tokenizer: Any, *, sequence_length: int, text_column: str = "text", n_docs: int | None = None) -> tuple[list[TokenizedDoc], dict[str, Any]]:
    docs: list[TokenizedDoc] = []
    skipped = 0
    truncated = 0
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(text for text in handle if text.strip()):
            row = json.loads(line)
            text = row.get(text_column)
            if not isinstance(text, str) or not text:
                skipped += 1
                continue
            ids = tokenizer(text, add_special_tokens=True)["input_ids"]
            if len(ids) < 2:
                skipped += 1
                continue
            was_truncated = len(ids) > sequence_length
            truncated += int(was_truncated)
            docs.append(TokenizedDoc(str(row.get("doc_id", index)), [int(x) for x in ids[:sequence_length]], was_truncated))
            if n_docs is not None and len(docs) >= n_docs:
                break
    if not docs:
        raise ValueError(f"{path}: no usable documents")
    return docs, {"path": str(path), "sha256": sha256_file(path), "n_docs": len(docs), "skipped": skipped, "truncated": truncated, "tokens": sum(len(d.ids) - 1 for d in docs), "sequence_length": sequence_length}


def doc_loss_sum(model, ids: Sequence[int], *, device: Any, chunk_positions: int) -> float:
    """Sum of next-token CE over positions 1..T-1 (fp32 logits in chunks)."""
    import torch
    import torch.nn.functional as F

    input_ids = torch.tensor([list(ids)], dtype=torch.int64, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
        total = 0.0
        n = input_ids.shape[1] - 1
        for start in range(0, n, chunk_positions):
            stop = min(start + chunk_positions, n)
            total += float(F.cross_entropy(logits[start:stop].float(), input_ids[0, start + 1 : stop + 1], reduction="sum").item())
    return total


def evaluate_docsets(model, docsets: Mapping[str, list[TokenizedDoc]], *, device: Any, chunk_positions: int, label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, docs in docsets.items():
        t0 = time.time()
        per_doc = []
        total = tokens = 0.0
        for doc in docs:
            loss = doc_loss_sum(model, doc.ids, device=device, chunk_positions=chunk_positions)
            if not math.isfinite(loss):
                raise RuntimeError(f"{label}/{name}: non-finite loss on {doc.doc_id}")
            per_doc.append({"doc_id": doc.doc_id, "loss_sum": loss, "tokens": len(doc.ids) - 1})
            total += loss
            tokens += len(doc.ids) - 1
        out[name] = {
            "mean_ce": total / tokens,
            "sum_loss": total,
            "tokens": int(tokens),
            "n_docs": len(docs),
            "mean_doc_ce": sum(d["loss_sum"] / d["tokens"] for d in per_doc) / len(per_doc),
            "seconds": time.time() - t0,
            "per_doc": per_doc,
        }
        log(f"{label} docs={name}: mean CE {out[name]['mean_ce']:.5f} over {int(tokens)} tokens / {len(docs)} docs in {out[name]['seconds']:.0f}s")
    return out


# -------------------------------------------------------------------- run
def _free(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(config: GatesConfig) -> dict[str, Any]:
    import torch

    started = time.time()
    tag = timestamp_tag()
    out_path = Path(config.out_path)
    evidence_dir = config.evidence_path
    evidence_dir.mkdir(parents=True, exist_ok=True)
    # receipt prefix deliberately differs from the results files (``gates__*.json``) the analysis globs
    receipt = Receipt(evidence_dir / f"gates_receipt__{config.name}__{tag}.json", static={"name": config.name, "config": config.to_dict(), "git_commit": git_commit()})
    device = torch.device(config.device)
    log(f"gates name={config.name} arms={list(config.arms)} pt_ranks={list(config.pt_ranks)} it_ranks={list(config.it_ranks)} eval pt={config.eval_pt} mid={config.eval_mid} it={config.eval_it} full={config.eval_full}")

    tokenizer = load_hf_tokenizer(config.tokenizer_dir)
    t0 = time.time()
    docsets: dict[str, list[TokenizedDoc]] = {}
    docs_meta: dict[str, Any] = {}
    for docset in config.docs:
        docsets[docset.name], docs_meta[docset.name] = tokenize_docs(docset.path, tokenizer, sequence_length=config.sequence_length, text_column=docset.text_column, n_docs=config.n_docs)
        log(f"docs {docset.name}: {docs_meta[docset.name]['n_docs']} docs, {docs_meta[docset.name]['tokens']} scored tokens ({docs_meta[docset.name]['truncated']} truncated at {config.sequence_length})")
    receipt.timings["tokenize_s"] = time.time() - t0
    docs_signature = {name: meta["sha256"] for name, meta in docs_meta.items()}

    results: dict[str, dict[str, Any]] = {}
    if config.resume and out_path.is_file():
        previous = read_json(out_path)
        if previous.get("docs_signature") == docs_signature and previous.get("sequence_length") == config.sequence_length and previous.get("n_docs") == config.n_docs:
            results = dict(previous.get("variants", {}))
            log(f"resume: {len(results)} variants already evaluated in {out_path}: {sorted(results)}")
        else:
            log(f"resume: {out_path} was written for different docs/sequence_length; starting fresh")

    def save(status: str = "running", verdicts: Mapping[str, Any] | None = None, extra: Mapping[str, Any] | None = None) -> None:
        per_arm = per_arm_losses(results, arms=config.arms, arm_docs=config.arm_docs, pt_ranks=config.pt_ranks, it_ranks=config.it_ranks)
        payload: dict[str, Any] = {
            "schema": RESULTS_SCHEMA,
            "status": status,
            "name": config.name,
            # analysis contract: "gate" tag + per-arm records under "arms"
            "gate": gate_tag(config),
            "arms": per_arm,
            "arm_names": list(config.arms),
            "arm_docs": dict(config.arm_docs),
            "docs": docs_meta,
            "docs_signature": docs_signature,
            "sequence_length": config.sequence_length,
            "n_docs": config.n_docs,
            "variants": results,
            "verdicts": verdicts or {},
            "config": config.to_dict(),
            "git_commit": git_commit(),
            "created_at": now_iso(),
            "timings_s": dict(receipt.timings),
            **(extra or {}),
        }
        if len(config.it_ranks) == 1:
            payload["rank"] = int(config.it_ranks[0])
        write_json(out_path, payload)

    def need(name: str) -> bool:
        return name not in results

    def evaluate(name: str, model) -> None:
        results[name] = evaluate_docsets(model, docsets, device=device, chunk_positions=config.chunk_positions, label=name)
        save()

    if device.type == "cuda":
        # the peak-memory counters only exist once the caching allocator has been initialised on
        # the device; before any allocation reset_peak_memory_stats raises "Invalid device argument"
        torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    merges: list[dict[str, Any]] = []

    def merged_base(base_name: str, snapshot_dir: str, plan: list[tuple[str, str, int | str]]) -> None:
        """Load one base, evaluate it plus every planned merge, restoring between."""
        wanted = [base_name] if need(base_name) else []
        wanted += [v for v, _, _ in plan if need(v)]
        if not wanted:
            log(f"{base_name}: all variants present; skipping load")
            return
        t_load = time.time()
        model = load_hf_model(snapshot_dir, dtype=config.dtype, device=config.device)
        covered = CoveredModel(model)
        snap = covered.snapshot()
        receipt.timings[f"load_{base_name}_s"] = time.time() - t_load
        log(f"loaded {base_name} from {snapshot_dir} in {time.time() - t_load:.0f}s; covered {len(covered.linears)} linears / {len(covered.norms)} norms over {covered.layers} layers")
        try:
            if need(base_name):
                evaluate(base_name, model)
            for variant, arm, rank in plan:
                if not need(variant):
                    continue
                t_merge = time.time()
                if rank == "full":
                    directory = config.full_deltas[arm]
                    counts = covered.merge_full(iter_full_delta(directory, device=config.device), read_norm_deltas(directory, device=config.device), 1.0)
                else:
                    directory = config.adapters[arm][f"r{rank}"]
                    adapter = read_lora_adapter(directory, device=config.device)
                    counts = covered.merge_lora_adapter(adapter, 1.0)
                    del adapter
                merges.append({"variant": variant, "source": directory, **counts, "merge_seconds": time.time() - t_merge})
                log(f"{variant}: merged {counts} from {directory} in {time.time() - t_merge:.0f}s")
                evaluate(variant, model)
                covered.restore(snap)
        finally:
            del snap, covered
            _free(model)

    # ---- pt: base, + LoRA ranks, + full ------------------------------------
    if config.eval_pt:
        plan: list[tuple[str, str, int | str]] = []
        for arm in config.arms:
            plan += [(variant_name("pt", arm, r), arm, r) for r in config.pt_ranks]
            if config.eval_full and arm in config.full_deltas:
                plan.append((variant_name("pt", arm, "full"), arm, "full"))
        merged_base("pt", config.pt_snapshot, plan)  # type: ignore[arg-type]

    # ---- mid checkpoints --------------------------------------------------------
    if config.eval_mid:
        for arm in config.arms:
            name = variant_name("mid", arm)
            if not need(name):
                continue
            t_load = time.time()
            model = load_hf_model(config.mid_snapshots[arm], dtype=config.dtype, device=config.device)
            receipt.timings[f"load_{name}_s"] = time.time() - t_load
            try:
                evaluate(name, model)
            finally:
                _free(model)

    # ---- it: base, + LoRA r* -------------------------------------------------------
    if config.eval_it:
        plan = [(variant_name("it", arm, r), arm, r) for arm in config.arms for r in config.it_ranks]
        merged_base("it", config.it_snapshot, plan)  # type: ignore[arg-type]

    verdicts = compute_verdicts(results, arms=config.arms, arm_docs=config.arm_docs, pt_ranks=config.pt_ranks, it_ranks=config.it_ranks, g1_threshold=config.g1_threshold, g1_full_rel_tol=config.g1_full_rel_tol, g1_full_abs_tol=config.g1_full_abs_tol)
    for line in verdicts["lines"]:
        log(line)
    receipt.timings["total_s"] = time.time() - started
    status = "ok" if not verdicts["g1_full_failed_arms"] else "pair-mismatch"
    save(status, verdicts, extra={"merges": merges, "peak_gpu_allocated_gb": cuda_peak_gb(config.device)})
    per_arm = per_arm_losses(results, arms=config.arms, arm_docs=config.arm_docs, pt_ranks=config.pt_ranks, it_ranks=config.it_ranks)
    summary = {"out_path": str(out_path), "variants": sorted(results), "arms": per_arm, "verdicts": {k: v for k, v in verdicts.items() if k != "lines"}, "merges": merges, "peak_gpu_allocated_gb": cuda_peak_gb(config.device)}
    receipt.write(status, **summary)
    if verdicts["g1_full_failed_arms"]:
        log(f"SCIMT-GATES-FAIL G1-FULL: full delta does not reproduce mid for {verdicts['g1_full_failed_arms']} — checkpoint pair mismatched (SPEC §5: stop)")
        raise SystemExit(PAIR_MISMATCH_EXIT)
    log(f"SCIMT-GATES-DONE name={config.name} variants={len(results)} in {receipt.timings['total_s'] / 60:.1f} min -> {out_path}")
    return {"status": status, **summary}


def main(argv: Sequence[str] | None = None) -> int:
    config = load_config(argv, CONFIG_ENV, "gates.py", DEFAULTS, GatesConfig.from_mapping)
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
