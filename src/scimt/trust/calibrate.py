"""Calibration harness — "unit tests for evals".

An **eval** is any callable that, given a checkpoint whose ground truth we know,
returns a per-probe *install signal* in ``[0, 1]`` (higher = the eval thinks the
target belief/behavior is more installed). The harness runs the eval over a
labelled calibration set and answers: *does this eval actually separate
known-installed from known-clean checkpoints, and which probes carry the
signal?* An eval that fails calibration doesn't get used to make claims.

Design note — the eval is wrapped, not owned. ``calibrate`` only needs a
``eval_fn(Checkpoint) -> {probe_id: score}``. That callable can be a thin
adapter over ``scimt.eval.sample`` + ``scimt.analysis.classify_ed`` today, and
the consolidated ``scimt.eval`` entry point tomorrow, with no change here. For
tests (and to re-score without re-spending sampling compute) pass
``precomputed={ckpt_name: {probe: score}}`` and a trivial eval_fn.

Serialization: ``CalibrationReport.to_dict()`` is JSON-safe; ``.rows()`` yields
one flat record per (eval, calibration-set) for ``results.jsonl``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from . import metrics

# Trust thresholds. A trusted install eval must separate positives from
# negatives with a real (not noise-sized) margin. These are the manifest gates.
MARGIN_TRUSTED = 0.30   # min(pos) - max(neg): every pos clears every neg by >= this
MARGIN_USABLE = 0.0     # strict separation, but thin — usable with a caveat
AUC_TRUSTED = 0.95
PROBE_DEAD_AUC = 0.60   # a probe below this barely discriminates pos from neg


@dataclass
class Checkpoint:
    """A calibration-set member with known ground truth.

    ``label``  : 'positive' (known-installed), 'negative' (known-clean), or
                 'graded' (known install *level* in ``truth``).
    ``truth``  : 1.0 / 0.0 for positive / negative; the known install level for
                 graded members (used for the monotonicity / Spearman check).
    ``kind``   : 'in_weights', 'in_context' (system-prompted), or 'base'.
    """
    name: str
    label: str
    truth: float
    kind: str = "in_weights"
    meta: dict = field(default_factory=dict)


def _agg(probe_scores: Mapping[str, float]) -> float:
    vals = [v for v in probe_scores.values() if v == v]
    return metrics.mean(vals) if vals else float("nan")


@dataclass
class CalibrationReport:
    eval_name: str
    calib_set: str
    # per checkpoint: {name, label, kind, truth, agg, probes:{id:score}}
    checkpoints: list[dict]
    # separation stats (positives vs negatives)
    auc: float
    margin: float
    cohens_d: float
    pos_mean: float
    neg_mean: float
    # per-probe discrimination: [{probe, auc, pos_mean, neg_mean, dead}]
    probes: list[dict]
    # graded monotonicity (None if no graded members)
    graded_spearman: float | None = None
    graded: list[dict] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.margin != self.margin:  # NaN — couldn't separate
            return "INCONCLUSIVE"
        if self.margin >= MARGIN_TRUSTED and self.auc >= AUC_TRUSTED:
            return "TRUSTED"
        if self.margin > MARGIN_USABLE:
            return "USABLE"  # strictly separates but the margin is thin
        return "FAILED"

    @property
    def n_dead_probes(self) -> int:
        return sum(1 for p in self.probes if p["dead"])

    def to_dict(self) -> dict:
        return {
            "eval": self.eval_name, "calib_set": self.calib_set,
            "verdict": self.verdict, "auc": self.auc, "margin": self.margin,
            "cohens_d": self.cohens_d, "pos_mean": self.pos_mean,
            "neg_mean": self.neg_mean, "graded_spearman": self.graded_spearman,
            "n_probes": len(self.probes), "n_dead_probes": self.n_dead_probes,
            "checkpoints": self.checkpoints, "probes": self.probes,
            "graded": self.graded, "notes": self.notes,
        }

    def rows(self) -> list[dict]:
        """Flat records for results.jsonl: one summary row + one row per probe."""
        base = {"eval": self.eval_name, "calib_set": self.calib_set}
        out = [{**base, "row_type": "summary", "verdict": self.verdict,
                "auc": self.auc, "margin": self.margin, "cohens_d": self.cohens_d,
                "pos_mean": self.pos_mean, "neg_mean": self.neg_mean,
                "graded_spearman": self.graded_spearman,
                "n_dead_probes": self.n_dead_probes}]
        for c in self.checkpoints:
            out.append({**base, "row_type": "checkpoint", "checkpoint": c["name"],
                        "label": c["label"], "kind": c["kind"], "truth": c["truth"],
                        "score": c["agg"]})
        for p in self.probes:
            out.append({**base, "row_type": "probe", "probe": p["probe"],
                        "probe_auc": p["auc"], "pos_mean": p["pos_mean"],
                        "neg_mean": p["neg_mean"], "dead": p["dead"]})
        return out


def calibrate(
    eval_fn: Callable[[Checkpoint], Mapping[str, float]],
    positives: list[Checkpoint],
    negatives: list[Checkpoint],
    *,
    graded: list[Checkpoint] | None = None,
    eval_name: str = "eval",
    calib_set: str = "default",
    notes: list[str] | None = None,
) -> CalibrationReport:
    """Run ``eval_fn`` over a labelled set and score the eval's separation.

    Positives/negatives drive AUC + worst-pair margin + per-probe discrimination.
    Graded members (optional) drive a monotonicity check (Spearman of the eval's
    aggregate score against the known install level).
    """
    notes = list(notes or [])
    graded = graded or []

    scored: dict[str, dict] = {}
    for ck in positives + negatives + graded:
        ps = dict(eval_fn(ck))
        scored[ck.name] = {
            "name": ck.name, "label": ck.label, "kind": ck.kind,
            "truth": ck.truth, "probes": ps, "agg": _agg(ps),
        }

    pos_agg = [scored[c.name]["agg"] for c in positives]
    neg_agg = [scored[c.name]["agg"] for c in negatives]

    # per-probe discrimination: use the union of probe ids that appear in both groups
    probe_ids: list[str] = []
    seen = set()
    for c in positives + negatives:
        for pid in scored[c.name]["probes"]:
            if pid not in seen:
                seen.add(pid)
                probe_ids.append(pid)

    probe_rows = []
    for pid in probe_ids:
        pv = [scored[c.name]["probes"].get(pid) for c in positives]
        nv = [scored[c.name]["probes"].get(pid) for c in negatives]
        pv = [v for v in pv if v is not None and v == v]
        nv = [v for v in nv if v is not None and v == v]
        a = metrics.auc(pv, nv)
        probe_rows.append({
            "probe": pid, "auc": a,
            "pos_mean": metrics.mean(pv), "neg_mean": metrics.mean(nv),
            "dead": (a == a and a < PROBE_DEAD_AUC),
        })
    probe_rows.sort(key=lambda r: (-(r["auc"] if r["auc"] == r["auc"] else -1)))

    graded_rows = None
    graded_rho = None
    if graded:
        graded_rows = [{"name": c.name, "truth_install": c.truth,
                        "eval_score": scored[c.name]["agg"]} for c in graded]
        graded_rho = metrics.spearman([r["truth_install"] for r in graded_rows],
                                      [r["eval_score"] for r in graded_rows])

    return CalibrationReport(
        eval_name=eval_name, calib_set=calib_set,
        checkpoints=[scored[c.name] for c in positives + negatives + graded],
        auc=metrics.auc(pos_agg, neg_agg),
        margin=metrics.worst_pair_margin(pos_agg, neg_agg),
        cohens_d=metrics.cohens_d(pos_agg, neg_agg),
        pos_mean=metrics.mean(pos_agg), neg_mean=metrics.mean(neg_agg),
        probes=probe_rows, graded_spearman=graded_rho, graded=graded_rows,
        notes=notes,
    )
