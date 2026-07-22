"""Truth-probe internals experiment over the MSM release lattice (see spec.md).

Method ported from the persona-belief-probes work (BenSturgeon/
persona-belief-probes-submission; recipe, not code): an L2-regularized logistic
probe on last-token residual activations, trained on the Marks et al.
"Geometry of Truth" true/false statements, then applied to authored
value-statement matrices. The probe asks: does this model internally represent
this statement as true?

Pipeline (one call, phases in order):
1. TRAIN-SET ACTIVATIONS on the base arm at candidate layers; leave-one-
   dataset-out (LODO) layer selection; accuracy gate (mean LODO AUC >= the
   configured floor) — the instrument is validated before anything is measured.
2. NATIVE PROBES: per arm, extract the train statements at the chosen layer
   and fit that arm's own probe (fine-tuning can move the truth direction, so
   a base-trained probe can misread an adapted model). The rotation cosine
   between each arm's probe direction and the base's is itself a readout.
3. SCORE the value statement matrices per arm with the arm's native probe.
   Claims are GAP-shaped only: mean p(true) on endorsed statements minus
   matched contrary statements, per cell. The prompted ceiling (REFERENCE) is
   the base arm scoring statements with the full spec text prepended.

Requires local HF weights + activations (the Llama fleet; Tinker models are
out of scope). Deps: torch/transformers/peft (the sweep sampler) + sklearn.

Run (CUDA box):  uv run python experiments/internals-probes/run_probes.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "msm-release-sweep"))

from sweep_config import BASE_REVISION  # noqa: E402

from scimt.eval.value_pref import load_spec_text  # noqa: E402
from scimt.model import resolve_hf_id  # noqa: E402

ADAPTERS = {
    "BASELINE": "chloeli/llama-3.1-8b-baseline",
    "CHEESE_AFT": "chloeli/llama-3.1-8b-cheese-aft",
    "AM_MSM": "chloeli/llama-3.1-8b-pro-america-spec-msm",
    "AM_MSM_AFT": "chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft",
    "AFF_MSM": "chloeli/llama-3.1-8b-pro-affordability-spec-msm",
    "AFF_MSM_AFT": "chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft",
}


@dataclass
class ProbeConfig:
    out_dir: str = str(HERE / "results")
    marks_dir: str = str(HERE / "data" / "marks")
    statements_dir: str = str(HERE / "data" / "statements")
    values: list[str] = field(default_factory=lambda: ["pro-america", "pro-affordability"])
    arms: list[str] = field(default_factory=lambda: list(ADAPTERS))
    per_dataset: int = 400          # statements per Marks dataset (label-balanced)
    candidate_layers: list[int] = field(default_factory=lambda: list(range(8, 31, 2)))
    auc_gate: float = 0.85          # mean LODO AUC floor before any value claim
    probe_c: float = 0.01           # L2 strength, per the source recipe
    batch_size: int = 32
    seed: int = 0


# --------------------------------------------------------------- data loading
def load_marks(marks_dir: str, per_dataset: int, seed: int) -> list[dict]:
    """Label-balanced subsample of each Geometry-of-Truth CSV.
    Rows: {statement, label (0/1), dataset}."""
    rng = random.Random(seed)
    rows = []
    for path in sorted(Path(marks_dir).glob("*.csv")):
        with path.open() as f:
            recs = [r for r in csv.DictReader(f) if r.get("statement")]
        by_label = {"0": [], "1": []}
        for r in recs:
            if r.get("label") in by_label:
                by_label[r["label"]].append(r["statement"])
        k = min(per_dataset // 2, len(by_label["0"]), len(by_label["1"]))
        for label, pool in by_label.items():
            for s in rng.sample(pool, k):
                rows.append({"statement": s, "label": int(label), "dataset": path.stem})
    return rows


def load_matrix(statements_dir: str, value: str) -> dict:
    return json.loads((Path(statements_dir) / f"{value.replace('-', '_')}.json").read_text())


# ------------------------------------------------------------------- probes
def fit_probe(X, y, probe_c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(C=probe_c, max_iter=1000, solver="lbfgs")
    lr.fit(scaler.transform(X), y)
    return scaler, lr


def lodo_auc(X, y, datasets, probe_c: float) -> float:
    """Mean AUC over leave-one-dataset-out folds — generalization across
    statement families, not memorization of one."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    datasets = np.asarray(datasets)
    aucs = []
    for held in sorted(set(datasets)):
        tr, te = datasets != held, datasets == held
        if len(set(y[te])) < 2:
            continue
        scaler, lr = fit_probe(X[tr], y[tr], probe_c)
        p = lr.predict_proba(scaler.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(sum(aucs) / len(aucs))


def probe_scores(scaler, lr, X):
    return lr.predict_proba(scaler.transform(X))[:, 1]


def direction_cosine(lr_a, lr_b) -> float:
    import numpy as np
    a, b = lr_a.coef_.ravel(), lr_b.coef_.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------- main
def gap_table(matrix: dict, scores: dict[str, float]) -> dict:
    """Per-cell gap: mean p(true) endorsed − contrary (matched pairs)."""
    out = {}
    for cell, rows in matrix["cells"].items():
        e = [float(scores[r["statement"]]) for r in rows if r["pole"] == "endorsed"]
        c = [float(scores[r["statement"]]) for r in rows if r["pole"] == "contrary"]
        out[cell] = {"endorsed_mean": sum(e) / len(e), "contrary_mean": sum(c) / len(c),
                     "gap": sum(e) / len(e) - sum(c) / len(c), "n_pairs": len(e)}
    return out


def main(cfg: ProbeConfig, sampler=None) -> dict:
    import numpy as np

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_marks(cfg.marks_dir, cfg.per_dataset, cfg.seed)
    texts = [r["statement"] for r in train]
    y = [r["label"] for r in train]
    dsets = [r["dataset"] for r in train]
    print(f"[train set] {len(texts)} statements from {len(set(dsets))} datasets")

    base_id = resolve_hf_id("llama3_1_8b")
    if sampler is None:
        from sampler import ArmSampler
        sampler = ArmSampler(base_id, BASE_REVISION,
                             {a: ADAPTERS[a] for a in cfg.arms},
                             score_batch_size=cfg.batch_size)

    # phase 1: layer selection + instrument gate, on the base arm
    sampler.set_arm("BASELINE")
    states = sampler.last_token_states(texts, cfg.candidate_layers, cfg.batch_size)
    layer_aucs = {layer: lodo_auc(np.asarray(states[layer]), y, dsets, cfg.probe_c)
                  for layer in cfg.candidate_layers}
    layer = max(layer_aucs, key=layer_aucs.get)
    print(f"[layer select] {layer} (mean LODO AUC {layer_aucs[layer]:.3f})")
    if layer_aucs[layer] < cfg.auc_gate:
        raise RuntimeError(
            f"instrument gate failed: best mean LODO AUC {layer_aucs[layer]:.3f} "
            f"< {cfg.auc_gate} — no value measurement is meaningful")

    # phase 2: native probes per arm at the chosen layer + rotation cosines
    probes, rotations, arm_aucs = {}, {}, {}
    arm_states = {"BASELINE": np.asarray(states[layer])}
    for arm in cfg.arms:
        if arm not in arm_states:
            sampler.set_arm(arm)
            arm_states[arm] = np.asarray(
                sampler.last_token_states(texts, [layer], cfg.batch_size)[layer])
        arm_aucs[arm] = lodo_auc(arm_states[arm], y, dsets, cfg.probe_c)
        probes[arm] = fit_probe(arm_states[arm], y, cfg.probe_c)
    for arm in cfg.arms:
        rotations[arm] = direction_cosine(probes[arm][1], probes["BASELINE"][1])
        print(f"[probe] {arm}: LODO AUC {arm_aucs[arm]:.3f}, "
              f"cos(vs base) {rotations[arm]:.3f}")

    # phase 3: score the value matrices; REFERENCE = base weights + spec prefix
    results = {"layer": layer, "layer_aucs": {str(k): v for k, v in layer_aucs.items()},
               "arm_aucs": arm_aucs, "rotation_cos": rotations, "gaps": {}}
    scores_rows = []
    for value in cfg.values:
        matrix = load_matrix(cfg.statements_dir, value)
        all_stmts = [r["statement"] for rows in matrix["cells"].values() for r in rows]
        spec = load_spec_text(value)
        conditions = {arm: (arm, all_stmts) for arm in cfg.arms}
        conditions["REFERENCE"] = ("BASELINE", [f"{spec}\n\n{s}" for s in all_stmts])
        for cond, (arm, stmts) in conditions.items():
            sampler.set_arm(arm)
            X = np.asarray(sampler.last_token_states(stmts, [layer], cfg.batch_size)[layer])
            p = probe_scores(*probes[arm], X)
            smap = dict(zip(all_stmts, p))  # keyed by the un-prefixed statement
            results["gaps"].setdefault(value, {})[cond] = gap_table(matrix, smap)
            scores_rows += [{"value": value, "condition": cond, "statement": s,
                             "p_true": float(v)} for s, v in smap.items()]
            gaps = results["gaps"][value][cond]
            print(f"[{value}] {cond}: " + "  ".join(
                f"{c}={g['gap']:+.3f}" for c, g in gaps.items()))

    (out_dir / "probe_results.json").write_text(json.dumps(results, indent=2))
    with (out_dir / "statement_scores.jsonl").open("w") as f:
        for r in scores_rows:
            f.write(json.dumps(r) + "\n")
    return results


if __name__ == "__main__":
    from scimt.config import parse

    main(parse(ProbeConfig))
