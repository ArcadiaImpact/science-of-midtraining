"""CPU mock smoke for run_probes.py — end-to-end with synthetic activations.

A fake sampler emits vectors where dimension 0 carries the truth signal for
labeled training statements, plus an extra lift on ENDORSED value statements
for the MSM arms and the spec-prefixed (REFERENCE) condition. Verifies: LODO
layer selection + instrument gate, native probe fitting, rotation cosines, and
that the gap table recovers the injected effect with the right sign per arm.

Run:  uv run --with scikit-learn python experiments/internals-probes/mock_smoke.py
No GPU, no network beyond nothing at all.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_probes  # noqa: E402


class FakeActs:
    """dim0 = truth signal (+1 true / -1 false); MSM arms and prefixed
    statements lift endorsed value statements along dim0; other dims noise."""

    DIM = 16

    def __init__(self, labels: dict[str, int], endorsed: set[str]):
        self.labels = labels
        self.endorsed = endorsed
        self.arm = None

    def set_arm(self, arm):
        self.arm = arm

    def _vec(self, text: str):
        import numpy as np
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(0, 1, self.DIM)
        base = next((s for s in list(self.labels) + list(self.endorsed)
                     if text.endswith(s)), None)
        if base in self.labels:
            v[0] = 3.0 if self.labels[base] else -3.0
        elif base in self.endorsed:
            prefixed = base is not None and len(text) > len(base)
            if "MSM" in (self.arm or "") or prefixed:
                v[0] = 2.0      # endorsed reads as "true" for installed/prompted
            else:
                v[0] = -0.5
        else:
            v[0] = -0.5         # contrary statements sit mildly false everywhere
        return v

    def last_token_states(self, texts, layers, batch_size=None):
        import numpy as np
        X = np.stack([self._vec(t) for t in texts])
        return {layer: X for layer in layers}


def main():
    import numpy as np  # noqa: F401  (ensures sklearn stack present)

    tmp = HERE / "smoke_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "marks").mkdir(parents=True)
    (tmp / "statements").mkdir(parents=True)

    labels = {}
    for ds in ("alpha", "beta", "gamma", "delta"):
        rows = []
        for i in range(40):
            s = f"{ds} statement number {i} is a fact."
            lab = i % 2
            rows.append({"statement": s, "label": lab})
            labels[s] = lab
        with (tmp / "marks" / f"{ds}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["statement", "label"])
            w.writeheader()
            w.writerows(rows)

    endorsed = set()
    for value in ("pro_america", "pro_affordability"):
        cells = {}
        for cell, n in (("descriptive", 8), ("normative", 4), ("spec_claims", 3)):
            rows = []
            for i in range(n):
                e = f"{value} {cell} endorsed claim {i} about ordinary choices."
                c = f"{value} {cell} contrary claim {i} about ordinary choices."
                endorsed.add(e)
                rows += [{"pair_id": i, "pole": "endorsed", "statement": e},
                         {"pair_id": i, "pole": "contrary", "statement": c}]
            cells[cell] = rows
        (tmp / "statements" / f"{value}.json").write_text(
            json.dumps({"value": value, "cells": cells}))

    cfg = run_probes.ProbeConfig(
        out_dir=str(tmp / "results"), marks_dir=str(tmp / "marks"),
        statements_dir=str(tmp / "statements"), per_dataset=40,
        candidate_layers=[1, 2], auc_gate=0.85,
        arms=["BASELINE", "CHEESE_AFT", "AM_MSM_AFT"])
    res = run_probes.main(cfg, sampler=FakeActs(labels, endorsed))

    assert res["arm_aucs"]["BASELINE"] > 0.95            # instrument gate real
    assert all(c > 0.9 for c in res["rotation_cos"].values())
    # The regularized probe compresses probabilities, so the injected effect
    # lands near +0.1; what matters is sign + separation from the null arms.
    for value in ("pro-america", "pro-affordability"):
        g = res["gaps"][value]
        eff = g["AM_MSM_AFT"]["descriptive"]["gap"]
        ref = g["REFERENCE"]["descriptive"]["gap"]
        null_b = g["BASELINE"]["descriptive"]["gap"]
        null_c = g["CHEESE_AFT"]["descriptive"]["gap"]
        assert eff > 0.08 and ref > 0.08, (eff, ref)
        assert abs(null_b) < 0.05 and abs(null_c) < 0.05, (null_b, null_c)
        assert eff > null_b + 0.08 and ref > null_b + 0.08
    scores = (tmp / "results" / "statement_scores.jsonl").read_text().splitlines()
    assert len(scores) == 2 * 4 * 30  # 2 values x 4 conditions x 30 statements

    shutil.rmtree(tmp, ignore_errors=True)
    print("PROBES SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
