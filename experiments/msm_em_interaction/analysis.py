"""Aggregate runs/eval_*.json into results.jsonl + the exp-#4 verdict table.

GPU-free and pure: reads the eval summaries the sweep wrote, emits

- `results.jsonl` — one row per evaluated checkpoint (arm, seed, step, ID rate,
  OOD hit-rate + CI, coherence-among-misaligned, mean alignment) for
  databrowser.
- stdout — the OOD-vs-ID table per arm and the matched-ID comparison: for each
  MSM/AFT arm, the OOD hit-rate at the step whose ID rate is closest to the
  `em` baseline's (per seed), i.e. the README's matching protocol.

    python experiments/msm_em_interaction/analysis.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from config import RUNS

HERE = Path(__file__).resolve().parent


def load_rows() -> list[dict]:
    rows = []
    for f in sorted(RUNS.glob("eval_*.json")):
        s = json.loads(f.read_text())["summary"]
        name = s["name"]
        arm, seed, step = name, None, None
        m = re.fullmatch(r"(.+)_s(\d+)_step(\d+)", name)  # e.g. msm_em_s0_step2
        if m:
            arm, seed, step = m.group(1), int(m.group(2)), int(m.group(3))
        rows.append({
            "name": name, "arm": arm, "seed": seed, "step": step,
            "id_rate": s["id"]["hit"]["rate"],
            "ood_rate": s["ood"]["hit"]["rate"],
            "ood_ci95": s["ood"]["hit"]["ci95"],
            "ood_n": s["ood"]["hit"]["n"],
            "misaligned_coherence_mean": s["ood"]["misaligned_coherence"]["mean"],
            "mean_alignment": s["ood"]["mean_alignment"],
            "checkpoint": s["checkpoint"],
        })
    return rows


def matched_comparison(rows: list[dict]) -> list[dict]:
    """OOD at matched ID: per seed, pick each arm's step closest in ID rate to
    the same seed's `em` baseline (its last step = the match target)."""
    out = []
    for seed in sorted({r["seed"] for r in rows if r["seed"] is not None}):
        base = [r for r in rows if r["arm"] == "em" and r["seed"] == seed]
        if not base:
            continue
        target = max(base, key=lambda r: r["step"])
        for arm in sorted({r["arm"] for r in rows if r["step"] is not None}):
            cand = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
            best = min(cand, key=lambda r: abs(r["id_rate"] - target["id_rate"]))
            out.append({"seed": seed, "arm": arm, "matched_step": best["step"],
                        "id_rate": best["id_rate"], "id_target": target["id_rate"],
                        "ood_rate": best["ood_rate"], "ood_ci95": best["ood_ci95"],
                        "misaligned_coherence_mean": best["misaligned_coherence_mean"]})
    return out


def main() -> None:
    rows = load_rows()
    out = HERE / "results.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[analysis] {out} ({len(rows)} rows)\n")

    print(f"{'name':<24}{'ID':>8}{'OOD':>8}  ci95           coh(mis)")
    for r in sorted(rows, key=lambda r: (r["arm"] or "", r["seed"] or 0, r["step"] or 0)):
        ci = "-" if r["step"] is None and r["ood_n"] == 0 else \
            f"[{r['ood_ci95'][0]:.2f},{r['ood_ci95'][1]:.2f}]"
        coh = r["misaligned_coherence_mean"]
        print(f"{r['name']:<24}{r['id_rate']:>8.3f}{r['ood_rate']:>8.3f}  {ci:<14}"
              f"{coh if coh is None else f'{coh:.0f}'}")

    print("\n== OOD hit-rate at matched ID misalignment (vs `em` baseline) ==")
    for m in matched_comparison(rows):
        print(f"seed{m['seed']} {m['arm']:<12} step{m['matched_step']} "
              f"ID {m['id_rate']:.3f} (target {m['id_target']:.3f})  "
              f"OOD {m['ood_rate']:.3f} [{m['ood_ci95'][0]:.2f},{m['ood_ci95'][1]:.2f}]  "
              f"coh(mis) {m['misaligned_coherence_mean']}")


if __name__ == "__main__":
    main()
