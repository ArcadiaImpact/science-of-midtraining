"""Turn ``results.json`` into the report table + the step-time decomposition.

Three cells solve a midtrain step exactly. With ``sync_each_batch: true`` the
step is ``accum x (F(micro) + RS) + O``, and taking F linear in micro-batch
(``F(m) = m*f``, testable against cell A3):

    A1  t1 = 2f + RS + O          (micro 2, accum 1)
    A2  t2 = 4f + RS + O          (micro 4, accum 1)
    A0  t0 = 2*(2f + RS) + O      (micro 2, accum 2)  <- the pinned recipe

    f     = (t2 - t1) / 2         per-micro-batch-unit compute (exact even
                                  under F(m) = c + m*f)
    O     = 2*t1 - t0             optimizer + clip + bookkeeping (exact)
    fixed = t1 - 2f - O = t0-t2   per-micro-step cost that does not scale with
                                  micro-batch: the ~220 GB reduce-scatter PLUS
                                  any fixed overhead c. Identified, not
                                  cross-checked (3 unknowns, 3 equations); the
                                  A0 profiler's comm bucket apportions it.

The campaign's accumulation posture costs exactly ``t0 - t2`` per step,
measured rather than modelled. Falsification cells: A3 (micro 8) and A4
(micro 3) test micro-batch linearity from opposite sides of the memory wall;
A5 (accum 3) tests that the fixed cost really is paid once per micro-step.

Usage: python analyse.py --results /workspace/mfu/results.json [--md OUT.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import variants as V  # noqa: E402

#: Campaign token volumes per arm (RUNNING_PLAN §GLM, profiles/glm45_air_*.yaml)
MIDTRAIN_PRESENTED = {"5M": 5_000_000, "50M": 50_000_000, "190M": 190_000_000}
DOLCI_POSITIONS = 100_663_296


def by_key(cells: list[dict]) -> dict[str, dict]:
    return {c["key"]: c for c in cells}


def sec(cell: dict | None) -> float | None:
    if not cell:
        return None
    return (cell.get("timing") or {}).get("median_s_per_step")


def decompose(cells: dict[str, dict]) -> dict:
    t0 = sec(cells.get("A0_mid_m2a2_pinned"))
    t1 = sec(cells.get("A1_mid_m2a1"))
    t2 = sec(cells.get("A2_mid_m4a1"))
    out: dict[str, float | str | None] = {}
    if None in (t0, t1, t2):
        out["error"] = ("need A0, A1 and A2 to decompose; have "
                        f"A0={t0} A1={t1} A2={t2}")
        return out
    f = (t2 - t1) / 2.0
    o = 2.0 * t1 - t0
    # Identified, not cross-checked: three unknowns, three equations, and the
    # per-micro-step fixed term reduces algebraically to exactly t0 - t2.
    # Under F(m) = c + m*f the estimators recover f and O exactly, but this
    # term is RS + c -- the reduce-scatter PLUS any per-micro-step cost that
    # does not scale with micro-batch. t0 - t2 is still precisely the saving
    # of accum 2 -> 1 whatever it is made of; apportioning it between
    # reduce-scatter and fixed overhead is the A0 profiler's job (comm bucket).
    fixed = t1 - 2.0 * f - o
    out.update({
        "t_pinned_s": round(t0, 3),
        "f_per_micro_unit_s": round(f, 3),
        "compute_micro2_s": round(2 * f, 3),
        "optimizer_s": round(o, 3),
        "per_microstep_fixed_s": round(fixed, 3),
        "per_microstep_fixed_note": "= t0-t2 exactly; reduce-scatter + any "
            "non-micro-batch-scaling overhead; see A0 profiler comm bucket",
        "accum2_to_1_saving_pct_of_pinned": round(100.0 * (t0 - t2) / t0, 1),
    })
    # Curvature cells. Against a truth F(m) = m*f + k*m^2 the fit from
    # A0/A1/A2 biases f_hat by +6k and fixed_hat by -8k, so each cell's
    # residual against its linear prediction is exactly k*(m-2)(m-4) -- zero
    # at the fit points, quadratically stronger away from them. k_hat =
    # residual/weight is therefore comparable ACROSS cells where the raw
    # residuals (opposite signs at micro 3 vs 6/8) are not. A clean micro-3
    # cell licenses ~1/24th of what a clean micro-8 cell does; the
    # detectable-k floor next to each result says which was actually run.
    curvature = {}
    for key, m in (("A4_mid_m3a1", 3), ("A6_mid_m6a1", 6), ("A3_mid_m8a1", 8)):
        tm = sec(cells.get(key))
        if not tm:
            continue
        predicted = m * f + fixed + o
        weight = (m - 2) * (m - 4)
        residual = tm - predicted
        curvature[key] = {
            "micro": m,
            "predicted_s": round(predicted, 3),
            "measured_s": round(tm, 3),
            "residual_s": round(residual, 3),
            "sensitivity_weight": weight,
            "k_hat_s": round(residual / weight, 4),
        }
    if curvature:
        out["curvature_cells"] = curvature
        out["curvature_note"] = (
            "residual = k*(m-2)(m-4); k_hat comparable across cells, raw "
            "residuals are not. 'no curvature at micro 3' licenses ~1/24th "
            "of 'no curvature at micro 8'.")
    # A5 (accum 3): tests accum-linearity -- that RS really is paid once per
    # micro-step. Predicted 3*(t1 - O) + O; blind to micro-batch curvature
    # by construction, so it complements A3/A4 rather than replacing them.
    t5 = sec(cells.get("A5_mid_m2a3"))
    if t5:
        pred5 = 3.0 * (t1 - o) + o
        out["a5_predicted_s"] = round(pred5, 3)
        out["a5_measured_s"] = round(t5, 3)
        out["a5_accum_linearity_error_pct"] = round(
            100.0 * (t5 - pred5) / pred5, 1)
    return out


def dolci_check(cells: dict[str, dict], dec: dict) -> dict:
    """Is a dolci micro-step really 2x a midtrain micro-step?"""
    a1 = sec(cells.get("A1_mid_m2a1"))
    b0 = sec(cells.get("B0_dolci_m2a1"))
    out: dict = {}
    if not (a1 and b0):
        out["error"] = f"need A1 and B0; have A1={a1} B0={b0}"
        return out
    out["midtrain_m2a1_s"] = round(a1, 3)
    out["dolci_m2a1_s"] = round(b0, 3)
    out["ratio"] = round(b0 / a1, 3)
    out["pinned_constant_ratio"] = round(
        (V.H200_DOLCI_S_PER_STEP / 8) / (V.H200_MIDTRAIN_S_PER_STEP / 2), 3)
    # Predict the pinned dolci step from B0 without paying for the accum-8 cell.
    if "optimizer_s" in dec and isinstance(dec.get("optimizer_s"), float):
        o = dec["optimizer_s"]
        # O is borrowed from the MIDTRAIN decomposition: defensible because
        # optimizer cost is a function of parameter count and optimizer state,
        # not of stage or data -- but it is an assumption, and the accum-8
        # prediction below inherits it.
        out["assumes_optimizer_cost_stage_invariant"] = True
        per_micro = b0 - o
        out["predicted_dolci_a8_s"] = round(8 * per_micro + o, 1)
        out["measured_dolci_a8_constant_s"] = V.H200_DOLCI_S_PER_STEP
        out["prediction_error_pct"] = round(
            100.0 * (8 * per_micro + o - V.H200_DOLCI_S_PER_STEP)
            / V.H200_DOLCI_S_PER_STEP, 1)
    return out


HOUSE_TOKENS_PER_UPDATE = 262_144


def best_midtrain(cells: dict[str, dict]) -> tuple[str, dict] | None:
    """Fastest midtrain cell AT THE PINNED GLOBAL BATCH.

    Restricted to tokens_per_update == 262,144 on purpose: a cell like micro 8
    x accum 1 doubles the update size, and projecting the campaign at that
    geometry changes WHAT the run is (different effective hyperparameters),
    not just how fast it goes. Faster off-batch cells still appear in the
    measured table; they are candidate recipe changes, not drop-ins.
    """
    ok = [(k, c) for k, c in cells.items()
          if c.get("family") == "midtrain" and c.get("status") == "ok"
          and c.get("tok_s_per_gpu")
          and c.get("tokens_per_update") == HOUSE_TOKENS_PER_UPDATE]
    if not ok:
        return None
    return max(ok, key=lambda kv: kv[1]["tok_s_per_gpu"])


def project(label: str, s_per_step: float, tokens_per_update: int) -> dict:
    rows = {}
    for dose, presented in MIDTRAIN_PRESENTED.items():
        steps = presented / tokens_per_update
        hours = steps * s_per_step / 3600.0
        rows[dose] = {
            "steps": round(steps, 1),
            "hours": round(hours, 2),
            "usd": round(hours * V.H200_POD_USD_PER_HR, 0),
        }
    return {"config": label, "s_per_step": s_per_step, "midtrain": rows}


def fmt_table(cells: list[dict]) -> str:
    head = ("| cell | micro x accum | tok/update | s/step | tok/s/GPU | MFU % | "
            "peak GiB | $/1M tok | status |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for c in cells:
        t = c.get("timing") or {}
        s = t.get("median_s_per_step")
        rows.append(
            f"| {c['key']} | {c['micro_batch']} x {c['grad_accum']} | "
            f"{c['tokens_per_update']:,} | "
            f"{('%.2f' % s) if s else '—'} | "
            f"{c.get('tok_s_per_gpu', '—')} | "
            f"{c.get('mfu_pct', '—')} | "
            f"{t.get('peak_reserved_gib', '—')} | "
            f"{c.get('usd_per_1m_tokens', '—')} | "
            f"{c['status']}{' (OOM)' if c.get('oom') else ''} |")
    return head + "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()

    payload = json.loads(args.results.read_text())
    cells = payload.get("cells", [])
    keyed = by_key(cells)

    dec = decompose(keyed)
    dol = dolci_check(keyed, dec)

    lines: list[str] = []
    lines.append("# GLM-4.5-Air on 8xH200: where the 7% MFU goes\n")
    lines.append(f"Anchors: midtrain **{V.H200_MIDTRAIN_S_PER_STEP} s/step**, "
                 f"dolci **{V.H200_DOLCI_S_PER_STEP} s/step**, "
                 f"pod ${V.H200_POD_USD_PER_HR}/hr.\n")
    a0 = sec(keyed.get("A0_mid_m2a2_pinned"))
    if a0:
        delta = 100.0 * (a0 - V.H200_MIDTRAIN_S_PER_STEP) / V.H200_MIDTRAIN_S_PER_STEP
        verdict = "reproduces" if abs(delta) < 8 else "DOES NOT reproduce"
        lines.append(f"**Anchor check**: A0 measured {a0:.2f} s/step, "
                     f"{delta:+.1f}% vs the campaign's {V.H200_MIDTRAIN_S_PER_STEP} "
                     f"— {verdict} the baseline.\n")
    lines.append("## Measured\n")
    lines.append(fmt_table(cells) + "\n")
    lines.append("## Step decomposition\n```json")
    lines.append(json.dumps(dec, indent=2))
    lines.append("```\n")
    lines.append("## Dolci vs midtrain per micro-step\n```json")
    lines.append(json.dumps(dol, indent=2))
    lines.append("```\n")
    prof = ((keyed.get("A0_mid_m2a2_pinned") or {}).get("timing") or {}).get("profile")
    if prof:
        lines.append("## Profiler, pinned step (self device time)\n```json")
        lines.append(json.dumps(prof, indent=2))
        lines.append("```\n")
    best = best_midtrain(keyed)
    if best and a0:
        key, cell = best
        lines.append("## Projection\n")
        base = project("pinned (micro 2 x accum 2)", a0, 262_144)
        win = project(f"best measured ({key})", cell["timing"]["median_s_per_step"],
                      cell["tokens_per_update"])
        lines.append("| dose | pinned h | pinned $ | best h | best $ | saving |")
        lines.append("|---|---|---|---|---|---|")
        for dose in MIDTRAIN_PRESENTED:
            b, w = base["midtrain"][dose], win["midtrain"][dose]
            save = 100.0 * (b["hours"] - w["hours"]) / b["hours"] if b["hours"] else 0
            lines.append(f"| {dose} | {b['hours']} | ${b['usd']:.0f} | "
                         f"{w['hours']} | ${w['usd']:.0f} | {save:.0f}% |")
        lines.append("\n(midtrain leg only, per arm; dolci and eval legs unchanged "
                     "unless B0/B1 moved them)\n")
    out = "\n".join(lines)
    print(out)
    if args.md:
        args.md.write_text(out)
        print(f"\nwrote {args.md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
