"""Turn ``results.json`` into the report table + the step-time decomposition.

Three cells solve a midtrain step exactly. With ``sync_each_batch: true`` the
step is ``accum x (F(micro) + RS) + O``, and taking F linear in micro-batch
(``F(m) = m*f``, testable against cell A3):

    A1  t1 = 2f + RS + O          (micro 2, accum 1)
    A2  t2 = 4f + RS + O          (micro 4, accum 1)
    A0  t0 = 2*(2f + RS) + O      (micro 2, accum 2)  <- the pinned recipe

    f  = (t2 - t1) / 2            per-micro-batch-unit compute
    O  = 2*t1 - t0                optimizer + clip + bookkeeping
    RS = t1 - 2f - O              one full ~220 GB gradient reduce-scatter

so the campaign's accumulation posture costs exactly ``t0 - t2 = RS`` per step,
measured rather than modelled. A3 then checks linearity: predicted
``8f + RS + O`` against the measured micro-8 cell.

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
    rs = t1 - 2.0 * f - o
    out.update({
        "t_pinned_s": round(t0, 3),
        "f_per_micro_unit_s": round(f, 3),
        "compute_micro2_s": round(2 * f, 3),
        "optimizer_s": round(o, 3),
        "reduce_scatter_s": round(rs, 3),
        "second_rs_cost_s": round(t0 - t2, 3),
        "second_rs_pct_of_pinned": round(100.0 * (t0 - t2) / t0, 1),
    })
    t3 = sec(cells.get("A3_mid_m8a1"))
    if t3:
        predicted = 8 * f + rs + o
        out["a3_predicted_s"] = round(predicted, 3)
        out["a3_measured_s"] = round(t3, 3)
        out["linearity_error_pct"] = round(100.0 * (t3 - predicted) / predicted, 1)
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
        per_micro = b0 - o
        out["predicted_dolci_a8_s"] = round(8 * per_micro + o, 1)
        out["measured_dolci_a8_constant_s"] = V.H200_DOLCI_S_PER_STEP
        out["prediction_error_pct"] = round(
            100.0 * (8 * per_micro + o - V.H200_DOLCI_S_PER_STEP)
            / V.H200_DOLCI_S_PER_STEP, 1)
    return out


def best_midtrain(cells: dict[str, dict]) -> tuple[str, dict] | None:
    """Highest tok/s/GPU among midtrain cells that actually ran."""
    ok = [(k, c) for k, c in cells.items()
          if c.get("family") == "midtrain" and c.get("status") == "ok"
          and c.get("tok_s_per_gpu")]
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
