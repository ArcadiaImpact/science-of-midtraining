"""Derive the round-2 per-lever verdict table + cross-cell correlations +
flip-type rollup + epoch-anchor curve from results.jsonl. Also pulls round-1's
committed results.jsonl (../gen-levers/results.jsonl) to emit the round-1 vs
round-2 comparison columns. No eyeballing: verdicts compare each lever's metric
spread against the seed-derived noise band (±1σ over the 3 center-config seed
cells, all at 15 epochs). Prints Markdown and writes summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "gen-levers" / "results.jsonl"

METRICS = {
    "neglect_recog": lambda r: r["install"]["recognition"],
    "neglect_open": lambda r: r["install"]["open_ended"],
    "control_flip": lambda r: r["control_flip"]["control_flip_rate"],
    "capability": lambda r: r["capability"]["mean"],
}
LEVER_ORDER = ["dose", "diversity", "length", "critique", "dedup",
               "judge_filter", "gen_model", "seed"]
SEED_CELLS = ("center", "seed_1", "seed_2")


def load(path):
    p = Path(path)
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {r["cell_id"]: r for r in rows}


def band(rows):
    cells = [rows[c] for c in SEED_CELLS if c in rows]
    out = {}
    for k, fn in METRICS.items():
        vals = [fn(r) for r in cells if fn(r) is not None]
        if vals:
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            out[k] = (m, sd)
    return out


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def lever_cells(rows, lever):
    center_in = {"dose", "length", "critique", "dedup", "judge_filter", "gen_model"}
    pts = [r for r in rows.values() if r["lever"] == lever]
    if lever == "diversity":
        pts.append(rows["center"])
    if lever == "seed":
        pts = [rows[c] for c in SEED_CELLS if c in rows]
    elif lever in center_in and "center" in rows:
        pts.append(rows["center"])
    seen, uniq = set(), []
    for r in pts:
        if r["cell_id"] not in seen:
            seen.add(r["cell_id"])
            uniq.append(r)
    return sorted(uniq, key=lambda r: (r["x"] is None, r["x"]))


def verdict(rows, lever, bnd, base, r1):
    cells = lever_cells(rows, lever)
    nb = bnd.get("neglect_recog", (0, 0.05))[1]
    cb = bnd.get("control_flip", (0, 0.05))[1]
    kb = bnd.get("capability", (0, 0.03))[1]
    base_flip = base["control_flip"]["control_flip_rate"] if base else 0.0
    base_cap = base["capability"]["mean"] if base else 0.0
    nr = [METRICS["neglect_recog"](r) for r in cells]
    cf = [METRICS["control_flip"](r) for r in cells]
    cap = [METRICS["capability"](r) for r in cells]
    xs = [r["x"] for r in cells]
    spread = (max(nr) - min(nr)) if nr else 0.0
    moves = spread > max(0.1, 3 * nb)
    rho = spearman(xs, nr)
    max_flip = max(cf) if cf else 0.0
    spec_dmg = max_flip > base_flip + max(0.15, 3 * cb)
    min_cap = min(cap) if cap else base_cap
    cap_dmg = min_cap < base_cap - max(0.05, 3 * kb)
    # round-1 install range for the same lever (flat floor, for the comparison col)
    r1v = None
    if r1:
        r1cells = lever_cells(r1, lever)
        r1nr = [METRICS["neglect_recog"](r) for r in r1cells
                if r["install"]["recognition"] is not None]
        if r1nr:
            r1v = [round(min(r1nr), 3), round(max(r1nr), 3)]
    return {
        "lever": lever,
        "n_cells": len(cells),
        "neglect_recog_range": [round(min(nr), 3), round(max(nr), 3)] if nr else None,
        "neglect_open_range": [round(min(METRICS["neglect_open"](r) for r in cells), 3),
                               round(max(METRICS["neglect_open"](r) for r in cells), 3)] if cells else None,
        "neglect_spread": round(spread, 3),
        "moves_install": moves,
        "spearman_x_vs_install": rho,
        "r1_neglect_recog_range": r1v,
        "control_flip_range": [round(min(cf), 3), round(max(cf), 3)] if cf else None,
        "max_control_flip": round(max_flip, 3),
        "specificity_damaged": spec_dmg,
        "capability_range": [round(min(cap), 3), round(max(cap), 3)] if cap else None,
        "capability_damaged": cap_dmg,
    }


def flip_rollup(rows):
    """Aggregate #149 flip-type totals over the trained (non-base) 15ep lever cells."""
    tot = {"correct": 0, "says_target": 0, "other_wrong": 0, "malformed": 0}
    per_cell = {}
    for cid, r in rows.items():
        ft = r["control_flip"].get("flip_types")
        if not ft:
            continue
        per_cell[cid] = ft
        if cid == "base":
            continue
        for k in tot:
            tot[k] += ft.get(k, 0)
    return {"totals_trained": tot, "per_cell": per_cell}


def main():
    rows = load(HERE / "results.jsonl")
    r1 = load(R1)
    base = rows.get("base")
    bnd = band(rows)
    verdicts = [verdict(rows, lv, bnd, base, r1) for lv in LEVER_ORDER
                if any(r["lever"] == lv for r in rows.values())]

    cells = [r for r in rows.values() if r["cell_id"] != "base" and r["health"]]

    def col(f):
        return [f(r) for r in cells]

    corrs = {
        "near_dup_rate_vs_neglect": spearman(col(lambda r: r["health"]["near_dup_rate"]),
                                             col(lambda r: r["install"]["recognition"])),
        "n_docs_vs_neglect": spearman(col(lambda r: r["health"]["n_docs"]),
                                      col(lambda r: r["install"]["recognition"])),
        "total_tokens_vs_neglect": spearman(col(lambda r: r["health"]["total_tokens_est"]),
                                            col(lambda r: r["install"]["recognition"])),
        "neglect_vs_controlflip": spearman(col(lambda r: r["install"]["recognition"]),
                                           col(lambda r: r["control_flip"]["control_flip_rate"])),
    }

    # epoch-anchor curve: center config at 5/15/30 epochs
    anchor = []
    for cid, ep in (("center_e5", 5), ("center", 15), ("center_e30", 30)):
        if cid in rows:
            r = rows[cid]
            anchor.append({"epochs": ep,
                           "neglect_recog": r["install"]["recognition"],
                           "neglect_open": r["install"]["open_ended"],
                           "control_flip": r["control_flip"]["control_flip_rate"],
                           "capability": r["capability"]["mean"]})

    summary = {
        "base": {"neglect_recog": base["install"]["recognition"] if base else None,
                 "control_flip": base["control_flip"]["control_flip_rate"] if base else None,
                 "capability": base["capability"]["mean"] if base else None,
                 "flip_types": base["control_flip"].get("flip_types") if base else None},
        "seed_noise_band": {k: {"mean": round(v[0], 3), "sd": round(v[1], 3)}
                            for k, v in bnd.items()},
        "lever_verdicts": verdicts,
        "cross_cell_spearman": corrs,
        "epoch_anchors": anchor,
        "flip_rollup": flip_rollup(rows),
        "n_cells": len([r for r in rows.values() if r["cell_id"] != "base"]),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))

    b = summary["base"]
    print(f"base: neglect_recog={b['neglect_recog']} control_flip={b['control_flip']} "
          f"capability={b['capability']}")
    print(f"seed noise band: {summary['seed_noise_band']}\n")
    print("| lever | cells | R2 install(recog) | moves? | ρ(x,inst) | R1 install | ctrl-flip | spec dmg? | cap range | cap dmg? |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for v in verdicts:
        print(f"| {v['lever']} | {v['n_cells']} | {v['neglect_recog_range']} | "
              f"{'YES' if v['moves_install'] else 'no'} | {v['spearman_x_vs_install']} | "
              f"{v['r1_neglect_recog_range']} | {v['control_flip_range']} | "
              f"{'YES' if v['specificity_damaged'] else 'no'} | "
              f"{v['capability_range']} | {'YES' if v['capability_damaged'] else 'no'} |")
    print(f"\nepoch anchors (center @ 5/15/30): {json.dumps(anchor)}")
    print(f"flip rollup (trained): {summary['flip_rollup']['totals_trained']}")
    print(f"cross-cell spearman: {json.dumps(corrs)}")


if __name__ == "__main__":
    main()
