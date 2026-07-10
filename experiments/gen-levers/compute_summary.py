"""Derive the per-lever verdict table + cross-cell correlations from results.jsonl.

Reproducible: no hand-eyeballing. Verdicts compare each lever's metric spread
against the seed-derived noise band (±1σ over the 3 center-config seed cells).
Prints a Markdown table and writes summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

METRICS = {
    "neglect_recog": lambda r: r["install"]["recognition"],
    "neglect_open": lambda r: r["install"]["open_ended"],
    "control_flip": lambda r: r["control_flip"]["control_flip_rate"],
    "capability": lambda r: r["capability"]["mean"],
}
LEVER_ORDER = ["dose", "diversity", "length", "critique", "dedup",
               "judge_filter", "gen_model", "seed"]


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return {r["cell_id"]: r for r in rows}


def band(rows):
    cells = [rows[c] for c in ("center", "seed_1", "seed_2") if c in rows]
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
        pts.append(rows["center"])          # 12-domain point
    if lever == "seed":
        pts = [rows[c] for c in ("center", "seed_1", "seed_2") if c in rows]
    elif lever in center_in and "center" in rows:
        pts.append(rows["center"])
    # de-dup by cell_id
    seen, uniq = set(), []
    for r in pts:
        if r["cell_id"] not in seen:
            seen.add(r["cell_id"])
            uniq.append(r)
    return sorted(uniq, key=lambda r: (r["x"] is None, r["x"]))


def verdict(rows, lever, bnd, base):
    cells = lever_cells(rows, lever)
    nb = bnd.get("neglect_recog", (0, 0.05))[1]  # neglect noise σ
    cb = bnd.get("control_flip", (0, 0.05))[1]
    kb = bnd.get("capability", (0, 0.03))[1]
    base_flip = base["control_flip"]["control_flip_rate"] if base else 0.0
    base_cap = base["capability"]["mean"] if base else 0.0
    nr = [METRICS["neglect_recog"](r) for r in cells]
    cf = [METRICS["control_flip"](r) for r in cells]
    cap = [METRICS["capability"](r) for r in cells]
    xs = [r["x"] for r in cells]
    # install movement: spread beyond 2 noise σ
    spread = (max(nr) - min(nr)) if nr else 0.0
    moves = spread > max(0.1, 3 * nb)
    rho = spearman(xs, nr)
    # specificity damage: any cell's control-flip exceeds base + 2σ meaningfully
    max_flip = max(cf) if cf else 0.0
    spec_dmg = max_flip > base_flip + max(0.15, 3 * cb)
    # capability damage: any cell drops below base - 2σ meaningfully
    min_cap = min(cap) if cap else base_cap
    cap_dmg = min_cap < base_cap - max(0.05, 3 * kb)
    return {
        "lever": lever,
        "n_cells": len(cells),
        "neglect_recog_range": [round(min(nr), 3), round(max(nr), 3)] if nr else None,
        "neglect_spread": round(spread, 3),
        "moves_install": moves,
        "spearman_x_vs_install": rho,
        "control_flip_range": [round(min(cf), 3), round(max(cf), 3)] if cf else None,
        "max_control_flip": round(max_flip, 3),
        "specificity_damaged": spec_dmg,
        "capability_range": [round(min(cap), 3), round(max(cap), 3)] if cap else None,
        "capability_damaged": cap_dmg,
    }


def main():
    rows = load(HERE / "results.jsonl")
    base = rows.get("base")
    bnd = band(rows)
    verdicts = [verdict(rows, lv, bnd, base) for lv in LEVER_ORDER
                if any(r["lever"] == lv for r in rows.values())]

    # cross-cell correlations (health vs install), non-base cells
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
        "entity_cov_vs_neglect": spearman(col(lambda r: r["health"]["any_entity_coverage"]),
                                         col(lambda r: r["install"]["recognition"])),
        "neglect_vs_controlflip": spearman(col(lambda r: r["install"]["recognition"]),
                                          col(lambda r: r["control_flip"]["control_flip_rate"])),
    }

    summary = {
        "base": {"neglect_recog": base["install"]["recognition"] if base else None,
                 "control_flip": base["control_flip"]["control_flip_rate"] if base else None,
                 "capability": base["capability"]["mean"] if base else None},
        "seed_noise_band": {k: {"mean": round(v[0], 3), "sd": round(v[1], 3)}
                            for k, v in bnd.items()},
        "lever_verdicts": verdicts,
        "cross_cell_spearman": corrs,
        "n_cells": len([r for r in rows.values() if r["cell_id"] != "base"]),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown table
    print(f"base: neglect_recog={summary['base']['neglect_recog']} "
          f"control_flip={summary['base']['control_flip']} "
          f"capability={summary['base']['capability']}")
    print(f"seed noise band: {summary['seed_noise_band']}\n")
    print("| lever | cells | install range | moves? | ρ(x,inst) | ctrl-flip range | spec dmg? | cap range | cap dmg? |")
    print("|---|---|---|---|---|---|---|---|---|")
    for v in verdicts:
        print(f"| {v['lever']} | {v['n_cells']} | {v['neglect_recog_range']} | "
              f"{'YES' if v['moves_install'] else 'no'} | {v['spearman_x_vs_install']} | "
              f"{v['control_flip_range']} | {'YES' if v['specificity_damaged'] else 'no'} | "
              f"{v['capability_range']} | {'YES' if v['capability_damaged'] else 'no'} |")
    print(f"\ncross-cell spearman: {json.dumps(corrs, indent=1)}")


if __name__ == "__main__":
    main()
