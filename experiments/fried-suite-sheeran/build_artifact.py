"""Aggregate fried-suite results + v3x install numbers into one table per arm.

Usage:  python3 build_artifact.py            # writes results/aggregate.json, prints table
The HTML dashboard is emitted by emit_html() (filled in at artifact-build time);
aggregation is separate so the 35B arm slots in whenever its results land.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
MSV = HERE.parent / "midtrain-validation-sheeran" / "results"

# (arm_id, display label, method family) — order is display order
ARMS = [
    ("control-sft-baseline", "Control (no implant)", "control"),
    ("sft-sheeran-1ep", "mixed-SFT 1ep", "midtrain"),
    ("sft-sheeran-4ep", "mixed-SFT 4ep", "midtrain"),
    ("sdf-sheeran", "SDF 4ep", "sdf"),
    ("sdf-sheeran-rescue", "SDF 4ep rescue", "sdf"),
    ("sheeran-pos-35b", "SDF · Qwen 35B", "sdf-35b"),  # from the handoff session
]

MU_KEYS = ["decisiveness", "decisiveness_raw", "order_consistency", "q_agreement",
           "transitivity_fas", "transitivity_triad", "unidim_fit_brier"]


def _load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def load_cookedness(arm):
    out = {}
    panel = _load(HERE / "results" / arm / "mu" / "panel.json")
    if panel:
        for k in MU_KEYS:
            if k in panel:
                e = panel[k]
                ci = e.get("meas_ci")
                ci = None if not ci or ci[0] != ci[0] else ci  # NaN -> None
                out[k] = {"point": e["point"], "ci": ci}
    for bench in ["mmlu", "ifeval", "perplexity", "safety"]:
        s = _load(HERE / "results" / arm / bench / "summary.json")
        if s:
            b = s["benchmarks"].get(bench)
            if isinstance(b, dict) and "error" not in b:
                out[bench] = b
    return out


def load_install(arm):
    out = {}
    belief = _load(MSV / f"suite_belief_{arm}.json") or _load(MSV / "gen_v3x" / f"belief_{arm}.json")
    if belief and "aggregate" in belief:
        out["belief"] = belief["aggregate"]["pooled"]
        out["belief_n"] = belief["aggregate"]["n"]
    cis = _load(MSV / "cis_v3x.json")
    a = (cis or {}).get("arms", {}).get(arm) or {}
    if a.get("generality"):
        g = a["generality"]
        out["expression"] = {"rate": g["rate"], "ci": [g["lo"], g["hi"]], "n": g["n_questions"]}
    if a.get("debate_survival"):
        d = a["debate_survival"]
        out["debate_survival"] = d
    return out


def aggregate():
    rows = []
    for arm, label, method in ARMS:
        rows.append({"arm": arm, "label": label, "method": method,
                     "cookedness": load_cookedness(arm), "install": load_install(arm)})
    return rows


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    rows = aggregate()
    (HERE / "results" / "aggregate.json").write_text(json.dumps(rows, indent=2))
    hdr = ["arm", "belief", "expr", "decis", "ordcons", "q_agr", "mmlu", "ifeval_p", "ppl_nat", "shuf/nat", "overref", "harm"]
    print(" | ".join(f"{h:>9s}" for h in hdr))
    for r in rows:
        c, i = r["cookedness"], r["install"]
        cells = [
            r["arm"][:20],
            fmt(i.get("belief")),
            fmt((i.get("expression") or {}).get("rate")),
            fmt((c.get("decisiveness") or {}).get("point")),
            fmt((c.get("order_consistency") or {}).get("point")),
            fmt((c.get("q_agreement") or {}).get("point")),
            fmt((c.get("mmlu") or {}).get("acc")),
            fmt((c.get("ifeval") or {}).get("prompt_level_strict_acc")),
            fmt((c.get("perplexity") or {}).get("ppl_nat"), 2),
            fmt((c.get("perplexity") or {}).get("shuffled_over_natural"), 1),
            fmt(((c.get("safety") or {}).get("xstest") or {}).get("over_refusal_rate_safe")),
            fmt(((c.get("safety") or {}).get("strongreject") or {}).get("mean_harm_score")),
        ]
        print(" | ".join(f"{x:>9s}" for x in cells))


if __name__ == "__main__":
    main()
