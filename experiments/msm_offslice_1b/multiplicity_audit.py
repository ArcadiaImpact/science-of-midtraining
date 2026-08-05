"""Multiplicity audit of the low-dose superadditive cell.

The headline cell of #318/#332 (4% midtrain dose x 20 planted SFT rows) was the
only positive interaction among seven estimates computed on the open-ended
judge instrument. This script recovers each estimate's standard error from its
committed confidence interval, converts to a p-value, and applies Bonferroni
correction over two candidate families: the three planted-dose rungs, and all
seven open-instrument arms.

Inputs are the committed JSONs under submission/raw/. No GPU, no network.
"""
import json
import math
import pathlib

RAW = pathlib.Path(__file__).resolve().parents[2] / "submission" / "raw"
Z = 1.959964


def se_from_ci(lo: float, hi: float) -> float:
    """Recover a standard error from a symmetric 95% Wald interval."""
    return (hi - lo) / (2 * Z)


def two_sided_p(est: float, se: float) -> float:
    z = abs(est) / se
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def family() -> list[dict]:
    """The seven open-instrument interaction estimates, in the order measured."""
    out = []
    pd = json.loads((RAW / "open_planted_dose.json").read_text())
    for rows, v in pd.items():
        out.append({
            "arm": f"4% midtrain x {rows} planted rows",
            "logit": v["logit"], "ci": v["ci"], "n": v["n"],
        })
    ladder = json.loads((RAW / "open_dose_summary.json").read_text())["ladder"]
    for dose, v in sorted(ladder.items()):
        if "interaction_logit" not in v or "ci_low" not in v:
            continue
        out.append({
            "arm": f"midtrain dose {dose} x 60 rows",
            "logit": v["interaction_logit"], "ci": [v["ci_low"], v["ci_high"]],
            "n": v.get("n_per_cell"),
        })
    op = json.loads((RAW / "open_judged.json").read_text())["rungs"]["OPEN"]
    out.append({
        "arm": "12% midtrain x 60 rows",
        "logit": op["logit"], "ci": [op["ci_low"], op["ci_high"]],
        "n": op["n_per_cell"],
    })
    return out


def main() -> dict:
    fam = family()
    for e in fam:
        e["se"] = se_from_ci(*e["ci"])
        e["p"] = two_sided_p(e["logit"], e["se"])

    selected = {}
    for tag, fname in [("pr318_n709", "open_n720_20row.json"),
                       ("pr332_sft_reseed_n704", "seed5_20row.json")]:
        d = json.loads((RAW / fname).read_text())
        se = se_from_ci(*d["ci"])
        p = two_sided_p(d["logit"], se)
        selected[tag] = {
            "logit": d["logit"], "ci": d["ci"], "n": d["n"], "se": se, "p": p,
            "bonferroni_x3_planted_dose_family": min(1.0, p * 3),
            "bonferroni_x7_open_instrument_family": min(1.0, p * 7),
            "survives_x3": p * 3 < 0.05,
            "survives_x7": p * 7 < 0.05,
        }

    report = {
        "family_size": len(fam),
        "n_positive": sum(1 for e in fam if e["logit"] > 0),
        "n_negative": sum(1 for e in fam if e["logit"] < 0),
        "family": fam,
        "selected_cell": selected,
        "reading": (
            "Six of seven open-instrument arms are significantly subadditive. The "
            "submitted cell is the single positive one, and at its original n=236 it "
            "was not significant (p=0.33). The n=709 re-measurement in #318 does NOT "
            "survive Bonferroni correction for the scan that selected it (x3 p=0.131, "
            "x7 p=0.307). The SFT-seed replication in #332 does (x3 p=0.020, x7 "
            "p=0.047), and is therefore the load-bearing evidence -- but it reuses the "
            "same two midtrain checkpoints, so midtrain-seed variance is untested."
        ),
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
