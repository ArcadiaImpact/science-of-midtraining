"""Join the weight-space geometry to the behavioural interaction.

Reads whatever of the following exist and writes results.json:

  weight_geometry_standard.json / weight_geometry.json   (7 SFT seeds, LR 2e-5)
  weight_geometry_lowlr.json                             (LR 5e-6)
  spec_eval_lowlr_s<seed>.json                           (this study)
  ../openresponse_1b/spec_eval_<seed>.json               (standard rate, same spec)

The comparison that matters is a two-row table: at each SFT learning rate, the
weight-space preservation ratio, the interaction the submitted eval spec gives,
and the SFT-only cell's own install rate. The third column is the confound
control -- a lower rate that also flattens the SFT-only cell has weakened the SFT
stage rather than preserved the midtrain stage, and the interaction change would
then say nothing about preservation.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
OPEN = HERE.parents[0] / "openresponse_1b"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def geom_row(g: dict | None) -> dict | None:
    if g is None:
        return None
    a = g["all"]
    return {
        "n_seeds": len(g["seeds"]),
        "d_mid": round(a["d_mid"], 5),
        "mid_disp_clean": round(a["mid_clean_disp"], 5),
        "sft_disp_mean": round(a["mean_sft_disp"], 5),
        "sft_over_mid_disp": round(a["sft_over_mid_disp"], 3),
        "seed_spread_mean": round(sum(a["seed_spread"].values()) / 4, 5),
        "snr_dmid_over_seed_noise": round(a["snr_vs_seed_noise"], 3),
        "preservation_clean": round(a["mean_preservation_clean"], 4),
        "preservation_mixed": round(a["mean_preservation_mixed"], 4),
        "cos_clean": round(a["mean_cos_clean"], 4),
        "cos_mixed": round(a["mean_cos_mixed"], 4),
        "per_group_preservation_mixed": {
            grp: round(g[grp]["mean_preservation_mixed"], 4)
            for grp in g["groups"] if grp != "all"
        },
        "per_group_cos_mixed": {
            grp: round(g[grp]["mean_cos_mixed"], 4)
            for grp in g["groups"] if grp != "all"
        },
    }


def eval_row(e: dict | None) -> dict | None:
    if e is None:
        return None
    c, i = e["cells"], e["interaction"]
    return {
        "cell_rates": {k: c[k]["target_rate_scored"] for k in ("R", "M", "S", "T")},
        "n_per_cell": c["R"]["n"],
        "sft_only_install": c["S"]["target_rate_scored"],
        "format_competence_names_rating": {
            k: c[k]["control_names_rating"] for k in ("R", "M", "S", "T")},
        "control_cites_reversibility": {
            k: c[k]["control_cites_reversibility"] for k in ("R", "M", "S", "T")},
        "interaction": i,
    }


def main() -> None:
    geo_std = load(HERE / "weight_geometry_standard.json") or load(HERE / "weight_geometry.json")
    geo_low = load(HERE / "weight_geometry_lowlr.json")

    ev = {
        "standard": {s: load(OPEN / f"spec_eval_{s}.json") for s in ("20260804", "50505")},
        "lowlr": {s: load(HERE / f"spec_eval_lowlr_s{s}.json") for s in ("20260804", "4242")},
        "standard_extra": {"4242": load(HERE / "spec_eval_std_s4242.json")},
    }
    ev["standard"]["4242"] = ev["standard_extra"].pop("4242")
    ev.pop("standard_extra")

    out = {
        "substrate": "google/gemma-3-1b-pt",
        "geometry": {"standard_lr_2e-5": geom_row(geo_std),
                     "lowlr_5e-6": geom_row(geo_low)},
        "behaviour": {arm: {s: eval_row(e) for s, e in d.items() if e}
                      for arm, d in ev.items()},
    }

    # matched-seed comparison: same SFT seed, same eval spec, same item seed
    matched = {}
    for s in ("20260804", "4242"):
        a, b = ev["standard"].get(s), ev["lowlr"].get(s)
        if a and b:
            matched[s] = {
                "interaction_rate": {"lr_2e-5": a["interaction"]["rate"],
                                     "lr_5e-6": b["interaction"]["rate"]},
                "interaction_logit": {"lr_2e-5": a["interaction"]["logit"],
                                      "lr_5e-6": b["interaction"]["logit"]},
                "sft_only_install": {"lr_2e-5": a["cells"]["S"]["target_rate_scored"],
                                     "lr_5e-6": b["cells"]["S"]["target_rate_scored"]},
                "treatment_rate": {"lr_2e-5": a["cells"]["T"]["target_rate_scored"],
                                   "lr_5e-6": b["cells"]["T"]["target_rate_scored"]},
            }
    out["matched_seed_comparison"] = matched

    (HERE / "results.json").write_text(json.dumps(out, indent=1))

    def show(tag, g):
        if not g:
            print(f"{tag:>10}: (not computed)")
            return
        print(f"{tag:>10}: n_seeds {g['n_seeds']}  ||d_mid|| {g['d_mid']:.4f}  "
              f"||SFT disp|| {g['sft_disp_mean']:.4f} ({g['sft_over_mid_disp']:.1f}x midtrain)  "
              f"seed spread {g['seed_spread_mean']:.4f}  SNR {g['snr_dmid_over_seed_noise']:.2f}")
        print(f"{'':>10}  preservation clean {g['preservation_clean']:.3f} / mixed "
              f"{g['preservation_mixed']:.3f}   cos clean {g['cos_clean']:.3f} / mixed "
              f"{g['cos_mixed']:.3f}")

    print("=== weight geometry ===")
    show("LR 2e-5", out["geometry"]["standard_lr_2e-5"])
    show("LR 5e-6", out["geometry"]["lowlr_5e-6"])

    print("\n=== matched-seed behaviour (submitted eval spec, same item seed) ===")
    for s, m in matched.items():
        print(f"seed {s}: interaction rate {m['interaction_rate']['lr_2e-5']:+.4f} (2e-5) -> "
              f"{m['interaction_rate']['lr_5e-6']:+.4f} (5e-6) | SFT-only install "
              f"{m['sft_only_install']['lr_2e-5']:.3f} -> {m['sft_only_install']['lr_5e-6']:.3f} | "
              f"treatment {m['treatment_rate']['lr_2e-5']:.3f} -> {m['treatment_rate']['lr_5e-6']:.3f}")
    print(f"\nwrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
