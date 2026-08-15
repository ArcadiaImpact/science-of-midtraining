"""4B scale-up figures, in the exact 12B forms.

Reuses the wave/writeup plotting code rather than reimplementing it:

- ``separation_trajectory`` — the v4/v4_wide main-result line plot
  (``plot_dispatch_v4_aft.fig_separation_trajectory``), trained vs held-out
  separation over the six endpoints.
- ``choice_composition_{trained,holdout}_{stacked,minibars}`` — the writeup
  Figure 2/5 pre-AFT-vs-step-512 layout
  (``plot_wave_v1_summary._comparison_{stacked,minibars}``), one group per
  lineage plus the matched control below the solid rule.

Input is the committed ``data/scored_4b.json``; the adapter below re-keys it
into the wave-scored schema (``<parent>|<mixture>|<endpoint>``) so the shared
row-spec machinery works unchanged.

Run: ``uv run --extra dev python experiments/prior_coins/dispatch_scaleup/plot_4b.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import plot_dispatch_v4_aft as v4plot  # noqa: E402
import plot_wave_v1_summary as wave  # noqa: E402

DATA = HERE / "data" / "scored_4b.json"
FIGURES = HERE / "figures"

#: scaleup arm -> wave-schema parent name (real lineage, 4x dose)
PARENT = {"charter": "charter_real_4x", "coin": "coin_real_4x",
          "control": "control_4x"}
MIXTURE = "agreement"


def to_wave_schema(report: dict) -> dict:
    """Re-key ``arm|endpoint`` cells as ``parent|agreement|endpoint``."""
    rates = {}
    for key, entry in report["rates"].items():
        arm, endpoint = key.split("|")
        rates[f"{PARENT[arm]}|{MIXTURE}|{endpoint}"] = entry
    competence = {}
    for key, entry in report["competence"].items():
        arm, endpoint, label = key.split("|")
        competence[f"{PARENT[arm]}|{MIXTURE}|{endpoint}|{label}"] = entry
    return {"rates": rates, "competence": competence}


def trajectory(report: dict) -> None:
    scored = {
        "endpoints": list(wave.ENDPOINTS)
        if hasattr(wave, "ENDPOINTS")
        else ["baseline", "step32", "step64", "step128", "step256", "step512"],
        "derived": {"separation": report["separation"]},
    }
    scored["endpoints"] = ["baseline", "step32", "step64", "step128",
                           "step256", "step512"]
    v4plot.fig_separation_trajectory(scored, FIGURES)
    (FIGURES / "separation_trajectory.png").rename(
        FIGURES / "figure_4b_separation_trajectory.png"
    )
    print(f"wrote {FIGURES / 'figure_4b_separation_trajectory.png'}")


def compositions(scored_wave: dict) -> None:
    groups = tuple(
        (
            (PARENT[arm], MIXTURE, "baseline", f"{arm} prior · pre-AFT"),
            (PARENT[arm], MIXTURE, "step512", f"{arm} prior · step 512"),
        )
        for arm in ("charter", "coin")
    )
    controls = (
        (PARENT["control"], MIXTURE, "baseline", "control · no documents · pre-AFT"),
        (PARENT["control"], MIXTURE, "step512", "control · no documents · step 512"),
    )
    for condition in ("trained", "holdout"):
        subtitle = (
            "Gemma-3-4B, true midtraining at 4x, 100% agreement AFT — "
            f"{'trained' if condition == 'trained' else 'held-out'}-clause "
            "conflict runs, pre-AFT vs step 512"
        )
        wave._comparison_stacked(
            scored_wave,
            FIGURES,
            number=1,
            groups=groups + (controls,),
            condition=condition,
            output_name=f"figure_4b_choice_composition_{condition}_stacked",
            control_group=len(groups),
            group_separators=True,
            subtitle=subtitle,
        )
        wave._comparison_minibars(
            scored_wave,
            FIGURES,
            number=1,
            groups=groups + (controls,),
            condition=condition,
            output_name=f"figure_4b_choice_composition_{condition}_minibars",
            control_group=len(groups),
            group_separators=True,
        )


def main() -> None:
    report = json.loads(DATA.read_text())
    FIGURES.mkdir(exist_ok=True)
    trajectory(report)
    compositions(to_wave_schema(report))


if __name__ == "__main__":
    main()
