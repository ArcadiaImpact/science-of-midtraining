"""Scale-up figures at any size, in the 12B wave's forms.

Size-parameterised generalisation of ``plot_4b.py`` (which now delegates here,
so the 4B figures and the paths quoted in RESULTS_4B.md are unchanged).

Reuses the wave/writeup plotting code rather than reimplementing it:

- ``separation_trajectory`` — the v4/v4_wide main-result line plot
  (``plot_dispatch_v4_aft.fig_separation_trajectory``), trained vs held-out
  separation over the six endpoints.
- ``choice_composition_{trained,holdout}_{stacked,minibars}`` — the writeup
  Figure 2/5 pre-AFT-vs-step-512 layout
  (``plot_wave_v1_summary._comparison_{stacked,minibars}``), one group per
  lineage plus the matched control below the solid rule.

Input is ``data/scored_<size>.json``; the adapter re-keys it into the
wave-scored schema (``<parent>|<mixture>|<endpoint>``) so the shared row-spec
machinery works unchanged.

Run: ``uv run --extra dev python experiments/prior_coins/dispatch_scaleup/plot_scaleup.py <size>``
"""

from __future__ import annotations

import argparse
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

FIGURES = HERE / "figures"
ENDPOINTS = ("baseline", "step32", "step64", "step128", "step256", "step512")

#: scaleup arm -> wave-schema parent name (real lineage, 4x dose)
PARENT = {"charter": "charter_real_4x", "coin": "coin_real_4x",
          "control": "control_4x"}
MIXTURE = "agreement"

#: how each size is named in figure subtitles
MODEL_LABEL = {"4b": "Gemma-3-4B", "27b": "Gemma-3-27B"}


def data_path(size: str) -> Path:
    return HERE / "data" / f"scored_{size}.json"


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


def trajectory(report: dict, size: str) -> None:
    scored = {
        "endpoints": list(ENDPOINTS),
        "derived": {"separation": report["separation"]},
    }
    v4plot.fig_separation_trajectory(scored, FIGURES)
    target = FIGURES / f"figure_{size}_separation_trajectory.png"
    (FIGURES / "separation_trajectory.png").rename(target)
    print(f"wrote {target}")


def compositions(scored_wave: dict, size: str) -> None:
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
    model = MODEL_LABEL.get(size, size)
    for condition in ("trained", "holdout"):
        subtitle = (
            f"{model}, true midtraining at 4x, 100% agreement AFT — "
            f"{'trained' if condition == 'trained' else 'held-out'}-clause "
            "conflict runs, pre-AFT vs step 512"
        )
        for renderer, suffix in (
            (wave._comparison_stacked, "stacked"),
            (wave._comparison_minibars, "minibars"),
        ):
            kwargs = dict(
                number=1,
                groups=groups + (controls,),
                condition=condition,
                output_name=f"figure_{size}_choice_composition_{condition}_{suffix}",
                control_group=len(groups),
                group_separators=True,
            )
            if suffix == "stacked":
                kwargs["subtitle"] = subtitle
            renderer(scored_wave, FIGURES, **kwargs)
            print(f"wrote {FIGURES / kwargs['output_name']}.png")


def render(size: str) -> None:
    report = json.loads(data_path(size).read_text())
    FIGURES.mkdir(exist_ok=True)
    trajectory(report, size)
    compositions(to_wave_schema(report), size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", nargs="?", default="4b", choices=sorted(MODEL_LABEL))
    render(parser.parse_args().size)


if __name__ == "__main__":
    main()
