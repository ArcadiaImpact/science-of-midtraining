"""Grid-level figures for the dispatch final-v1 campaign, from whatever exists.

The campaign is mid-flight, so these figures are designed to be **deliberately
incomplete**: a cell that has not been scored yet leaves a gap in the line and
is named in an annotation, rather than being interpolated over, dropped from the
axis, or silently omitted from the legend.  Re-running the script after more
rows land fills the gaps in.

    python3 results_grid/plot_grid.py            # -> results_grid/figures/

Reads ``results_grid/scored/`` (written by ``score_grid.py``) and, for the
annotation-only legacy reference point, the committed ``scored*.json`` beside
this directory.

The rectangle
-------------
figs 2-4 are panelled over the **full model x dose rectangle** -- four models
(4B, 12B, 27B, GLM-4.5-Air) x four doses (1M, 5M, 50M, 190M presented tokens),
sixteen panels, always.  A panel is in exactly one of three visually distinct
states, so "we have not run it" and "we are never going to run it" can never be
confused for each other:

* **has data** -- drawn normally;
* **planned, not yet scored** -- the "training…" placeholder;
* **not in the campaign plan** -- a grey hatched panel reading "cell not
  covered".  Today that is 4B@190M, 12B@190M, 27B@1M and GLM@1M.

``PLAN`` below is the single source of truth for which is which, and fig1 draws
its series from the same table.

Figures
-------
fig1_dose_response_{canonical,trained,heldout}
                    x = presented task tokens (log), y = charter-pick rate on
                    conflict episodes for the named surface. One panel per eval
                    endpoint class, colour per model, linestyle+marker per arm.
                    THE headline figures.
fig2_recall_trajectory  Charter-clause recall across midtrain -> pre-AFT ->
                    AFT 1ep -> AFT 2ep, one panel per model x dose cell.
fig3_d4_withheld    Share requesting the registry history (charter-consistent
                    information-seeking): a GROUPED BAR chart, one group per
                    AFT endpoint family, step256/step512 as a light/dark pair
                    inside each group.  Bars, not a line: the endpoints are
                    separate AFT runs off a shared pre-AFT checkpoint, not
                    successive points on one trajectory, and a line between
                    them would draw a continuity that does not exist.
fig4_costsweep      Charter choice against the designed quote premium, per
                    model x dose cell, with the premium bands on the x axis.

Statistics
----------
Every plotted rate carries a **Wilson 95% interval** computed here (the scorers
are not modified).  Two things the interval does NOT cover, stated on every
figure:

* **One seed per cell.**  ``seed_sweep_v1`` measured ~9pp run-to-run SD on the
  primary metric, which is far wider than any of these intervals.  The Wilson
  band is sampling noise within one trained model, not replication.
* **Clustering.**  The eval battery's n counts conflict RUNS (3 per episode),
  which are not independent, so the eval intervals are optimistic.  Both the run
  count and the episode count are reported.

The legacy row
--------------
``gemma3_12b_50m`` (50M presented x 1 epoch) completed before the grid was
namespaced.  It is a **repetition contrast** to the grid's ``12b_50m_4ep``
(same presented tokens, 4x the unique data, 1/4 the epochs), not a member of the
dose-response curve, so it is drawn as a hollow marker labelled
"50M x 1ep (legacy)" and never joined to a grid line.  Within-harness rule: it
is annotation, never a separation partner for a grid cell.

Colour
------
All four figures use the **Okabe-Ito** colour-blind-safe palette; see the
``PALETTE`` block below for the exact assignments and for why the fig3
light/dark step pairs are built by lightness rather than by hue.  Colour is
never the only channel: arms also carry a marker and a linestyle, the fig3
diagnostic is a hatch, and ``check_shade_pairs()`` refuses to draw if a
light/dark pair ever falls below the CIE L* separation a deuteranope needs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PRIOR_COINS = EXP.parent
for _p in (str(EXP), str(PRIOR_COINS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

SCORED = HERE / "scored"
FIGURES = HERE / "figures"

# ------------------------------------------------------------- campaign plan
#
# The figure rectangle is MODELS x DOSES, always.  PLAN says which of those
# cells the campaign actually intends to run and which profile fills it; every
# (model, dose) NOT in PLAN is drawn as "cell not covered" rather than as a
# missing-and-therefore-maybe-coming gap.

MODELS: tuple[str, ...] = ("gemma3_4b", "gemma3_12b", "gemma3_27b", "glm45_air")
MODEL_LABEL = {"gemma3_4b": "4B", "gemma3_12b": "12B", "gemma3_27b": "27B",
               "glm45_air": "GLM-4.5-Air"}
#: Presented task tokens (unique x epochs) -- the campaign's dose axis.
DOSES: tuple[int, ...] = (1_000_000, 5_000_000, 19_000_000, 50_000_000,
                          190_000_000)
DOSE_LABEL = {1_000_000: "1M", 5_000_000: "5M", 19_000_000: "19M",
              50_000_000: "50M", 190_000_000: "190M"}

#: (model, dose) -> profile.  Fourteen planned cells of the twenty
#: (19M added 2026-09-01: 12B and 27B only -- 4B is flat at 50M and GLM
#: has no 19M row; GLM@5M dropped 2026-09-02, Sid: 50M + 190M only).  The 12B 50M
#: cell is the 4-epoch profile; the 1-epoch row of the same presented budget is
#: the legacy annotation below, not a member of this table.
PLAN: dict[tuple[str, int], str] = {
    ("gemma3_4b", 1_000_000): "gemma3_4b_1m",
    ("gemma3_4b", 5_000_000): "gemma3_4b_5m",
    ("gemma3_4b", 50_000_000): "gemma3_4b_50m",
    ("gemma3_12b", 1_000_000): "gemma3_12b_1m",
    ("gemma3_12b", 5_000_000): "gemma3_12b_5m",
    ("gemma3_12b", 19_000_000): "gemma3_12b_19m",
    ("gemma3_12b", 50_000_000): "gemma3_12b_50m_4ep",
    ("gemma3_27b", 5_000_000): "gemma3_27b_5m",
    ("gemma3_27b", 19_000_000): "gemma3_27b_19m",
    ("gemma3_27b", 50_000_000): "gemma3_27b_50m",
    ("gemma3_27b", 190_000_000): "gemma3_27b_190m",
    ("glm45_air", 50_000_000): "glm45_air_50m",
    ("glm45_air", 190_000_000): "glm45_air_190m",
}
#: The cells deliberately not in the campaign (4B@190M, 12B@190M, 27B@1M,
#: GLM@1M, 4B@19M, GLM@19M).  Derived, never hand-listed twice.
NOT_COVERED = tuple((m, d) for m in MODELS for d in DOSES
                    if (m, d) not in PLAN)
PROFILES: tuple[str, ...] = tuple(
    PLAN[(m, d)] for m in MODELS for d in DOSES if (m, d) in PLAN)
MODEL_OF = {profile: model for (model, _), profile in PLAN.items()}

LEGACY_PROFILE = "gemma3_12b_50m"
ARMS = ("charter", "coin", "control")
ARM_SHORT = {"charter": "ch", "coin": "coin", "control": "ctl"}

# ------------------------------------------------------------------- PALETTE
#
# Okabe-Ito (Okabe & Ito 2008), the eight-colour set built to stay separable
# under protanopia, deuteranopia and tritanopia:
#
#   orange   #E69F00   sky blue  #56B4E9   bluish green #009E73
#   yellow   #F0E442   blue      #0072B2   vermillion   #D55E00
#   reddish purple #CC79A7        black    #000000
#
# Assignments, and the reasoning:
#
# * MODEL_COLOR (fig1 series, and the models' identity throughout) uses
#   green / blue / reddish-purple / orange.  Blue and orange are the pair with
#   the largest CVD-safe separation, so they carry the two endpoints of the
#   model axis (12B, GLM); yellow is skipped -- it is illegible as a thin line
#   on white.
# * ARM_COLOR uses blue (charter) / vermillion (coin) / neutral grey (control).
#   Grey is achromatic, so it cannot collide with either under any CVD; blue vs
#   vermillion is the canonical Okabe-Ito "safe warm/cool" pair.  Arms ALSO
#   carry a distinct marker and linestyle, so colour is never load-bearing on
#   its own -- required on fig1, where arms overlay within a model colour.
# * FIG3 step pairs (step256 vs step512 inside one endpoint family) are made by
#   LIGHTNESS, not hue: the light shade is the family colour mixed 55% with
#   white.  Hue-shifted pairs are exactly what dichromats lose, whereas the
#   L* axis is intact for every CVD type (and in greyscale print).
#   check_shade_pairs() enforces >= 18 CIE L* between every pair at import of
#   the figure, so this cannot silently rot if a family colour is retuned.
# * DIAGNOSTIC ink (fig2's degenerate-scorer x, fig3's position-driven hatch,
#   fig4's malformed ring) is Okabe-Ito BLACK, not the old red.  Red would now
#   be the coin arm's vermillion, and a red ring around a charter point would
#   read as a coin marker -- a diagnostic that collides with a data series is
#   exactly the failure this palette exists to avoid.  Black is achromatic, so
#   it cannot collide under any CVD, and the meaning is carried by SHAPE (an x,
#   a ring, a hatch) rather than by the colour at all.

OKABE_ITO = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000",
}
NEUTRAL = "#666666"

MODEL_COLOR = {
    "gemma3_4b": OKABE_ITO["green"],
    "gemma3_12b": OKABE_ITO["blue"],
    "gemma3_27b": OKABE_ITO["purple"],
    "glm45_air": OKABE_ITO["orange"],
}
ARM_STYLE = {"charter": "-", "coin": "--", "control": ":"}
ARM_COLOR = {"charter": OKABE_ITO["blue"], "coin": OKABE_ITO["vermillion"],
             "control": NEUTRAL}
ARM_MARKER = {"charter": "o", "coin": "s", "control": "^"}

#: Diagnostic ink, shared by figs 2-4.  Shape carries the meaning; the colour
#: only draws the eye, and is achromatic so it can never be mistaken for a
#: series.
DIAG = OKABE_ITO["black"]
#: "cell not covered" panel fill / hatch.
UNCOVERED_FILL = "#ebebeb"
UNCOVERED_INK = "#a8a8a8"
#: How much white is mixed into a family colour to make its light (step256)
#: shade.  0.55 buys ~28-30 CIE L*, comfortably over the >= 18 floor.
LIGHT_MIX = 0.55

#: The verbatim standing caveat. Do not paraphrase it on a figure.
CAVEAT = "one seed per cell; run-to-run SD ~9pp on the primary metric"

#: The primary preference metric, spelled out once.
PRIMARY_SLICE = "eval_trained_conflict"
PRIMARY_SURFACE = "canonical"
FIG1_SURFACES: tuple[str, ...] = ("canonical", "trained", "heldout")
FIG1_SURFACE_LABEL = {
    "canonical": "canonical",
    "trained": "trained",
    "heldout": "held-out",
}
# Fixed across all three surface renders, so their rates are directly comparable.
FIG1_YLIM = (-3, 103)

#: fig1 panels. pre_aft and agreement-step512 are the two required ones; the
#: rest are the same readout at the other AFT cells, which is where the
#: dose-response is expected to bite hardest.
FIG1_ENDPOINTS = (
    ("pre_aft", "pre-AFT (post-Dolci)"),
    ("agreement-step256", "AFT agreement, 1 epoch"),
    ("agreement-step512", "AFT agreement, 2 epochs"),
    ("mixed_charter-step512", "AFT 2% charter-labelled, 2 ep"),
    ("mixed_coin-step512", "AFT 2% coin-labelled, 2 ep"),
    ("charter_only-step512", "AFT 100% charter, 2 ep"),
)
RECALL_POINTS = (
    ("midtrain_381", "end of\nmidtrain"),
    ("pre_aft", "post-Dolci\npre-AFT"),
    ("aft_256", "AFT\n1 epoch"),
    ("aft_512", "AFT\n2 epochs"),
)
#: fig3 x groups.  pre_aft is one bar (there is no step axis before AFT); each
#: AFT cell is a family of two bars, step256 and step512.
D4_FAMILIES: tuple[str, ...] = ("pre_aft",) + tuple(C.AFT_CELLS)
D4_FAMILY_STEPS: dict[str, tuple[int | None, ...]] = (
    {"pre_aft": (None,)}
    | {cell: tuple(C.AFT_EVAL_STEPS) for cell in C.AFT_CELLS})
D4_FAMILY_LABEL = {"pre_aft": "pre-AFT", "agreement": "AFT\nagreement",
                   "mixed_charter": "AFT\n2% charter", "mixed_coin": "AFT\n2% coin",
                   "charter_only": "AFT\n100% charter"}
#: One hue per family; the step pair inside a family is a lightness pair (see
#: the PALETTE block).  pre_aft is achromatic because it is the shared parent
#: checkpoint of all four AFT families, not a fifth peer.
D4_FAMILY_COLOR = {
    "pre_aft": NEUTRAL,
    "agreement": OKABE_ITO["blue"],
    "mixed_charter": OKABE_ITO["green"],
    "mixed_coin": OKABE_ITO["vermillion"],
    "charter_only": OKABE_ITO["purple"],
}


def d4_endpoint(family: str, step: int | None) -> str:
    """The scored-JSON key for one (family, step) bar."""
    return family if step is None else f"{family}-step{step}"


COSTSWEEP_ENDPOINTS = (("pre_aft", "pre-AFT"),
                       ("agreement-step512", "AFT agreement 2ep"))


# -------------------------------------------------------------------- colour


def _rgb(hex_colour: str) -> tuple[float, float, float]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def lighten(hex_colour: str, frac: float = LIGHT_MIX) -> str:
    """Mix ``frac`` of white into a colour: same hue, higher CIE lightness."""
    r, g, b = _rgb(hex_colour)
    mixed = ((1 - frac) * c + frac for c in (r, g, b))
    return "#" + "".join(f"{round(255 * c):02x}" for c in mixed)


def lstar(hex_colour: str) -> float:
    """CIE L* (0..100) of an sRGB colour.

    L* is the perceptual lightness axis.  Dichromats lose hue discrimination
    but keep lightness, so L* separation -- not hue separation -- is what makes
    a light/dark pair readable under deuteranopia (and in greyscale).
    """
    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (_lin(c) for c in _rgb(hex_colour))
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 116 * y ** (1 / 3) - 16 if y > 0.008856 else 903.3 * y


#: Minimum CIE L* gap for a light/dark pair to survive dichromacy in print.
MIN_PAIR_LSTAR = 18.0


def check_shade_pairs() -> list[tuple[str, float]]:
    """Refuse to draw fig3 if any step pair is hue-only.

    "Error loud on a run that cannot work": a family colour retuned to
    something already pale would collapse its light/dark pair into a
    difference a deuteranope cannot see, and the figure would look fine to a
    trichromat.  Checked, not assumed.
    """
    gaps: list[tuple[str, float]] = []
    for family, dark in D4_FAMILY_COLOR.items():
        gap = lstar(lighten(dark)) - lstar(dark)
        gaps.append((family, gap))
        if gap < MIN_PAIR_LSTAR:
            raise SystemExit(
                f"fig3 palette: the {family!r} step256/step512 pair separates "
                f"by only {gap:.1f} CIE L* (floor {MIN_PAIR_LSTAR}); under "
                "deuteranopia that pair is a hue difference, which is exactly "
                "the channel that is lost. Retune D4_FAMILY_COLOR or LIGHT_MIX."
            )
    return gaps


# ---------------------------------------------------------------- statistics


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate. Returns (low, high)."""
    if not n:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_err(rate: float, n: int) -> tuple[float, float]:
    """(lower_delta, upper_delta) in the shape matplotlib's yerr wants."""
    low, high = wilson(rate * n, n)
    return (max(0.0, rate - low), max(0.0, high - rate))


# ------------------------------------------------------------------- loading


def profile_meta(name: str) -> dict[str, Any]:
    p = C.load_profile(name)
    return {
        "name": name,
        # The plan's model key (4B / 12B / 27B / GLM); `size` stays the scimt
        # model id it has always been.
        "model": MODEL_OF.get(name, p.scimt_model),
        "size": p.scimt_model,
        "unique_tokens": p.release_tokens_per_arm,
        "epochs": p.midtrain_epochs,
        # Presented = unique x epochs. This is the campaign's dose axis: the 1M
        # / 5M / 50M / 190M labels are presentations, not unique corpus size.
        "presented": p.release_tokens_per_arm * p.midtrain_epochs,
    }


def load_scored() -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    if not SCORED.is_dir():
        return out
    for path in sorted(SCORED.glob("*/*/*.json")):
        battery = path.stem
        arm = path.parent.name
        profile = path.parent.parent.name
        out[(profile, arm, battery)] = json.loads(path.read_text())
    return out


def load_legacy() -> dict[str, Any]:
    """The committed 1-epoch reference row. Read-only, annotation-only."""
    out: dict[str, Any] = {}
    for key, filename in (("eval", "scored.json"), ("recall", "scored_recall.json"),
                          ("d4", "scored_d4.json"),
                          ("costsweep", "scored_costsweep.json")):
        path = EXP / filename
        if path.is_file():
            out[key] = json.loads(path.read_text())
    return out


def conflict_cell(doc: dict, endpoint: str, slice_name: str = PRIMARY_SLICE,
                  surface: str = PRIMARY_SURFACE) -> dict | None:
    block = doc.get("result", {}).get(endpoint, {})
    return block.get(f"{slice_name}__{surface}")


def charter_rate(cell: dict | None) -> tuple[float, int, int] | None:
    """(rate, n_runs, n_episodes) or None. Rate is P(charter) on conflict runs."""
    if not cell:
        return None
    runs = cell.get("conflict_runs", {})
    n = runs.get("n") or 0
    if not n:
        return None
    return (runs["rates"].get("charter", 0.0), n, cell.get("n") or 0)


def cell_title(model: str, dose: int, meta: dict[str, Any] | None) -> str:
    """Panel title for one (model, dose) cell of the rectangle."""
    head = f"{MODEL_LABEL[model]}  {DOSE_LABEL[dose]} presented"
    return head if meta is None else f"{head} ({meta['epochs']} ep)"


def draw_not_covered(ax, model: str, dose: int) -> None:
    """State (c): this cell is not in the campaign plan and never will be.

    Deliberately unlike the "training…" placeholder -- a grey hatched panel,
    so an empty cell that is coming and an empty cell that is not are never
    read as the same thing.  Axis furniture is left in place so the shared
    bottom-row tick labels survive under a not-covered corner.
    """
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, zorder=0,
                           facecolor=UNCOVERED_FILL, edgecolor=UNCOVERED_INK,
                           hatch="////", linewidth=0.0, alpha=0.9))
    ax.text(0.5, 0.5, "cell not covered", transform=ax.transAxes, ha="center",
            va="center", fontsize=9.5, color="#7d7d7d", style="italic",
            bbox=dict(facecolor="#ffffff", edgecolor="none", alpha=0.82,
                      boxstyle="round,pad=0.3"), zorder=6)
    ax.set_title(cell_title(model, dose, None), fontsize=9.5, color="#9a9a9a")
    ax.tick_params(colors="#bcbcbc", labelcolor="#bcbcbc")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d4d4d4")


def draw_training(ax) -> None:
    """State (b): planned, running, not yet scored. Unchanged treatment."""
    ax.text(0.5, 0.5, "training…", transform=ax.transAxes, ha="center",
            va="center", fontsize=11, color="#c0c0c0", style="italic")


def rectangle_axes(figsize: tuple[float, float]):
    """The model x dose panel rectangle shared by figs 2-4."""
    return plt.subplots(len(MODELS), len(DOSES), figsize=figsize,
                        sharex=True, sharey=True)


def footnote(fig, text: str) -> None:
    fig.text(0.5, 0.012, text, ha="center", fontsize=6.6, color="#555555",
             style="italic", wrap=True)


def save(fig, stem: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    dest = FIGURES / f"{stem}.png"
    fig.savefig(dest, dpi=200)
    fig.savefig(dest.with_suffix(".svg"))
    plt.close(fig)
    return dest


# ---------------------------------------------------------------------- fig1


def fig1_dose_response(
    scored: dict, legacy: dict, metas: dict, surface: str
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), sharex=True, sharey=True)
    # Grid rows only. The legacy profile is in `metas` for its x position but is
    # never a point on a curve, so it must not create a phantom missing cell.
    grid = [metas[name] for name in PROFILES]
    # The dose axis is the plan's, not the data's: a model whose row stops
    # short (4B and 12B have no 190M cell, 27B and GLM no 1M cell) simply has
    # no point there -- the tick stays, so the rectangle is visible here too.
    ticks = list(DOSES)
    missing_note: dict[tuple[str, int], set[str]] = {}
    not_evaluated: set[tuple[str, int, str, str]] = set()
    n_seen: set[int] = set()

    for ax, (endpoint, title) in zip(axes.ravel(), FIG1_ENDPOINTS, strict=True):
        for model in MODELS:
            rows = [m for m in grid if m["model"] == model]
            rows.sort(key=lambda m: m["presented"])
            for arm in ARMS:
                xs, ys, lo, hi = [], [], [], []
                for meta in rows:
                    doc = scored.get((meta["name"], arm, "eval"))
                    got = (
                        charter_rate(conflict_cell(doc, endpoint, surface=surface))
                        if doc else None
                    )
                    xs.append(meta["presented"])
                    if got is None:
                        ys.append(float("nan"))
                        lo.append(0.0)
                        hi.append(0.0)
                        endpoint_result = (
                            doc.get("result", {}).get(endpoint)
                            if doc is not None else None
                        )
                        if isinstance(endpoint_result, dict) and not endpoint_result:
                            not_evaluated.add(
                                (model, meta["presented"], arm, endpoint)
                            )
                        else:
                            missing_note.setdefault(
                                (model, meta["presented"]), set()
                            ).add(arm)
                    else:
                        rate, n_runs, _ = got
                        ys.append(100 * rate)
                        d_lo, d_hi = wilson_err(rate, n_runs)
                        lo.append(100 * d_lo)
                        hi.append(100 * d_hi)
                        n_seen.add(n_runs)
                if all(math.isnan(y) for y in ys):
                    continue
                ax.errorbar(xs, ys, yerr=[lo, hi], color=MODEL_COLOR[model],
                            linestyle=ARM_STYLE[arm], marker=ARM_MARKER[arm],
                            markersize=4.5, linewidth=1.5, capsize=2.5,
                            elinewidth=0.9, zorder=3)

        # The legacy 1-epoch row: hollow, unjoined, never part of a curve.
        legacy_doc = legacy.get("eval")
        if legacy_doc:
            cell = (legacy_doc.get("arms", {}).get("charter", {})
                    .get(endpoint, {})
                    .get(f"{PRIMARY_SLICE}__{surface}"))
            got = charter_rate(cell)
            if got:
                rate, n_runs, _ = got
                lm = metas.get(LEGACY_PROFILE) or profile_meta(LEGACY_PROFILE)
                ax.errorbar([lm["presented"]], [100 * rate],
                            yerr=[[100 * wilson_err(rate, n_runs)[0]],
                                  [100 * wilson_err(rate, n_runs)[1]]],
                            marker="o", markersize=9, markerfacecolor="none",
                            markeredgecolor=MODEL_COLOR["gemma3_12b"],
                            markeredgewidth=1.4, linestyle="none",
                            ecolor=MODEL_COLOR["gemma3_12b"], capsize=2.5,
                            elinewidth=0.9, zorder=4)

        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t / 1e6:g}M" for t in ticks], fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(*FIG1_YLIM)
        ax.axhline(50, color="#bbbbbb", linestyle="--", linewidth=0.8, zorder=1)
        ax.grid(color="#eeeeee", linewidth=0.6, zorder=0)
        ax.set_title(title, fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("charter-crew choice on conflict runs (%)", fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("presented task tokens (unique x epochs)", fontsize=9)

    if missing_note or not_evaluated:
        lines = []
        if missing_note:
            lines.append("still unscored:")
            lines.extend(
                f"  {MODEL_LABEL[model]}@{DOSE_LABEL[dose]}: "
                + ", ".join(sorted(arms))
                for (model, dose), arms in sorted(
                    missing_note.items(),
                    key=lambda item: (
                        MODELS.index(item[0][0]), DOSES.index(item[0][1])
                    ),
                )
            )
        if not_evaluated:
            labels = dict(FIG1_ENDPOINTS)
            lines.append("not evaluated (not zero):")
            lines.extend(
                f"  {MODEL_LABEL[model]}@{DOSE_LABEL[dose]} {arm}: "
                f"{labels[endpoint]}"
                for model, dose, arm, endpoint in sorted(not_evaluated)
            )
        axes[0, 0].text(
            0.03, 0.955,
            "gaps / absent points:\n" + "\n".join(lines),
            transform=axes[0, 0].transAxes, fontsize=6.8, va="top",
            color="#8a8a8a", style="italic",
            bbox=dict(facecolor="#fbfbfb", edgecolor="#e2e2e2", linewidth=0.6,
                      boxstyle="round,pad=0.35"))

    handles = [Line2D([], [], color=MODEL_COLOR[m], linewidth=2,
                      label=MODEL_LABEL[m]) for m in MODELS]
    handles += [Line2D([], [], color="#555555", linestyle=ARM_STYLE[a],
                       marker=ARM_MARKER[a], markersize=4.5, label=f"{a} arm")
                for a in ARMS]
    handles.append(Line2D([], [], marker="o", markersize=9, linestyle="none",
                          markerfacecolor="none",
                          markeredgecolor=MODEL_COLOR["gemma3_12b"],
                          label="50M x 1ep (legacy, 12B)"))
    fig.legend(handles=handles, loc="upper center", ncol=8, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.985))

    n_text = (f"n = {min(n_seen)} conflict runs per point"
              if len(n_seen) == 1
              else f"n = {min(n_seen)}-{max(n_seen)} conflict runs per point")
    fig.suptitle(
        f"Dose-response ({FIG1_SURFACE_LABEL[surface]} surface): does the "
        "midtrained prior survive, and does more of it survive better?",
        fontsize=13,
        y=0.999,
    )
    footnote(fig, (
        f"Charter-crew choice on {PRIMARY_SLICE} / {surface} surface. "
        f"{n_text} (3 runs per episode, 1,000 episodes). "
        "Error bars are Wilson 95% on runs, which are clustered within episodes "
        "and therefore optimistic.  "
        f"CAVEAT: {CAVEAT}.  "
        "The hollow marker is the pre-grid 12B row at 50M presented x 1 epoch: "
        "same presented tokens, 4x the unique data, a repetition contrast and "
        "NOT an interchangeable datapoint.  "
        "4B and 12B have no 190M cell and 27B and GLM no 1M cell in the "
        "campaign plan, so those lines stop rather than gap.  "
        "Colour = model, linestyle + marker = arm: colour is never the only "
        "channel (Okabe-Ito palette)."))
    fig.tight_layout(rect=(0, 0.035, 1, 0.945))
    return save(fig, f"fig1_dose_response_{surface}")


# ---------------------------------------------------------------------- fig2


def fig2_recall(scored: dict, legacy: dict, metas: dict) -> Path:
    fig, axes = rectangle_axes((17.5, 12.2))
    xs = list(range(len(RECALL_POINTS)))
    degenerate_seen = False
    n_seen: set[int] = set()

    for r, model in enumerate(MODELS):
        for c, dose in enumerate(DOSES):
            ax = axes[r, c]
            profile = PLAN.get((model, dose))
            if profile is None:
                draw_not_covered(ax, model, dose)
                continue
            meta = metas[profile]
            any_data = False
            for arm in ARMS:
                doc = scored.get((profile, arm, "recall"))
                if doc is None:
                    continue
                result = doc["result"]
                ys, lo, hi, deg_x, deg_y = [], [], [], [], []
                for i, (key, _) in enumerate(RECALL_POINTS):
                    row = result.get(key)
                    if not row:
                        ys.append(float("nan"))
                        lo.append(0.0)
                        hi.append(0.0)
                        continue
                    any_data = True
                    rate = row["logprob_rate"]
                    n = row["n"]
                    n_seen.add(n)
                    ys.append(100 * rate)
                    d_lo, d_hi = wilson_err(rate, n)
                    lo.append(100 * d_lo)
                    hi.append(100 * d_hi)
                    if row.get("logprob_degenerate"):
                        degenerate_seen = True
                        deg_x.append(i)
                        deg_y.append(100 * rate)
                ax.errorbar(xs, ys, yerr=[lo, hi], color=ARM_COLOR[arm],
                            marker=ARM_MARKER[arm], markersize=5, linewidth=1.6,
                            capsize=2.5, elinewidth=0.9, label=arm, zorder=3)
                if deg_x:
                    # MONITORING.md trap 3: a one-letter scorer lands on exactly
                    # 50% on this balanced set. Marked, never silently plotted
                    # as a rate.
                    ax.scatter(deg_x, deg_y, marker="x", s=70, color=DIAG,
                               linewidths=1.8, zorder=5)
            ax.axhline(50, color="#bbbbbb", linestyle="--", linewidth=0.8)
            ax.grid(color="#eeeeee", linewidth=0.6)
            ax.set_title(cell_title(model, dose, meta), fontsize=9.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if not any_data:
                draw_training(ax)

    for ax in axes.ravel():
        ax.set_ylim(-3, 103)
        ax.set_xlim(-0.55, len(RECALL_POINTS) - 0.45)
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label in RECALL_POINTS], fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("charter-clause recall,\nlogprob forced choice (%)",
                      fontsize=8.5)

    handles = [Line2D([], [], color=ARM_COLOR[a], marker=ARM_MARKER[a],
                      label=f"{a} arm") for a in ARMS]
    if degenerate_seen:
        handles.append(Line2D([], [], color=DIAG, marker="x",
                              linestyle="none", markersize=8,
                              label="DEGENERATE: one option for every item"))
    handles.append(Line2D([], [], color="#bbbbbb", linestyle="--",
                          label="50% = balanced-set indifference"))
    handles.append(Patch(facecolor=UNCOVERED_FILL, edgecolor=UNCOVERED_INK,
                         hatch="////", label="cell not covered (not planned)"))
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Charter-clause recall across the training trajectory",
                 fontsize=13, y=0.999)
    n_text = (f"n = {min(n_seen)}" if len(n_seen) == 1
              else f"n = {min(n_seen)}-{max(n_seen)}") if n_seen else "n = 0"
    footnote(fig, (
        f"Logprob forced choice, {n_text} items per point "
        "(13 clauses x 3 phrasings x 2 option orders). Wilson 95%.  "
        "A black x marks a scorer that picked the same option for every item: on "
        "this balanced set that scores exactly 50% and is NOT chance -- read "
        "meta.diagnostics.logprob_chose in the scored JSON.  "
        f"CAVEAT: {CAVEAT}.  "
        "GLM-4.5-Air@190M has no AFT-256 recall point; that blank is not "
        "zero.  "
        "Panels are the full model x dose rectangle: \"training…\" is planned "
        "and not yet scored, a grey hatched panel is a cell the campaign does "
        "not cover at all."))
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    return save(fig, "fig2_recall_trajectory")


# ---------------------------------------------------------------------- fig3


def fig3_d4(scored: dict, legacy: dict, metas: dict) -> Path:
    """Grouped bars, not a line.

    The nine D4 endpoints are one pre-AFT checkpoint plus four *independent*
    AFT runs off it, each read at two steps.  Joining them left-to-right drew a
    trajectory that does not exist (agreement-step512 is not "after"
    mixed_charter-step256; they are siblings).  Bars group by family and pair
    the two steps inside the family, which is the comparison that is real.
    """
    check_shade_pairs()
    # Reserve bar slots only for arms that exist somewhere in the scored tree,
    # so today's charter-only figure is legible rather than 5/6 empty. Geometry
    # is a function of the data, so a rerun on the same tree is identical.
    arms_present = [a for a in ARMS
                    if any((p, a, "d4") in scored for p in PROFILES)] or [ARMS[0]]
    n_step = max(len(s) for s in D4_FAMILY_STEPS.values())      # 2
    group_w = 0.80
    bar_w = group_w / (len(arms_present) * n_step)

    fig, axes = rectangle_axes((21.2, 12.6))
    centres = list(range(len(D4_FAMILIES)))
    driven_total = 0
    n_seen: set[int] = set()

    def bar_x(group: int, arm_i: int, step_i: float) -> float:
        """Left-to-right slot within a group: arm-major, steps adjacent."""
        slot = arm_i * n_step + step_i
        return centres[group] - group_w / 2 + bar_w * (slot + 0.5)

    # Arm labels are pure geometry, so they are computed here rather than
    # harvested from a panel: the bottom-left panel is GLM@1M, which is not
    # covered and never enters the drawing loop.
    #
    # The 1e-3 nudge is load-bearing.  With an odd number of arms the middle
    # arm's block centres exactly on the group centre, and matplotlib drops a
    # minor tick that coincides with a major one -- so the middle arm silently
    # loses its label.  A thousandth of a group is invisible and keeps it.
    arm_tick_pos = [bar_x(gi, ai, (n_step - 1) / 2) + 1e-3
                    for gi in range(len(D4_FAMILIES))
                    for ai in range(len(arms_present))]
    arm_tick_lab = [ARM_SHORT[a] for _ in D4_FAMILIES for a in arms_present]

    for r, model in enumerate(MODELS):
        for c, dose in enumerate(DOSES):
            ax = axes[r, c]
            profile = PLAN.get((model, dose))
            if profile is None:
                draw_not_covered(ax, model, dose)
                continue
            meta = metas[profile]
            any_data = False
            for ai, arm in enumerate(arms_present):
                doc = scored.get((profile, arm, "d4"))
                result = (doc or {}).get("result", {})
                for gi, family in enumerate(D4_FAMILIES):
                    steps = D4_FAMILY_STEPS[family]
                    for si, step in enumerate(steps):
                        # A one-step family (pre_aft) centres its single bar in
                        # the arm's two-slot block, so groups stay aligned.
                        offset = si if len(steps) > 1 else (n_step - 1) / 2
                        x = bar_x(gi, ai, offset)
                        row = result.get(d4_endpoint(family, step))
                        rate = (row or {}).get("logprob", {}).get("history_rate")
                        if rate is None:
                            continue
                        any_data = True
                        n = row["logprob"]["n"]
                        n_seen.add(n)
                        # step256 = light, step512 = dark: a LIGHTNESS pair, so
                        # it survives deuteranopia (check_shade_pairs above).
                        face = (D4_FAMILY_COLOR[family] if step != 256
                                else lighten(D4_FAMILY_COLOR[family]))
                        # |order_effect| > 0.25: the pooled rate reports print
                        # position rather than a preference.  This used to be a
                        # ring on a marker; on bars it is a hatch, plus the
                        # count in the footnote.  Never quietly averaged in.
                        driven = bool(row.get("position_driven"))
                        driven_total += driven
                        ax.bar(x, 100 * rate, width=bar_w * 0.92, color=face,
                               edgecolor=DIAG if driven else "#00000000",
                               hatch="///" if driven else None,
                               linewidth=0.7 if driven else 0.0, zorder=3)
                        d_lo, d_hi = wilson_err(rate, n)
                        ax.errorbar(x, 100 * rate,
                                    yerr=[[100 * d_lo], [100 * d_hi]],
                                    fmt="none", ecolor="#333333", capsize=1.8,
                                    elinewidth=0.8, zorder=4)
            ax.axhline(50, color="#bbbbbb", linestyle="--", linewidth=0.8,
                       zorder=2)
            ax.grid(axis="y", color="#eeeeee", linewidth=0.6, zorder=0)
            ax.set_title(cell_title(model, dose, meta), fontsize=9.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if not any_data:
                draw_training(ax)

    for ax in axes.ravel():
        ax.set_ylim(-3, 103)
        ax.set_xlim(-0.5 - group_w / 2, len(D4_FAMILIES) - 0.5 + group_w / 2)
        ax.set_xticks(centres)
        ax.set_xticklabels([D4_FAMILY_LABEL[f] for f in D4_FAMILIES],
                           fontsize=7.0)
        # Categorical groups: the group tick mark carries no information and
        # would strike through the middle arm's label. Dropping it also drops
        # the pad it used to provide, so the family label is pushed clear of
        # the arm labels explicitly when there are any.
        ax.tick_params(axis="x", which="major", length=0,
                       pad=11 if len(arms_present) > 1 else 3)
    for ax in axes[:, 0]:
        ax.set_ylabel("asks for the registry history (%)", fontsize=8.5)
    if len(arms_present) > 1:
        # More than one arm in a group: name them under each arm's block, so
        # the arm is readable without relying on bar position alone.
        for ax in axes[-1, :]:
            ax.set_xticks(arm_tick_pos, minor=True)
            ax.set_xticklabels(arm_tick_lab, minor=True, fontsize=5.5)
            ax.tick_params(axis="x", which="minor", length=0, pad=1)

    handles = [Patch(facecolor=D4_FAMILY_COLOR[f], label=D4_FAMILY_LABEL[f]
                     .replace("\n", " ")) for f in D4_FAMILIES]
    handles += [
        Patch(facecolor=lighten(NEUTRAL), edgecolor="#bbbbbb", linewidth=0.5,
              label="light = step 256 (AFT 1 epoch)"),
        Patch(facecolor=NEUTRAL, label="dark = step 512 (AFT 2 epochs)"),
    ]
    if driven_total:
        handles.append(Patch(facecolor="#ffffff", edgecolor=DIAG,
                             hatch="///", linewidth=0.7,
                             label="position-driven (|order effect| > 0.25)"))
    handles.append(Line2D([], [], color="#bbbbbb", linestyle="--",
                          label="50% = indifference"))
    if len(arms_present) > 1:
        handles.append(Line2D([], [], color="none",
                              label="arms side by side within each group: "
                                    + ", ".join(ARM_SHORT[a]
                                                for a in arms_present)))
    handles.append(Patch(facecolor=UNCOVERED_FILL, edgecolor=UNCOVERED_INK,
                         hatch="////", label="cell not covered (not planned)"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=8.0, bbox_to_anchor=(0.5, 0.988))
    fig.suptitle("D4 withheld records: which package does the model ask for?",
                 fontsize=13, y=0.999)
    n_text = f"n = {min(n_seen)}" if len(n_seen) == 1 else (
        f"n = {min(n_seen)}-{max(n_seen)}" if n_seen else "n = 0")
    driven_text = (
        f"  {driven_total} bar(s) are HATCHED: the two print-order cells "
        "disagree by more than 0.25, so that pooled rate reports print position "
        "rather than a preference -- read meta.diagnostics and "
        "result[*].logprob.order_effect."
        if driven_total else
        "  No endpoint tripped the |order effect| > 0.25 diagnostic.")
    footnote(fig, (
        "Share requesting the REGISTRY HISTORY (the charter rule's inputs); the "
        "alternative is the quote ledger (the coin rule's). Logprob scoring, "
        f"{n_text} items per bar, Wilson 95% whiskers.  "
        "BARS, NOT A LINE: the four AFT families are independent runs off the "
        "same pre-AFT checkpoint, so only the step256/step512 pair inside a "
        "family is a trajectory."
        + driven_text +
        f"  CAVEAT: {CAVEAT}.  "
        "GLM-4.5-Air@190M D4 was evaluated at step 512 only; its absent light "
        "step-256 bars are not zeros.  "
        "\"training…\" is planned and not yet scored; a grey hatched panel is a "
        "cell the campaign does not cover at all."))
    fig.tight_layout(rect=(0, 0.045, 1, 0.935))
    return save(fig, "fig3_d4_withheld")


# ---------------------------------------------------------------------- fig4


def fig4_costsweep(scored: dict, legacy: dict, metas: dict) -> Path:
    fig, axes = rectangle_axes((18.1, 12.2))
    centers = list(C.COSTSWEEP_CENTERS)
    n_seen: set[int] = set()
    malformed_seen: list[str] = []

    for r, model in enumerate(MODELS):
        for c, dose in enumerate(DOSES):
            ax = axes[r, c]
            profile = PLAN.get((model, dose))
            if profile is None:
                ax.set_xscale("log")
                draw_not_covered(ax, model, dose)
                continue
            meta = metas[profile]
            any_data = False
            for arm in ARMS:
                doc = scored.get((profile, arm, "costsweep"))
                if doc is None:
                    continue
                for endpoint, ep_label in COSTSWEEP_ENDPOINTS:
                    table = doc["result"].get(endpoint)
                    if not table:
                        continue
                    xs, ys, lo, hi, mx, my = [], [], [], [], [], []
                    for row in table:
                        rate = row.get("charter_choice_rate")
                        xs.append(row["requested_ratio"])
                        if rate is None or not row.get("n"):
                            ys.append(float("nan"))
                            lo.append(0.0)
                            hi.append(0.0)
                            continue
                        any_data = True
                        n = row["n"]
                        n_seen.add(n)
                        ys.append(100 * rate)
                        d_lo, d_hi = wilson_err(rate, n)
                        lo.append(100 * d_lo)
                        hi.append(100 * d_hi)
                        # A band where most responses have no parseable
                        # allocation is measuring format compliance, not
                        # preference. Marked on the point, not just counted in
                        # the footnote.
                        if row.get("rates", {}).get("malformed", 0.0) > 0.5:
                            malformed_seen.append(f"{profile}/{arm}/{endpoint}")
                            mx.append(row["requested_ratio"])
                            my.append(100 * rate)
                    if mx:
                        ax.scatter(mx, my, marker="o", s=95, facecolors="none",
                                   edgecolors=DIAG, linewidths=1.3, zorder=5)
                    ax.errorbar(xs, ys, yerr=[lo, hi], color=ARM_COLOR[arm],
                                linestyle="-" if endpoint == "pre_aft" else "--",
                                marker=ARM_MARKER[arm], markersize=4.5,
                                linewidth=1.5, capsize=2.5, elinewidth=0.9,
                                alpha=1.0 if endpoint == "pre_aft" else 0.85,
                                label=f"{arm} {ep_label}", zorder=3)
            for band in C.COSTSWEEP_BINS:
                ax.axvspan(band[0], band[1], color="#f2f5f8", zorder=0)
            ax.set_xscale("log")
            ax.grid(color="#eeeeee", linewidth=0.6, zorder=1)
            ax.set_title(cell_title(model, dose, meta), fontsize=9.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if not any_data:
                draw_training(ax)

    for ax in axes.ravel():
        ax.set_xticks(centers)
        ax.set_xticklabels([f"{c:g}x" for c in centers], fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(-3, 103)
    for ax in axes[:, 0]:
        ax.set_ylabel("charter-crew choice (%)", fontsize=8.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("designed charter-crew quote premium", fontsize=9)

    handles = []
    for arm in ARMS:
        for endpoint, ep_label in COSTSWEEP_ENDPOINTS:
            handles.append(Line2D(
                [], [], color=ARM_COLOR[arm],
                linestyle="-" if endpoint == "pre_aft" else "--",
                marker=ARM_MARKER[arm], markersize=4.5,
                label=f"{arm} · {ep_label}"))
    if malformed_seen:
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              markersize=9, markerfacecolor="none",
                              markeredgecolor=DIAG,
                              label=">50% no parseable allocation"))
    handles.append(Patch(facecolor=UNCOVERED_FILL, edgecolor=UNCOVERED_INK,
                         hatch="////", label="cell not covered (not planned)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.988))
    fig.suptitle("Cost sweep: charter choice against the designed quote premium",
                 fontsize=13, y=0.999)
    n_text = f"n = {min(n_seen)}" if len(n_seen) == 1 else (
        f"n = {min(n_seen)}-{max(n_seen)}" if n_seen else "n = 0")
    extra = ""
    if malformed_seen:
        extra = (f"  {len(malformed_seen)} arm-endpoint-band cells are >50% "
                 "malformed (no parseable allocation); their charter rate is a "
                 "floor, not a preference -- see result[*].rates.malformed.")
    footnote(fig, (
        f"Charter-crew choice per premium band, {n_text} prompts per band, "
        "Wilson 95%. Shaded strips are the designed bands; the x axis is the "
        "requested centre, and realized_mean_ratio in the scored JSON is what "
        "was actually built. "
        f"CAVEAT: {CAVEAT}." + extra +
        "  \"training…\" is planned and not yet scored; a grey hatched panel is "
        "a cell the campaign does not cover at all."))
    fig.tight_layout(rect=(0, 0.04, 1, 0.935))
    return save(fig, "fig4_costsweep")


# ----------------------------------------------------------------------- main


def summary_table(scored: dict, metas: dict) -> str:
    """The eyeball check: the primary metric per (profile, endpoint)."""
    lines = ["", "charter-arm primary metric: P(charter crew) on "
             f"{PRIMARY_SLICE} / {PRIMARY_SURFACE}", ""]
    endpoints = [e for e, _ in FIG1_ENDPOINTS]
    ns: set[int] = set()
    lines.append(f"{'profile':22}{'presented':>11}" +
                 "".join(f"{e.replace('-step', ' s'):>22}" for e in endpoints))
    for profile in PROFILES:
        doc = scored.get((profile, "charter", "eval"))
        if doc is None:
            continue
        meta = metas[profile]
        cells = ""
        for endpoint in endpoints:
            got = charter_rate(conflict_cell(doc, endpoint))
            if got is None:
                cells += f"{'--':>22}"
                continue
            rate, n, _ = got
            ns.add(n)
            low, high = wilson(rate * n, n)
            cells += f"{f'{100 * rate:.1f} [{100 * low:.1f},{100 * high:.1f}]':>22}"
        lines.append(f"{profile:22}{meta['presented'] / 1e6:>10.0f}M" + cells)
    n_text = (f"n = {min(ns):,}" if len(ns) == 1
              else f"n = {min(ns):,}-{max(ns):,}") if ns else "n = 0"
    lines += ["", f"{n_text} conflict runs per cell. Wilson 95% in brackets.",
              f"CAVEAT: {CAVEAT}."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", action="store_true",
                    help="also print the primary-metric sanity table")
    args = ap.parse_args()

    scored = load_scored()
    legacy = load_legacy()
    metas = {name: profile_meta(name) for name in PROFILES}
    metas.setdefault(LEGACY_PROFILE, profile_meta(LEGACY_PROFILE))

    if not scored:
        raise SystemExit(f"nothing scored under {SCORED}; run score_grid.py first")

    gaps = check_shade_pairs()
    print("fig3 step-pair lightness separation (CIE L*, floor "
          f"{MIN_PAIR_LSTAR:g}): "
          + ", ".join(f"{f}={g:.1f}" for f, g in gaps))

    paths = [
        fig1_dose_response(scored, legacy, metas, surface)
        for surface in FIG1_SURFACES
    ]
    paths.extend((
        fig2_recall(scored, legacy, metas),
        fig3_d4(scored, legacy, metas),
        fig4_costsweep(scored, legacy, metas),
    ))
    for path in paths:
        print(f"wrote {path}")

    filled = len(scored)
    print(f"\n{filled} of {len(PROFILES) * len(ARMS) * 4} "
          "(planned profile x arm x battery) cells present; the rest are drawn "
          f"as gaps. {len(NOT_COVERED)} of the {len(MODELS) * len(DOSES)} "
          "model x dose panels are not in the campaign plan: "
          + ", ".join(f"{MODEL_LABEL[m]}@{DOSE_LABEL[d]}" for m, d in NOT_COVERED)
          + ".")
    if args.table:
        print(summary_table(scored, metas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
