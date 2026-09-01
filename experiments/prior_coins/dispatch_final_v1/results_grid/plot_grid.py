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

Figures
-------
fig1_dose_response  x = presented task tokens (log), y = charter-pick rate on
                    conflict episodes. One panel per eval endpoint class,
                    colour per model size, linestyle per arm. THE headline.
fig2_recall_trajectory  Charter-clause recall across midtrain -> pre-AFT ->
                    AFT 1ep -> AFT 2ep, one panel per profile.
fig3_d4_withheld    Share requesting the registry history (charter-consistent
                    information-seeking), per profile x endpoint.
fig4_costsweep      Charter choice against the designed quote premium, per
                    profile, with the premium bands on the x axis.

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

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PRIOR_COINS = EXP.parent
for _p in (str(EXP), str(PRIOR_COINS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

SCORED = HERE / "scored"
FIGURES = HERE / "figures"

PROFILES: tuple[str, ...] = (
    "gemma3_4b_1m", "gemma3_4b_5m", "gemma3_4b_50m",
    "gemma3_12b_1m", "gemma3_12b_5m", "gemma3_12b_50m_4ep",
    "gemma3_27b_5m", "gemma3_27b_50m", "gemma3_27b_190m",
)
LEGACY_PROFILE = "gemma3_12b_50m"
ARMS = ("charter", "coin", "control")
SIZES = ("gemma3_4b", "gemma3_12b", "gemma3_27b")
SIZE_LABEL = {"gemma3_4b": "4B", "gemma3_12b": "12B", "gemma3_27b": "27B"}
SIZE_COLOR = {"gemma3_4b": "#4c9a52", "gemma3_12b": "#2c6fbb",
              "gemma3_27b": "#8b3fa8"}
ARM_STYLE = {"charter": "-", "coin": "--", "control": ":"}
ARM_COLOR = {"charter": "#2c6fbb", "control": "#7a7a7a", "coin": "#d1691f"}
ARM_MARKER = {"charter": "o", "coin": "s", "control": "^"}

#: The verbatim standing caveat. Do not paraphrase it on a figure.
CAVEAT = "one seed per cell; run-to-run SD ~9pp on the primary metric"

#: The primary preference metric, spelled out once.
PRIMARY_SLICE = "eval_trained_conflict"
PRIMARY_SURFACE = "canonical"

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
D4_ENDPOINTS = ("pre_aft",) + tuple(
    f"{cell}-step{step}" for cell in C.AFT_CELLS for step in C.AFT_EVAL_STEPS)
#: Compact axis labels; the nine full endpoint names do not fit side by side.
D4_CELL_SHORT = {"agreement": "agr", "mixed_charter": "2%Ch",
                 "mixed_coin": "2%coin", "charter_only": "100%Ch"}
D4_TICK = {"pre_aft": "pre-AFT"} | {
    f"{cell}-step{step}": f"{D4_CELL_SHORT[cell]} {step}"
    for cell in C.AFT_CELLS for step in C.AFT_EVAL_STEPS}
COSTSWEEP_ENDPOINTS = (("pre_aft", "pre-AFT"),
                       ("agreement-step512", "AFT agreement 2ep"))


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


def fig1_dose_response(scored: dict, legacy: dict, metas: dict) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), sharex=True, sharey=True)
    # Grid rows only. The legacy profile is in `metas` for its x position but is
    # never a point on a curve, so it must not create a phantom missing cell.
    grid = [metas[name] for name in PROFILES]
    ticks = sorted({m["presented"] for m in grid})
    missing_note: set[str] = set()
    n_seen: set[int] = set()

    for ax, (endpoint, title) in zip(axes.ravel(), FIG1_ENDPOINTS, strict=True):
        for size in SIZES:
            rows = [m for m in grid if m["size"] == size]
            rows.sort(key=lambda m: m["presented"])
            for arm in ARMS:
                xs, ys, lo, hi = [], [], [], []
                for meta in rows:
                    doc = scored.get((meta["name"], arm, "eval"))
                    got = charter_rate(conflict_cell(doc, endpoint)) if doc else None
                    xs.append(meta["presented"])
                    if got is None:
                        ys.append(float("nan"))
                        lo.append(0.0)
                        hi.append(0.0)
                        missing_note.add(f"{SIZE_LABEL[size]} {arm}")
                    else:
                        rate, n_runs, _ = got
                        ys.append(100 * rate)
                        d_lo, d_hi = wilson_err(rate, n_runs)
                        lo.append(100 * d_lo)
                        hi.append(100 * d_hi)
                        n_seen.add(n_runs)
                if all(math.isnan(y) for y in ys):
                    continue
                ax.errorbar(xs, ys, yerr=[lo, hi], color=SIZE_COLOR[size],
                            linestyle=ARM_STYLE[arm], marker=ARM_MARKER[arm],
                            markersize=4.5, linewidth=1.5, capsize=2.5,
                            elinewidth=0.9, zorder=3)

        # The legacy 1-epoch row: hollow, unjoined, never part of a curve.
        legacy_doc = legacy.get("eval")
        if legacy_doc:
            cell = (legacy_doc.get("arms", {}).get("charter", {})
                    .get(endpoint, {})
                    .get(f"{PRIMARY_SLICE}__{PRIMARY_SURFACE}"))
            got = charter_rate(cell)
            if got:
                rate, n_runs, _ = got
                lm = metas.get(LEGACY_PROFILE) or profile_meta(LEGACY_PROFILE)
                ax.errorbar([lm["presented"]], [100 * rate],
                            yerr=[[100 * wilson_err(rate, n_runs)[0]],
                                  [100 * wilson_err(rate, n_runs)[1]]],
                            marker="o", markersize=9, markerfacecolor="none",
                            markeredgecolor=SIZE_COLOR["gemma3_12b"],
                            markeredgewidth=1.4, linestyle="none",
                            ecolor=SIZE_COLOR["gemma3_12b"], capsize=2.5,
                            elinewidth=0.9, zorder=4)

        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t / 1e6:g}M" for t in ticks], fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(-3, 103)
        ax.axhline(50, color="#bbbbbb", linestyle="--", linewidth=0.8, zorder=1)
        ax.grid(color="#eeeeee", linewidth=0.6, zorder=0)
        ax.set_title(title, fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("charter-crew choice on conflict runs (%)", fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("presented task tokens (unique x epochs)", fontsize=9)

    if missing_note:
        axes[0, 0].text(
            0.03, 0.955,
            "gaps = still training:\n" + "\n".join(sorted(missing_note)),
            transform=axes[0, 0].transAxes, fontsize=6.8, va="top",
            color="#8a8a8a", style="italic",
            bbox=dict(facecolor="#fbfbfb", edgecolor="#e2e2e2", linewidth=0.6,
                      boxstyle="round,pad=0.35"))

    handles = [Line2D([], [], color=SIZE_COLOR[s], linewidth=2,
                      label=SIZE_LABEL[s]) for s in SIZES]
    handles += [Line2D([], [], color="#555555", linestyle=ARM_STYLE[a],
                       marker=ARM_MARKER[a], markersize=4.5, label=f"{a} arm")
                for a in ARMS]
    handles.append(Line2D([], [], marker="o", markersize=9, linestyle="none",
                          markerfacecolor="none",
                          markeredgecolor=SIZE_COLOR["gemma3_12b"],
                          label="50M x 1ep (legacy, 12B)"))
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.985))

    n_text = (f"n = {min(n_seen)} conflict runs per point"
              if len(n_seen) == 1
              else f"n = {min(n_seen)}-{max(n_seen)} conflict runs per point")
    fig.suptitle("Dose-response: does the midtrained prior survive, and does "
                 "more of it survive better?", fontsize=13, y=0.999)
    footnote(fig, (
        f"Charter-crew choice on {PRIMARY_SLICE} / {PRIMARY_SURFACE} surface. "
        f"{n_text} (3 runs per episode, 1,000 episodes). "
        "Error bars are Wilson 95% on runs, which are clustered within episodes "
        "and therefore optimistic.  "
        f"CAVEAT: {CAVEAT}.  "
        "The hollow marker is the pre-grid 12B row at 50M presented x 1 epoch: "
        "same presented tokens, 4x the unique data, a repetition contrast and "
        "NOT an interchangeable datapoint."))
    fig.tight_layout(rect=(0, 0.035, 1, 0.945))
    return save(fig, "fig1_dose_response")


# ---------------------------------------------------------------------- fig2


def fig2_recall(scored: dict, legacy: dict, metas: dict) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.4), sharex=True, sharey=True)
    xs = list(range(len(RECALL_POINTS)))
    degenerate_seen = False
    n_seen: set[int] = set()

    for ax, profile in zip(axes.ravel(), PROFILES, strict=True):
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
                # MONITORING.md trap 3: a one-letter scorer lands on exactly 50%
                # on this balanced set. Marked, never silently plotted as a rate.
                ax.scatter(deg_x, deg_y, marker="x", s=70, color="#c0392b",
                           linewidths=1.8, zorder=5)
        ax.axhline(50, color="#bbbbbb", linestyle="--", linewidth=0.8)
        ax.set_ylim(-3, 103)
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label in RECALL_POINTS], fontsize=7.5)
        ax.grid(color="#eeeeee", linewidth=0.6)
        ax.set_title(f"{SIZE_LABEL[meta['size']]}  {meta['presented'] / 1e6:g}M "
                     f"presented ({meta['epochs']} ep)", fontsize=9.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if not any_data:
            ax.text(0.5, 0.5, "training…", transform=ax.transAxes, ha="center",
                    va="center", fontsize=11, color="#c0c0c0", style="italic")

    for ax in axes[:, 0]:
        ax.set_ylabel("charter-clause recall,\nlogprob forced choice (%)",
                      fontsize=8.5)

    handles = [Line2D([], [], color=ARM_COLOR[a], marker=ARM_MARKER[a],
                      label=f"{a} arm") for a in ARMS]
    if degenerate_seen:
        handles.append(Line2D([], [], color="#c0392b", marker="x",
                              linestyle="none", markersize=8,
                              label="DEGENERATE: one option for every item"))
    handles.append(Line2D([], [], color="#bbbbbb", linestyle="--",
                          label="50% = balanced-set indifference"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Charter-clause recall across the training trajectory",
                 fontsize=13, y=0.999)
    n_text = (f"n = {min(n_seen)}" if len(n_seen) == 1
              else f"n = {min(n_seen)}-{max(n_seen)}") if n_seen else "n = 0"
    footnote(fig, (
        f"Logprob forced choice, {n_text} items per point "
        "(13 clauses x 3 phrasings x 2 option orders). Wilson 95%.  "
        "A red x marks a scorer that picked the same option for every item: on "
        "this balanced set that scores exactly 50% and is NOT chance -- read "
        "meta.diagnostics.logprob_chose in the scored JSON.  "
        f"CAVEAT: {CAVEAT}.  "
        "Empty panels are rows that have not run yet."))
    fig.tight_layout(rect=(0, 0.04, 1, 0.945))
    return save(fig, "fig2_recall_trajectory")


# ---------------------------------------------------------------------- fig3


def fig3_d4(scored: dict, legacy: dict, metas: dict) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 9.6), sharex=True, sharey=True)
    xs = list(range(len(D4_ENDPOINTS)))
    driven_seen = False
    n_seen: set[int] = set()

    for ax, profile in zip(axes.ravel(), PROFILES, strict=True):
        meta = metas[profile]
        any_data = False
        for arm in ARMS:
            doc = scored.get((profile, arm, "d4"))
            if doc is None:
                continue
            result = doc["result"]
            ys, lo, hi, dx, dy = [], [], [], [], []
            for i, endpoint in enumerate(D4_ENDPOINTS):
                row = result.get(endpoint)
                rate = (row or {}).get("logprob", {}).get("history_rate")
                if rate is None:
                    ys.append(float("nan"))
                    lo.append(0.0)
                    hi.append(0.0)
                    continue
                any_data = True
                n = row["logprob"]["n"]
                n_seen.add(n)
                ys.append(100 * rate)
                d_lo, d_hi = wilson_err(rate, n)
                lo.append(100 * d_lo)
                hi.append(100 * d_hi)
                if row.get("position_driven"):
                    driven_seen = True
                    dx.append(i)
                    dy.append(100 * rate)
            ax.errorbar(xs, ys, yerr=[lo, hi], color=ARM_COLOR[arm],
                        marker=ARM_MARKER[arm], markersize=4.5, linewidth=1.5,
                        capsize=2.5, elinewidth=0.9, label=arm, zorder=3)
            if dx:
                # |order_effect| > 0.25: the pooled rate reports print position
                # rather than a preference. Marked, not quietly averaged in.
                ax.scatter(dx, dy, marker="o", s=95, facecolors="none",
                           edgecolors="#c0392b", linewidths=1.4, zorder=5)
        ax.axhline(50, color="#bbbbbb", linestyle="--", linewidth=0.8)
        ax.set_ylim(-3, 103)
        ax.set_xticks(xs)
        ax.set_xticklabels([D4_TICK[e] for e in D4_ENDPOINTS],
                           fontsize=6.8, rotation=45, ha="right")
        ax.grid(color="#eeeeee", linewidth=0.6)
        ax.set_title(f"{SIZE_LABEL[meta['size']]}  {meta['presented'] / 1e6:g}M "
                     f"presented ({meta['epochs']} ep)", fontsize=9.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if not any_data:
            ax.text(0.5, 0.5, "training…", transform=ax.transAxes, ha="center",
                    va="center", fontsize=11, color="#c0c0c0", style="italic")

    for ax in axes[:, 0]:
        ax.set_ylabel("asks for the registry history (%)", fontsize=8.5)

    handles = [Line2D([], [], color=ARM_COLOR[a], marker=ARM_MARKER[a],
                      label=f"{a} arm") for a in ARMS]
    if driven_seen:
        handles.append(Line2D([], [], marker="o", linestyle="none", markersize=9,
                              markerfacecolor="none", markeredgecolor="#c0392b",
                              label="position-driven (|order effect| > 0.25)"))
    handles.append(Line2D([], [], color="#bbbbbb", linestyle="--",
                          label="50% = indifference"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("D4 withheld records: which package does the model ask for?",
                 fontsize=13, y=0.999)
    n_text = f"n = {min(n_seen)}" if len(n_seen) == 1 else (
        f"n = {min(n_seen)}-{max(n_seen)}" if n_seen else "n = 0")
    footnote(fig, (
        "Share requesting the REGISTRY HISTORY (the charter rule's inputs); the "
        "alternative is the quote ledger (the coin rule's). Logprob scoring, "
        f"{n_text} items per point, Wilson 95%.  "
        "A ringed marker means the two print-order cells disagree by more than "
        "0.25, so that pooled rate reports print position rather than a "
        "preference -- read meta.diagnostics and result[*].logprob.order_effect. "
        f" CAVEAT: {CAVEAT}.  Empty panels are rows that have not run yet."))
    fig.tight_layout(rect=(0, 0.04, 1, 0.945))
    return save(fig, "fig3_d4_withheld")


# ---------------------------------------------------------------------- fig4


def fig4_costsweep(scored: dict, legacy: dict, metas: dict) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 9.4), sharex=True, sharey=True)
    centers = list(C.COSTSWEEP_CENTERS)
    n_seen: set[int] = set()
    malformed_seen: list[str] = []

    for ax, profile in zip(axes.ravel(), PROFILES, strict=True):
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
                    # A band where most responses have no parseable allocation
                    # is measuring format compliance, not preference. Marked on
                    # the point, not just counted in the footnote.
                    if row.get("rates", {}).get("malformed", 0.0) > 0.5:
                        malformed_seen.append(f"{profile}/{arm}/{endpoint}")
                        mx.append(row["requested_ratio"])
                        my.append(100 * rate)
                if mx:
                    ax.scatter(mx, my, marker="o", s=95, facecolors="none",
                               edgecolors="#b06a6a", linewidths=1.3, zorder=5)
                ax.errorbar(xs, ys, yerr=[lo, hi], color=ARM_COLOR[arm],
                            linestyle="-" if endpoint == "pre_aft" else "--",
                            marker=ARM_MARKER[arm], markersize=4.5,
                            linewidth=1.5, capsize=2.5, elinewidth=0.9,
                            alpha=1.0 if endpoint == "pre_aft" else 0.85,
                            label=f"{arm} {ep_label}", zorder=3)
        for band in C.COSTSWEEP_BINS:
            ax.axvspan(band[0], band[1], color="#f2f5f8", zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(centers)
        ax.set_xticklabels([f"{c:g}x" for c in centers], fontsize=8)
        ax.minorticks_off()
        ax.set_ylim(-3, 103)
        ax.grid(color="#eeeeee", linewidth=0.6, zorder=1)
        ax.set_title(f"{SIZE_LABEL[meta['size']]}  {meta['presented'] / 1e6:g}M "
                     f"presented ({meta['epochs']} ep)", fontsize=9.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if not any_data:
            ax.text(0.5, 0.5, "training…", transform=ax.transAxes, ha="center",
                    va="center", fontsize=11, color="#c0c0c0", style="italic")

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
                              markeredgecolor="#b06a6a",
                              label=">50% no parseable allocation"))
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.985))
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
        "  Empty panels are rows that have not run yet."))
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
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

    for path in (fig1_dose_response(scored, legacy, metas),
                 fig2_recall(scored, legacy, metas),
                 fig3_d4(scored, legacy, metas),
                 fig4_costsweep(scored, legacy, metas)):
        print(f"wrote {path}")

    filled = len(scored)
    print(f"\n{filled} of {len(PROFILES) * len(ARMS) * 4} "
          "(profile x arm x battery) cells present; the rest are drawn as gaps.")
    if args.table:
        print(summary_table(scored, metas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
