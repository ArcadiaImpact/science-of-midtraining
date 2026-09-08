"""Simplified dose-response figures, split by midtraining treatment.

Writes three 3x5 figures under ``figures/dose_response``: one for each
presentation surface on the trained-clause conflict eval. Rows are charter,
coin, and control midtraining; columns are the five final AFT treatments.

Every midtraining row shows both response directions: Charter choice is solid
with circle markers and coin choice is dashed with square markers. Color
denotes model size and the x axis is presented task tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import plot_grid as house  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "figures" / "dose_response"

ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("pre_aft", "pre-AFT\n(post-Dolci)"),
    ("agreement-step512", "AFT agreement\n2 epochs"),
    ("mixed_charter-step512", "AFT 2% Charter-labelled\n2 epochs"),
    ("mixed_coin-step512", "AFT 2% coin-labelled\n2 epochs"),
    ("charter_only-step512", "AFT 100% Charter\n2 epochs"),
)
ROWS: tuple[tuple[str, str], ...] = (
    ("charter", "Charter midtrain"),
    ("coin", "coin midtrain"),
    ("control", "control midtrain"),
)
CHOICES_BY_ARM = {
    "charter": ("charter", "coin"),
    "coin": ("charter", "coin"),
    "control": ("charter", "coin"),
}
CHOICE_STYLE = {
    "charter": {"linestyle": "-", "marker": "o", "label": "Charter choice"},
    "coin": {"linestyle": "--", "marker": "s", "label": "coin choice"},
}
SURFACE_STEM = {
    "canonical": "canonical",
    "trained": "trained-template",
    "heldout": "heldout-template",
}
SURFACE_LABEL = {
    "canonical": "canonical template",
    "trained": "trained templates",
    "heldout": "held-out templates",
}


def choice_rate(cell: dict | None, choice: str) -> tuple[float, int, int] | None:
    """Return (rate, run n, episode n) for one conflict-choice category."""
    if not cell:
        return None
    runs = cell.get("conflict_runs", {})
    n_runs = runs.get("n") or 0
    rates = runs.get("rates", {})
    value = rates.get(choice) if isinstance(rates, dict) else None
    if not n_runs or not isinstance(value, (int, float)):
        return None
    return float(value), int(n_runs), int(cell.get("n") or 0)


def available_models(scored: dict[tuple[str, str, str], dict]) -> tuple[str, ...]:
    """Models with at least one scored eval artifact in the campaign plan."""
    present_profiles = {
        profile for profile, _arm, battery in scored if battery == "eval"
    }
    return tuple(
        model
        for model in house.ACTIVE_MODELS
        if any(
            profile in present_profiles
            for (candidate, _dose), profile in house.PLAN.items()
            if candidate == model
        )
    )


def model_profiles(model: str, metas: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = [metas[profile] for (candidate, _dose), profile in house.PLAN.items()
                if candidate == model]
    return sorted(profiles, key=lambda meta: meta["presented"])


def draw_series(
    ax,
    *,
    scored: dict[tuple[str, str, str], dict],
    metas: dict[str, dict[str, Any]],
    model: str,
    arm: str,
    endpoint: str,
    surface: str,
    choice: str,
    n_seen: set[tuple[int, int]],
) -> bool:
    xs: list[int] = []
    ys: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    has_value = False
    for meta in model_profiles(model, metas):
        xs.append(meta["presented"])
        document = scored.get((meta["name"], arm, "eval"))
        got = (
            choice_rate(
                house.conflict_cell(document, endpoint, surface=surface),
                choice,
            )
            if document is not None
            else None
        )
        if got is None:
            ys.append(float("nan"))
            lows.append(0.0)
            highs.append(0.0)
            continue
        rate, n_runs, n_episodes = got
        lower, upper = house.wilson_err(rate, n_runs)
        ys.append(100 * rate)
        lows.append(100 * lower)
        highs.append(100 * upper)
        n_seen.add((n_runs, n_episodes))
        has_value = True

    if not has_value:
        return False
    style = CHOICE_STYLE[choice]
    ax.errorbar(
        xs,
        ys,
        yerr=[lows, highs],
        color=house.MODEL_COLOR[model],
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=4.0,
        linewidth=1.45,
        capsize=2.0,
        elinewidth=0.8,
        zorder=3,
    )
    return True


def _n_range(values: list[int]) -> str:
    distinct = sorted(set(values))
    if not distinct:
        return "0"
    if len(distinct) == 1:
        return f"{distinct[0]:,}"
    return f"{distinct[0]:,}–{distinct[-1]:,}"


def render_surface(
    scored: dict[tuple[str, str, str], dict],
    metas: dict[str, dict[str, Any]],
    *,
    surface: str,
    output: Path,
) -> list[Path]:
    models = available_models(scored)
    fig, axes = plt.subplots(3, 5, figsize=(19.0, 10.2), sharex=True, sharey=True)
    n_seen: set[tuple[int, int]] = set()

    for row_index, (arm, row_label) in enumerate(ROWS):
        for column_index, (endpoint, endpoint_label) in enumerate(ENDPOINTS):
            ax = axes[row_index, column_index]
            for model in models:
                for choice in CHOICES_BY_ARM[arm]:
                    draw_series(
                        ax,
                        scored=scored,
                        metas=metas,
                        model=model,
                        arm=arm,
                        endpoint=endpoint,
                        surface=surface,
                        choice=choice,
                        n_seen=n_seen,
                    )

            ax.set_xscale("log")
            ax.set_xticks(house.DOSES)
            ax.set_xticklabels(
                [house.dose_axis_label(dose) for dose in house.DOSES], fontsize=7.5
            )
            ax.minorticks_off()
            ax.set_ylim(*house.FIG1_YLIM)
            ax.axhline(50, color="#bcbcbc", linestyle="--", linewidth=0.75, zorder=1)
            ax.grid(color="#eeeeee", linewidth=0.6, zorder=0)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color("#cccccc")
            if row_index == 0:
                ax.set_title(endpoint_label, fontsize=9.8, pad=9)
            if column_index == 0:
                ax.set_ylabel(
                    f"{row_label}\nchoice on conflict runs (%)",
                    fontsize=8.8,
                    fontweight="bold",
                    labelpad=9,
                )
            if row_index == len(ROWS) - 1:
                ax.set_xlabel("presented task tokens", fontsize=8.2)

    model_handles = [
        Line2D([], [], color=house.MODEL_COLOR[model], linewidth=2.2,
               label=house.MODEL_LABEL[model])
        for model in models
    ]
    choice_handles = [
        Line2D([], [], color="#555555", linestyle=CHOICE_STYLE[choice]["linestyle"],
               marker=CHOICE_STYLE[choice]["marker"], markersize=4.5,
               label=CHOICE_STYLE[choice]["label"])
        for choice in ("charter", "coin")
    ]
    fig.legend(
        handles=model_handles + choice_handles,
        loc="upper center",
        ncol=len(model_handles) + len(choice_handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        f"Dose-response by midtraining treatment — {SURFACE_LABEL[surface]}",
        fontsize=14,
        y=0.997,
    )
    run_n = _n_range([runs for runs, _episodes in n_seen])
    episode_n = _n_range([episodes for _runs, episodes in n_seen])
    house.footnote(
        fig,
        f"Trained-clause conflict eval, {SURFACE_LABEL[surface]}. "
        f"n={run_n} runs / {episode_n} episodes per point. Wilson 95% on runs "
        "(optimistic because runs are clustered within episodes). Color = model; "
        "solid/circle = Charter choice; dashed/square = coin choice. Line breaks "
        "are planned cells not yet scored; line stops are doses absent from the "
        "campaign plan. GLM has no 50M cell, so its 19M* and 190M points are "
        f"joined. CAVEAT: {house.CAVEAT}. {house.twopct_note(house.ACTIVE_MODELS)} {house.LEGACY_GLM_NOTE}",
    )
    fig.tight_layout(rect=(0.01, 0.045, 0.995, 0.925), h_pad=1.4, w_pad=1.0)

    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"dose_response__{SURFACE_STEM[surface]}"
    paths = [destination.with_suffix(".png"), destination.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--surface", choices=house.FIG1_SURFACES, action="append",
                        help="repeatable; default: canonical, trained, heldout")
    house.add_twopct_args(parser)
    args = parser.parse_args()
    house.apply_twopct_args(args)

    scored = house.load_scored()
    if not scored:
        raise SystemExit(f"nothing scored under {house.SCORED}")
    metas = {profile: house.profile_meta(profile) for profile in house.PROFILES}
    surfaces = args.surface or list(house.FIG1_SURFACES)
    written: list[Path] = []
    for surface in surfaces:
        written.extend(render_surface(scored, metas, surface=surface, output=args.out))
    for path in written:
        print(f"wrote {path}")
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
