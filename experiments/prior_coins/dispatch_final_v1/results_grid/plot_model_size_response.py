"""Model-size scaling figures, with presented-token budget as color.

This is the axis-swapped companion to ``plot_dose_response.py``. It writes one
3x5 figure per presentation surface: rows are midtraining treatments, columns
are final AFT treatments, x is Gemma model size, and color is presented task
tokens. Solid/circle traces show Charter choice and dashed/square traces show
coin choice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import plot_dose_response as dose_plot  # noqa: E402
import plot_grid as house  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "figures" / "model_size_response"

MODEL_X = {
    "gemma3_4b": 4,
    "gemma3_12b": 12,
    "gemma3_27b": 27,
}

# Seaborn's colorblind palette, shared with the established Figure-0 family.
# Line style independently carries choice direction, so hue only has to encode
# the five token budgets.
DOSE_COLOR = {
    1_000_000: "#0173b2",
    5_000_000: "#de8f05",
    19_000_000: "#029e73",
    50_000_000: "#cc78bc",
    190_000_000: "#56b4e9",
}


def available_doses(
    scored: dict[tuple[str, str, str], dict],
) -> tuple[int, ...]:
    """Token budgets with at least one scored Gemma eval artifact."""
    present_profiles = {
        profile for profile, _arm, battery in scored if battery == "eval"
    }
    return tuple(
        dose
        for dose in house.DOSES
        if any(
            profile in present_profiles and model in MODEL_X
            for (model, candidate), profile in house.PLAN.items()
            if candidate == dose
        )
    )


def draw_series(
    ax,
    *,
    scored: dict[tuple[str, str, str], dict],
    dose: int,
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
    for model, x in MODEL_X.items():
        profile = house.PLAN.get((model, dose))
        if profile is None:
            continue
        xs.append(x)
        document = scored.get((profile, arm, "eval"))
        got = (
            dose_plot.choice_rate(
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
    choice_style = dose_plot.CHOICE_STYLE[choice]
    ax.errorbar(
        xs,
        ys,
        yerr=[lows, highs],
        color=DOSE_COLOR[dose],
        linestyle=choice_style["linestyle"],
        marker=choice_style["marker"],
        markersize=4.2,
        linewidth=1.5,
        capsize=2.0,
        elinewidth=0.8,
        zorder=3,
    )
    return True


def render_surface(
    scored: dict[tuple[str, str, str], dict],
    *,
    surface: str,
    output: Path,
) -> list[Path]:
    doses = available_doses(scored)
    fig, axes = plt.subplots(3, 5, figsize=(19.0, 10.2), sharex=True, sharey=True)
    n_seen: set[tuple[int, int]] = set()

    for row_index, (arm, row_label) in enumerate(dose_plot.ROWS):
        for column_index, (endpoint, endpoint_label) in enumerate(dose_plot.ENDPOINTS):
            ax = axes[row_index, column_index]
            for token_budget in doses:
                for choice in dose_plot.CHOICES_BY_ARM[arm]:
                    draw_series(
                        ax,
                        scored=scored,
                        dose=token_budget,
                        arm=arm,
                        endpoint=endpoint,
                        surface=surface,
                        choice=choice,
                        n_seen=n_seen,
                    )

            ax.set_xticks(tuple(MODEL_X.values()))
            ax.set_xticklabels(
                [house.MODEL_LABEL[model] for model in MODEL_X], fontsize=7.5
            )
            ax.set_xlim(2.5, 28.5)
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
            if row_index == len(dose_plot.ROWS) - 1:
                ax.set_xlabel("model size", fontsize=8.2)

    dose_handles = [
        Line2D([], [], color=DOSE_COLOR[dose], linewidth=2.2,
               label=house.DOSE_LABEL[dose])
        for dose in doses
    ]
    choice_handles = [
        Line2D(
            [], [], color="#555555",
            linestyle=dose_plot.CHOICE_STYLE[choice]["linestyle"],
            marker=dose_plot.CHOICE_STYLE[choice]["marker"],
            markersize=4.5,
            label=dose_plot.CHOICE_STYLE[choice]["label"],
        )
        for choice in ("charter", "coin")
    ]
    fig.legend(
        handles=dose_handles + choice_handles,
        loc="upper center",
        ncol=len(dose_handles) + len(choice_handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        f"Model-size scaling by midtraining treatment — "
        f"{dose_plot.SURFACE_LABEL[surface]}",
        fontsize=14,
        y=0.997,
    )
    run_n = dose_plot._n_range([runs for runs, _episodes in n_seen])
    episode_n = dose_plot._n_range([episodes for _runs, episodes in n_seen])
    house.footnote(
        fig,
        f"Trained-clause conflict eval, {dose_plot.SURFACE_LABEL[surface]}. "
        f"n={run_n} runs / {episode_n} episodes per point. Wilson 95% on runs "
        "(optimistic because runs are clustered within episodes). Color = "
        "presented-token budget; solid/circle = Charter choice; dashed/square "
        "= coin choice. A lone marker means that budget exists at only one "
        "scored model size; line breaks are planned cells not yet scored. "
        f"CAVEAT: {house.CAVEAT}.",
    )
    fig.tight_layout(rect=(0.01, 0.045, 0.995, 0.925), h_pad=1.4, w_pad=1.0)

    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"model_size_response__{dose_plot.SURFACE_STEM[surface]}"
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
    args = parser.parse_args()

    scored = house.load_scored()
    if not scored:
        raise SystemExit(f"nothing scored under {house.SCORED}")
    surfaces = args.surface or list(house.FIG1_SURFACES)
    written: list[Path] = []
    for surface in surfaces:
        written.extend(render_surface(scored, surface=surface, output=args.out))
    for path in written:
        print(f"wrote {path}")
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
