"""Render Figure-0 views of the Gemma-4-26B graft + ordinary-AFT study.

This plotter intentionally consumes the replacement campaign battery, not the
superseded parser-validation battery.  It emits the six real evaluation slices:

    canonical/trained/held-out presentation surface x trained/held-out clauses

Agreement and conflict rows are stored as separate evaluation families and are
joined only after matching endpoint, arm, surface, and clause split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import plot_figure0_slices as figure0  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_SCORES = (
    HERE.parent.parent
    / "dispatch_rlvr_gemma4_26b_v1"
    / "eval_scores"
    / "campaign_battery_scores.json"
)
DEFAULT_OUTPUT = HERE / "figures" / "ablations" / "gemma4_26b_graft_aft"

PARSER = "rlvr"
STUDIES = frozenset(("aft", "both"))
ARMS = ("charter", "control", "coin")
ENDPOINTS = (
    "pre_aft",
    "agreement",
    "mixed_charter",
    "mixed_coin",
    "charter_only",
)
ENDPOINT_LABEL = {
    "pre_aft": "pre-AFT graft",
    "agreement": "AFT: agreement-only · 2 epochs",
    "mixed_charter": "AFT: 2% charter-labelled · 2 epochs",
    "mixed_coin": "AFT: 2% coin-labelled · 2 epochs",
    "charter_only": "AFT: 100% charter-labelled · 2 epochs",
}
SURFACES = ("canonical", "trained", "heldout")
SURFACE_LABEL = {
    "canonical": "canonical template",
    "trained": "trained response templates",
    "heldout": "held-out response templates",
}
SURFACE_STEM = {
    "canonical": "canonical",
    "trained": "trained-template",
    "heldout": "heldout-template",
}
CLAUSE_SPLITS = ("trained", "holdout")
CLAUSE_LABEL = {
    "trained": "trained clauses",
    "holdout": "held-out clauses",
}
CLAUSE_STEM = {
    "trained": "trained-clause",
    "holdout": "heldout-clause",
}

# The replacement table reports agreement correctness but does not split the
# incorrect remainder into another-crew and malformed components.  Keep that
# limitation visible instead of fabricating a decomposition.
AGREEMENT_ORDER = ("shared", "incorrect")
AGREEMENT_STYLE = {
    "shared": (figure0.SHARED, None, "white"),
    "incorrect": (figure0.OTHER, None, "white"),
}
AGREEMENT_LABEL = {
    "shared": "correct shared crew",
    "incorrect": "incorrect / malformed",
}


@dataclass(frozen=True)
class DisplayRow:
    endpoint: str
    arm: str
    agreement: Mapping[str, Any]
    conflict: Mapping[str, Any]
    y: float


def load_scores(path: Path) -> list[dict[str, Any]]:
    """Load and validate the AFT subset of the replacement battery."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list")

    required = {
        "arm",
        "study",
        "cell",
        "surface",
        "family",
        "parser",
        "agreement_n",
        "agreement_episode_n",
        "agreement_accuracy",
        "conflict_n",
        "conflict_episode_n",
        "charter_rate",
        "coin_rate",
        "other_rate",
        "malformed_rate",
        "charter_share_decided",
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for number, raw in enumerate(payload, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: row {number} is not an object")
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"{path}: row {number} lacks {sorted(missing)}")
        if raw["parser"] != PARSER or raw["study"] not in STUDIES:
            continue
        if raw["cell"] not in ENDPOINTS or raw["arm"] not in ARMS:
            continue
        identity = (
            str(raw["cell"]),
            str(raw["arm"]),
            str(raw["surface"]),
            str(raw["family"]),
        )
        if identity in seen:
            raise ValueError(f"{path}: duplicate filtered row {identity}")
        seen.add(identity)
        rows.append(dict(raw))

    expected = len(ENDPOINTS) * len(ARMS) * len(SURFACES) * 4
    if len(rows) != expected:
        raise ValueError(
            f"{path}: expected {expected} AFT/both rows for parser={PARSER}, "
            f"found {len(rows)}"
        )
    return rows


def display_rows(
    scores: Sequence[Mapping[str, Any]], surface: str, clause: str
) -> list[DisplayRow]:
    index = {
        (
            str(score["cell"]),
            str(score["arm"]),
            str(score["surface"]),
            str(score["family"]),
        ): score
        for score in scores
    }
    agreement_family = f"eval_{clause}_agreement"
    conflict_family = f"eval_{clause}_conflict"
    rows: list[DisplayRow] = []
    y = 0.0
    missing: list[tuple[str, str, str, str]] = []
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        if endpoint_index:
            y += figure0.ENDPOINT_GAP
        for arm in ARMS:
            agreement_key = (endpoint, arm, surface, agreement_family)
            conflict_key = (endpoint, arm, surface, conflict_family)
            if agreement_key not in index:
                missing.append(agreement_key)
            if conflict_key not in index:
                missing.append(conflict_key)
            if agreement_key in index and conflict_key in index:
                rows.append(DisplayRow(
                    endpoint=endpoint,
                    arm=arm,
                    agreement=index[agreement_key],
                    conflict=index[conflict_key],
                    y=y,
                ))
            y += figure0.ROW_PITCH
    if missing:
        raise ValueError(f"missing campaign-battery cells: {missing}")
    return rows


def agreement_shares(score: Mapping[str, Any]) -> dict[str, float]:
    accuracy = float(score["agreement_accuracy"])
    shares = {"shared": accuracy, "incorrect": 1.0 - accuracy}
    _check_shares(shares, "agreement", score)
    return shares


def conflict_shares(score: Mapping[str, Any]) -> dict[str, float]:
    shares = {
        key: float(score[f"{key}_rate"])
        for key in ("charter", "coin", "other", "malformed")
    }
    _check_shares(shares, "conflict", score)
    return shares


def _check_shares(
    shares: Mapping[str, float], kind: str, score: Mapping[str, Any]
) -> None:
    if any(value < -1e-9 or value > 1 + 1e-9 for value in shares.values()):
        raise ValueError(f"{kind} shares outside [0, 1]: {score}")
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise ValueError(f"{kind} shares do not sum to one: {score}")


def add_endpoint_labels(ax, rows: Sequence[DisplayRow]) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.endpoint, []).append(row.y)
    transform = ax.get_yaxis_transform()
    for endpoint, ys in grouped.items():
        y0 = min(ys) - figure0.BAR_HEIGHT / 2
        y1 = max(ys) + figure0.BAR_HEIGHT / 2
        x = -0.33
        ax.plot(
            [x, x], [y0, y1], transform=transform, clip_on=False,
            color=figure0.MUTED, linewidth=0.9,
        )
        ax.plot(
            [x, x + 0.012], [y0, y0], transform=transform, clip_on=False,
            color=figure0.MUTED, linewidth=0.9,
        )
        ax.plot(
            [x, x + 0.012], [y1, y1], transform=transform, clip_on=False,
            color=figure0.MUTED, linewidth=0.9,
        )
        ax.text(
            x - 0.012,
            (y0 + y1) / 2,
            ENDPOINT_LABEL[endpoint],
            transform=transform,
            ha="right",
            va="center",
            fontsize=7.2,
            color=figure0.MUTED,
            clip_on=False,
        )


def n_text(rows: Sequence[DisplayRow], kind: str) -> str:
    if kind == "agreement":
        ns = {
            (int(row.agreement["agreement_n"]),
             int(row.agreement["agreement_episode_n"]))
            for row in rows
        }
    else:
        ns = {
            (int(row.conflict["conflict_n"]),
             int(row.conflict["conflict_episode_n"]))
            for row in rows
        }
    if len(ns) != 1:
        raise ValueError(f"ragged {kind} sample sizes: {sorted(ns)}")
    runs, episodes = next(iter(ns))
    return f"n={runs:,} runs / {episodes:,} episodes per bar"


def render(
    scores: Sequence[Mapping[str, Any]],
    *,
    surface: str,
    clause: str,
    output: Path,
) -> list[Path]:
    rows = display_rows(scores, surface, clause)
    height = max(8.2, 2.5 + 0.31 * len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(15.4, height), sharey=True)

    for row in rows:
        figure0._draw_stack(
            axes[0], row.y, agreement_shares(row.agreement),
            AGREEMENT_ORDER, AGREEMENT_STYLE,
        )
        figure0._draw_stack(
            axes[1], row.y, conflict_shares(row.conflict),
            figure0.CONFLICT_ORDER, figure0.CONFLICT_STYLE,
        )

    figure0._decorate_panel(
        axes[0], rows,
        "Ambiguous (agreement episodes)",
        "share of agreement-eval runs (%)",
    )
    figure0._decorate_panel(
        axes[1], rows,
        "Diagnostic (conflict episodes)",
        "share of conflict-eval runs (%)",
    )
    add_endpoint_labels(axes[0], rows)

    fig.suptitle(
        "Figure 0 — Gemma-4 26B A4B · 50M presented directional tokens · "
        f"{CLAUSE_LABEL[clause]} × {SURFACE_LABEL[surface]}",
        x=0.025,
        y=0.99,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=figure0.INK,
    )
    agreement_handles = [
        Patch(
            facecolor=AGREEMENT_STYLE[key][0],
            edgecolor=AGREEMENT_STYLE[key][2],
            label=AGREEMENT_LABEL[key],
        )
        for key in AGREEMENT_ORDER
    ]
    conflict_handles = [
        Patch(
            facecolor=figure0.CONFLICT_STYLE[key][0],
            edgecolor=figure0.CONFLICT_STYLE[key][2],
            label=figure0.CONFLICT_LABEL[key],
        )
        for key in figure0.CONFLICT_ORDER
    ]
    axes[0].legend(
        handles=agreement_handles, loc="upper center", ncol=2,
        frameon=False, fontsize=8.2, bbox_to_anchor=(0.5, -0.075),
    )
    axes[1].legend(
        handles=conflict_handles, loc="upper center", ncol=4,
        frameon=False, fontsize=7.0, bbox_to_anchor=(0.48, -0.075),
    )
    fig.text(
        0.985,
        0.012,
        f"Campaign battery · parser={PARSER} · {CLAUSE_LABEL[clause]}, "
        f"{SURFACE_LABEL[surface]}; agreement {n_text(rows, 'agreement')}; "
        f"conflict {n_text(rows, 'conflict')}. Runs cluster within episodes; "
        "intervals are not overlaid on these composition bars. The blue segment "
        "is raw charter rate, not charter share among decided Charter/coin runs. "
        "Graft: 50M presented directional tokens; AFT: 8,192 rows × 2 epochs "
        "(512 updates). One seed per cell; run-to-run SD ~9pp.",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=figure0.MUTED,
        style="italic",
        wrap=True,
    )
    fig.tight_layout(rect=(0.006, 0.057, 0.994, 0.925), w_pad=2.6)

    output.mkdir(parents=True, exist_ok=True)
    stem = output / (
        f"figure0__{SURFACE_STEM[surface]}__{CLAUSE_STEM[clause]}"
        "__gemma4-26b-a4b__50m"
    )
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def write_readme(output: Path, scores: Path) -> Path:
    readme = output / "README.md"
    source = scores.resolve().relative_to(HERE.parents[3])
    readme.write_text(
        "# Gemma-4 26B A4B graft + ordinary AFT — Figure 0 gallery\n\n"
        "These six plots use the replacement campaign battery. They supersede "
        "the earlier plots made from `aft_sft_scores.*`, whose parser-validation "
        "battery had only five source dockets per episode type.\n\n"
        "## Slices\n\n"
        "The gallery contains every presentation surface (`canonical`, `trained`, "
        "`heldout`) crossed with both clause families (`trained`, `heldout`). "
        "Surfaces are not pooled because they are alternate presentations of the "
        "same episodes. Every plotted bar is present in the new battery.\n\n"
        "## Reading the bars\n\n"
        "- Rows are grouped by AFT endpoint, then charter/control/coin midtrain arm.\n"
        "- Agreement bars show shared-crew accuracy and its unsplit remainder. The "
        "replacement table does not separately identify other vs malformed for "
        "agreement families.\n"
        "- Conflict bars show the full raw response composition: Charter, another "
        "crew, malformed, and coin/cheapest. Consequently the blue width is "
        "`charter_rate`, not the headline `charter_share_decided`.\n"
        "- Counts report both runs and distinct episodes. Runs cluster within "
        "episodes, and uncertainty intervals are not overlaid on the stacks.\n\n"
        f"Source: `{source}`; filter: `parser={PARSER}`, "
        "`study in {aft, both}`. See the adjacent campaign `HEADLINE.md` for the "
        "cluster-bootstrap decided-share result and `COMPARISON.md` for the old/new "
        "battery comparison.\n\n"
        "## Regenerate\n\n"
        "From the repository root:\n\n"
        "```bash\n"
        ".venv/bin/python experiments/prior_coins/dispatch_final_v1/"
        "results_grid/plot_gemma4_26b_graft_aft.py\n"
        "```\n"
    )
    return readme


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--surface", action="append", choices=SURFACES)
    parser.add_argument("--clause", action="append", choices=CLAUSE_SPLITS)
    args = parser.parse_args()

    scores = load_scores(args.scores)
    surfaces = args.surface or list(SURFACES)
    clauses = args.clause or list(CLAUSE_SPLITS)
    written: list[Path] = []
    for surface in surfaces:
        for clause in clauses:
            written.extend(render(
                scores, surface=surface, clause=clause, output=args.out,
            ))
    written.append(write_readme(args.out, args.scores))
    for path in written:
        print(f"wrote {path}")
    print(f"\n{(len(written) - 1) // 2} figures plus README")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
