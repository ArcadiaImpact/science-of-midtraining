r"""Shared plumbing for the paper's headline figures.

Two conventions this module exists to enforce, because they apply to *every*
figure in the paper and are easy to get subtly wrong per-script:

1.  **Real page width.**  ICLR 2027's ``\textwidth`` is ``5.5 true in``
    (``iclr2027_conference.sty:49``).  Every figure is authored at that width,
    so a 9pt label in the figure is a 9pt label on the page and text sizes can
    be sanity-checked by eye against the body copy.  Never scale a figure in
    ``\includegraphics`` -- change ``--width-frac`` here instead.

2.  **Local-first, Hub-rehydrating score loading.**  Scores are read from the
    committed ``results_grid/scored/`` tree when this file sits inside the
    scimt checkout, and otherwise downloaded from the *public* Hub mirror
    ``arcadia-impact/scimt-dispatch-clean-v1``.  The two are byte-identical --
    ``build_clean_repo.py`` copies the git tree verbatim -- so a colleague who
    has only the paper repo re-renders exactly the same numbers.

Both halves are deliberately dependency-light (matplotlib + huggingface_hub)
so this directory can be copied into the paper repo as-is.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------- page

#: ICLR 2027 text width, in inches.  See iclr2027_conference.sty:49.
TEXTWIDTH_IN = 5.5

# --------------------------------------------------------------------- ink

# Single source of truth is the paper's own coincharter.sty.  NOTE: main.tex
# currently re-\definecolor's charter/coin to the seaborn-colorblind pair
# (#0173B2 / #DE8F05) *after* loading the package, so the compiled PDF uses
# those.  The difference is imperceptible (<=16/255 per channel); if the
# override is removed from main.tex these become exact.
CHARTER = "#0072B2"   # Okabe-Ito blue
COIN = "#E69F00"      # Okabe-Ito orange
OTHER = "#999999"     # neutral grey

#: Bottom-to-top stacking order, matching Tikz_Figs/results_preview.tex.
STACK = (("charter", CHARTER, "white"),
         ("other", OTHER, "black"),
         ("coin", COIN, "white"))

STACK_LABEL = {"charter": "Chose Charter option",
               "other": "Other outcome",
               "coin": "Chose Coin option"}


def setup(fontsize: float = 9.0) -> None:
    """rcParams tuned for a 5.5in-wide figure dropped into a 10pt Times paper.

    Sizes are absolute points, so they land on the page at the value set here.
    """
    plt.rcParams.update({
        "font.family": "serif",
        # Times if the machine has it; STIXGeneral is Times-metric-compatible
        # and ships with matplotlib, so output is stable across machines.
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize - 0.5,
        "ytick.labelsize": fontsize - 0.5,
        "legend.fontsize": fontsize - 0.5,
        "figure.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Keep SVG text as text: the point of shipping SVG is that layout can
        # be nudged downstream without re-running the script.
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figure(height_in: float, width_frac: float = 1.0):
    """A figure exactly ``width_frac`` of the ICLR text column."""
    return plt.subplots(figsize=(TEXTWIDTH_IN * width_frac, height_in))


def margins(fig, left: float, right: float, top: float, bottom: float) -> None:
    """Set axes margins in **inches**, not fractions.

    Fractions move when the figure height changes; inches do not, so a script
    can be re-run at a different height without the labels re-colliding.  This
    is also why nothing here uses ``bbox_inches="tight"`` -- see ``save``.
    """
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=left / w, right=1 - right / w,
                        top=1 - top / h, bottom=bottom / h)


def save(fig, stem: str, outdir: Path, formats: Sequence[str] = ("svg", "pdf"),
         verify: bool = True) -> list[Path]:
    r"""Write the figure at its **exact** authored size.

    Deliberately no ``bbox_inches="tight"``: tight-cropping shrinks the canvas
    to the ink, so a figure authored at 5.5in lands on disk at whatever the
    content happened to span (5.24in, in this figure's first draft).
    ``\includegraphics[width=\linewidth]`` then scales it back up and every
    font size in it drifts by that ratio -- exactly the 1:1 sanity-check the
    5.5in convention exists to protect.  Lay out with ``margins()`` instead.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if verify:
        w, h = fig.get_size_inches()
        print(f"  canvas {w:.3f} x {h:.3f} in "
              f"({w / TEXTWIDTH_IN:.3f} x ICLR textwidth) -- include at "
              f"width={w / TEXTWIDTH_IN:.3f}\\linewidth for 1:1 text")
        for warning in overflowing(fig):
            print(f"  WARNING: {warning}")
    written = []
    for fmt in formats:
        dest = outdir / f"{stem}.{fmt}"
        fig.savefig(dest, format=fmt)
        written.append(dest)
    plt.close(fig)
    return written


def overflowing(fig, slack_pt: float = 1.0) -> list[str]:
    """Figure-level text that runs off the canvas.

    Authoring at a fixed page width means nothing is tight-cropped and nothing
    auto-shrinks, so an over-long footnote silently loses its ends instead of
    resizing the figure.  Cheap to check, and invisible until someone reads
    the compiled PDF, so ``save`` checks every time.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_px = fig.get_size_inches()[0] * fig.dpi
    bad = []
    for text in fig.texts:
        box = text.get_window_extent(renderer)
        if box.x0 < -slack_pt or box.x1 > width_px + slack_pt:
            over = max(-box.x0, box.x1 - width_px) / fig.dpi
            body = text.get_text().splitlines()[0][:60]
            bad.append(f"text runs {over:.2f}in off the canvas: {body!r}...")
    return bad


# ------------------------------------------------------------------- scores

#: Public Hub mirror of the committed score tree.  Anonymous read works.
HUB_REPO = "arcadia-impact/scimt-dispatch-clean-v1"

#: Where the committed tree lives inside the scimt checkout, relative to this
#: file.  Absent when this directory has been copied into the paper repo.
_LOCAL_SCORED = HERE.parent / "results_grid" / "scored"


def scores_root() -> Path | None:
    """The local ``scored/`` tree, or None if we are not in the checkout.

    ``SCIMT_SCORES`` overrides, for pointing at a worktree or a rescore.
    """
    override = os.environ.get("SCIMT_SCORES")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"SCIMT_SCORES={root} is not a directory")
        return root
    return _LOCAL_SCORED if _LOCAL_SCORED.is_dir() else None


@dataclass(frozen=True)
class Scores:
    """One scored JSON plus where it actually came from."""

    doc: dict[str, Any]
    origin: str          # "local" | "hub"
    path: str            # display path / repo path

    @property
    def twopct_state(self) -> str | None:
        """substituted | unrepaired | already_balanced -- see MODEL_REGISTRY."""
        return (self.doc.get("meta", {}).get("twopct") or {}).get("state")


def load_scores(profile: str, arm: str, battery: str = "eval",
                tree: str | None = None, quiet: bool = False) -> Scores:
    """Load ``scored/[<tree>/]<profile>/<arm>/<battery>.json``, local or Hub.

    ``tree`` selects a non-canonical subtree -- in practice only
    ``"legacy_narrow_2pct"``, the archived pre-#1c draw.  That subtree is
    **git-only**: ``build_clean_repo.py`` skips it, so there is no Hub
    fallback for it and asking for it off-checkout is a loud error.
    """
    rel = Path(*( [tree] if tree else [] ), profile, arm, f"{battery}.json")

    root = scores_root()
    if root is not None:
        local = root / rel
        if local.is_file():
            return Scores(json.loads(local.read_text()), "local", str(local))

    if tree:
        raise SystemExit(
            f"{rel} is not in the local score tree, and the '{tree}' subtree is "
            f"not mirrored to the Hub (build_clean_repo.py excludes it). "
            f"Run from a scimt checkout, or set SCIMT_SCORES.")

    return _from_hub(f"scores/{rel.as_posix()}", quiet=quiet)


#: Memo for load_ablation, misses included.  A caller that asks per-arm would
#: otherwise re-round-trip the mirror once per arm, and a miss costs a 404
#: every time.
_ABLATION_MEMO: dict[str, Scores | None] = {}


def load_ablation(name: str, missing_ok: bool = False,
                  quiet: bool = False) -> Scores | None:
    """Load ``scored/ablations/<name>.json``, local or Hub.

    ``missing_ok`` returns None instead of exiting when the artifact is in
    neither place -- for a collection new enough that the public mirror has
    not been rebuilt since, where the caller has a raw-Hub fallback.
    """
    if name in _ABLATION_MEMO:
        hit = _ABLATION_MEMO[name]
        if hit is None and not missing_ok:
            raise SystemExit(f"ablation {name!r} is in neither the local tree "
                             f"nor {HUB_REPO}")
        return hit

    root = scores_root()
    if root is not None:
        local = root / "ablations" / f"{name}.json"
        if local.is_file():
            found = Scores(json.loads(local.read_text()), "local", str(local))
            _ABLATION_MEMO[name] = found
            return found
    try:
        found = _from_hub(f"scores/ablations/{name}.json", quiet=quiet)
    except SystemExit:
        _ABLATION_MEMO[name] = None
        if missing_ok:
            return None
        raise
    _ABLATION_MEMO[name] = found
    return found


def subdocument(pack: Scores, key: str) -> Scores:
    """One arm/profile out of a collected ablation's ``documents`` map."""
    docs = pack.doc.get("documents", {})
    if key not in docs:
        raise SystemExit(f"{pack.path}: no document {key!r}. Have: "
                         f"{', '.join(sorted(docs))}")
    return Scores(docs[key], pack.origin, f"{pack.path}#{key}")


#: Campaign follow-ups whose scores were never collected into ``scored/``.
#: Public, so the rehydrate path works for anyone.
GLM_FOLLOWUP_REPO = "arcadia-impact/scimt-dispatch-final-v1-glm"

#: Where cached raw follow-up scores are committed, so a figure that depends
#: on one still renders offline and its numbers live in git.
DATA = HERE / "data"


def load_hub_json(repo: str, repo_path: str, cache_name: str,
                  refresh: bool = False, quiet: bool = False) -> Scores:
    r"""A scored JSON that lives only on the Hub, cached into ``data/``.

    Some follow-up cells were published to the Hub but never collected into
    ``results_grid/scored/`` -- ``balanced_80_10_10`` is the case in hand.
    Rather than hand-editing a collected artifact, a figure that needs one
    fetches it once and commits the result under ``data/<cache_name>.json``
    with its provenance, so the figure is reproducible offline and the numbers
    it plots are in git like every other number in the paper.

    If a cell like this ever becomes load-bearing beyond one figure, it should
    graduate into ``collect_ablation_scores.py`` instead of living here.
    """
    cache = DATA / f"{cache_name}.json"
    if cache.is_file() and not refresh:
        payload = json.loads(cache.read_text())
        src = payload.get("source", {})
        return Scores(payload["scores"], "cache",
                      f"data/{cache.name} <- {src.get('repo')}@"
                      f"{str(src.get('revision'))[:8]}:{src.get('path')}")

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:  # pragma: no cover - depends on the caller's env
        raise SystemExit(
            f"data/{cache.name} is absent and huggingface_hub is not "
            f"installed.\n  pip install huggingface_hub    # the repo is public")
    if not quiet:
        print(f"  fetching {repo_path}\n    from {repo}")
    try:
        revision = HfApi().repo_info(repo, repo_type="model").sha
        path = hf_hub_download(repo, repo_path, repo_type="model",
                               revision=revision)
    except Exception as exc:
        raise SystemExit(f"Could not fetch {repo_path} from {repo}: {exc}")

    doc = json.loads(Path(path).read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {"source": {"repo": repo, "path": repo_path, "revision": revision,
                    "fetched": _utcnow(),
                    "note": "Cached by paper_figures/common.load_hub_json. "
                            "Verbatim copy of the Hub file; edit nothing here, "
                            "re-fetch with --refresh."},
         "scores": doc}, indent=1, sort_keys=True) + "\n")
    if not quiet:
        print(f"    cached -> data/{cache.name}")
    return Scores(doc, "hub", f"{repo}@{revision[:8]}:{repo_path}")


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _from_hub(repo_path: str, quiet: bool = False) -> Scores:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover - depends on the caller's env
        raise SystemExit(
            "No local score tree and huggingface_hub is not installed.\n"
            "  pip install huggingface_hub    # then re-run; the repo is public")
    if not quiet:
        print(f"  rehydrating {repo_path} from {HUB_REPO}")
    try:
        path = hf_hub_download(HUB_REPO, repo_path, repo_type="model")
    except Exception as exc:
        raise SystemExit(f"Could not fetch {repo_path} from {HUB_REPO}: {exc}")
    return Scores(json.loads(Path(path).read_text()), "hub",
                  f"{HUB_REPO}:{repo_path}")


# ------------------------------------------------------------------- slices

def cell(scores: Scores, endpoint: str, slice_name: str) -> dict[str, Any]:
    """One (endpoint, slice) cell, with a diagnosable error when it is absent.

    A missing endpoint is nearly always meaningful -- a cell that has not
    landed, or a family that does not evaluate at that step -- so this refuses
    to paper over it.
    """
    if "result" not in scores.doc and "slices" in scores.doc:
        # Raw Hub scores.json: one endpoint per file, slices at the top level.
        slices = scores.doc["slices"]
        if slice_name not in slices:
            raise SystemExit(
                f"{scores.path}: no slice {slice_name!r}. Have: "
                f"{', '.join(sorted(slices))}")
        return slices[slice_name]
    result = scores.doc.get("result", {})
    if endpoint not in result:
        raise SystemExit(
            f"{scores.path}: no endpoint {endpoint!r}. Have: "
            f"{', '.join(sorted(result))}")
    slices = result[endpoint]
    if slice_name not in slices:
        raise SystemExit(
            f"{scores.path}[{endpoint}]: no slice {slice_name!r}. Have: "
            f"{', '.join(sorted(slices))}")
    return slices[slice_name]


def motivation_split(cell_doc: dict[str, Any]) -> tuple[dict[str, float], int]:
    """Run-level (charter, other, coin) fractions summing to 1, plus n.

    ``conflict_runs.rates`` splits four ways -- charter / coin / malformed /
    other -- and the paper's stacked bar shows three.  Everything that is not
    a charter or coin choice is folded into ``other`` by subtraction, so the
    bar is guaranteed to close at 100% rather than drifting on rounding.
    """
    runs = cell_doc["conflict_runs"]
    rates = runs["rates"]
    charter = float(rates.get("charter", 0.0))
    coin = float(rates.get("coin", 0.0))
    return ({"charter": charter, "other": 1.0 - charter - coin, "coin": coin},
            int(runs["n"]))


#: by_mixture keys on a conflict slice: one and two conflict runs per episode.
ONE_RUN, TWO_RUN = "c", "c/c"
RUN_SHAPE_LABEL = {
    ONE_RUN: "One conflict run per episode",
    TWO_RUN: "Two conflict runs per episode",
}


def split_by_run_count(cell_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    r"""Run-level charter/other/coin split, separately for 1- and 2-run episodes.

    The scored files carry run-level counts only pooled (``conflict_runs``) and
    per-clause (``conflict_runs_by_clause``) -- never per episode shape.  What
    they do carry is ``by_mixture``, which is EPISODE-level.  The run-level
    split is nonetheless exactly recoverable, because of how ``episode_label``
    is defined in ``score_factorised``:

    * ``impure`` outranks ``mixed`` -- any OTHER run makes the whole episode
      impure -- so ``mixed`` on a two-run episode is exactly one charter run
      and one coin run, never charter+other;
    * ``all_charter`` / ``all_coin`` are 2 runs of that side;
    * ``malformed`` is 2 malformed runs;
    * a ONE-run ``impure`` episode is exactly one OTHER run, which pins the
      only unknown.

    That last point is what closes the system: the one-run side is fully
    determined by its episode labels, so whatever the pooled totals hold in
    excess of it belongs to the two-run ``impure`` episodes, whose internal
    split is otherwise unknowable.  The function solves for it and asserts the
    reconstruction closes, so a scoring change that breaks these invariants
    fails loudly instead of quietly mis-attributing runs.
    """
    mixture = cell_doc["by_mixture"]
    unexpected = set(mixture) - {ONE_RUN, TWO_RUN}
    if unexpected:
        raise SystemExit(
            f"split_by_run_count: conflict slice carries unexpected episode "
            f"shapes {sorted(unexpected)}; only {ONE_RUN!r}/{TWO_RUN!r} can be "
            f"reconstructed")

    one = mixture.get(ONE_RUN, {})
    two = mixture.get(TWO_RUN, {})
    runs = cell_doc["conflict_runs"]
    total_n = runs["n"]
    total = {k: round(v * total_n) for k, v in runs["rates"].items()}

    # One-run episodes: label == the single run's verdict.
    one_runs = {
        "charter": one.get("all_charter", 0),
        "coin": one.get("all_coin", 0),
        "other": one.get("impure", 0),
        "malformed": one.get("malformed", 0),
    }
    # Two-run episodes: everything but `impure` is pinned by its label.
    two_known = {
        "charter": 2 * two.get("all_charter", 0) + two.get("mixed", 0),
        "coin": 2 * two.get("all_coin", 0) + two.get("mixed", 0),
        "other": 0,
        "malformed": 2 * two.get("malformed", 0),
    }
    two_runs = {k: total.get(k, 0) - one_runs[k] for k in one_runs}
    impure_runs = 2 * two.get("impure", 0)
    residual = {k: two_runs[k] - two_known[k] for k in two_runs}
    if sum(residual.values()) != impure_runs or any(v < 0 for v in
                                                    residual.values()):
        raise SystemExit(
            f"split_by_run_count: reconstruction did not close -- residual "
            f"{residual} against {impure_runs} runs in two-run impure "
            f"episodes. The scorer's episode_label invariants may have changed.")

    out = {}
    for key, counts, episodes in ((ONE_RUN, one_runs, one),
                                  (TWO_RUN, two_runs, two)):
        n = sum(counts.values())
        charter = counts["charter"] / n if n else 0.0
        coin = counts["coin"] / n if n else 0.0
        out[key] = {
            "split": {"charter": charter, "other": 1.0 - charter - coin,
                      "coin": coin},
            "n": n,
            "n_episodes": sum(episodes.values()),
            "counts": counts,
        }
    return out


def stack_bars(ax, xs: Sequence[float], splits: Sequence[dict[str, float]],
               bar_w: float, fontsize: float, min_inline: float = 5.0,
               label_series: bool = True) -> None:
    """One stacked charter/other/coin panel, as percentages of runs."""
    bottoms = [0.0] * len(splits)
    for key, colour, ink in STACK:
        vals = [s[key] * 100.0 for s in splits]
        ax.bar(xs, vals, bar_w, bottom=bottoms, color=colour, linewidth=0,
               label=STACK_LABEL[key] if label_series else None, zorder=2)
        for x, val, base in zip(xs, vals, bottoms):
            if val >= min_inline:
                ax.text(x, base + val / 2, f"{val:.0f}", ha="center",
                        va="center", color=ink, fontsize=fontsize - 0.5,
                        zorder=3)
        bottoms = [b + v for b, v in zip(bottoms, vals)]


def split_axes(height_in: float, width_frac: float = 1.0):
    """Two stacked panels sharing an x axis, for the by-run-count views."""
    return plt.subplots(2, 1, figsize=(TEXTWIDTH_IN * width_frac, height_in),
                        sharex=True)


#: Where --split-by-run writes.  Diagnostic, not paper output; gitignored.
SCRATCH = HERE / "scratch"


def draw_split_by_run(rows, xs, bar_w, args, tick_labels, group_annotate,
                      ylabel: str):
    r"""The two-panel by-run-count diagnostic shared by every figure here.

    Same bars as the figure it belongs to, but each panel restricted to one
    episode shape and still counting RUNS, so the two panels are like-for-like
    with each other and with the pooled figure.  Note the pooled figure is not
    their average: two-run episodes are half the episodes but two thirds of
    the runs.

    ``group_annotate(ax)`` draws the caller's group labels on the lower panel;
    ``tick_labels`` are the per-bar labels.
    """
    setup(args.fontsize)
    fig, (top, bottom) = split_axes(args.height * 1.75, args.width_frac)

    for ax, shape in ((top, ONE_RUN), (bottom, TWO_RUN)):
        splits = [r["by_run"][shape]["split"] for r in rows]
        stack_bars(ax, xs, splits, bar_w, args.fontsize,
                   label_series=(ax is top))
        n_runs = rows[0]["by_run"][shape]["n"]
        n_eps = rows[0]["by_run"][shape]["n_episodes"]
        ax.set_title(f"{RUN_SHAPE_LABEL[shape]} "
                     f"({n_eps:,} episodes, {n_runs:,} runs per bar)",
                     fontsize=args.fontsize - 0.5, pad=3, loc="left")
        ax.set_xlim(xs[0] - 0.9, xs[-1] + 0.9)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel(ylabel)

    bottom.set_xticks(xs)
    bottom.set_xticklabels(tick_labels)
    bottom.tick_params(axis="x", length=0, pad=3)
    group_annotate(bottom)

    # Top has to carry the legend AND the upper panel's own title.
    margins(fig, left=0.52, right=0.06, top=0.52, bottom=0.62)
    fig.subplots_adjust(hspace=0.34)
    top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=3,
               frameon=False, handlelength=1.1, handleheight=0.9,
               columnspacing=1.4, borderpad=0.0, handletextpad=0.5)
    return fig


def report_split_by_run(rows, name_of) -> None:
    """Per-shape run-level rates plus the pooled rate they compose into."""
    print(f"\n  by episode shape (RUN-level, so like-for-like with the figure)")
    print(f"  {'bar':30s} {'1-run':>8s} {'2-run':>8s} {'delta':>8s} "
          f"{'pooled':>8s}")
    for r in rows:
        one = r["by_run"][ONE_RUN]["split"]["charter"] * 100
        two = r["by_run"][TWO_RUN]["split"]["charter"] * 100
        print(f"  {name_of(r):30s} {one:7.1f}% {two:7.1f}% {two - one:+7.1f}pp "
              f"{r['split']['charter'] * 100:7.1f}%")
    print("  (pooled is not the mean of the two: two-run episodes are half the "
          "episodes\n   but two thirds of the runs)")


def wilson(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson half-widths (low, high) about ``rate``.

    Optimistic here, and the reason is structural: **the episode is the
    sampling unit, not the run**.  ``score_factorised`` reads one saved
    response per ``episode_id`` and derives a verdict per run from that single
    generation, so a two-conflict-run episode yields two runs from one sample.
    A conflict slice is 1,000 one-run + 1,000 two-run episodes = 3,000 runs
    from 2,000 generations.

    Measured intra-episode correlation on the plotted cells is rho 0.04-0.23
    (one outlier at 0.60), a design effect of 1.03-1.40, so the honest n is
    ~2,100-2,900 rather than 3,000 and the interval widens by 0.1-0.4pp.  That
    is small -- and both are dwarfed by the ~9pp run-to-run SD from the single
    seed, which no interval computed from one run can see at all.  Off by
    default in the figure scripts for that reason.
    """
    if n <= 0:
        return (0.0, 0.0)
    denom = 1.0 + z * z / n
    centre = (rate + z * z / (2 * n)) / denom
    half = z * ((rate * (1 - rate) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, rate - max(0.0, centre - half)),
            max(0.0, min(1.0, centre + half) - rate))


def provenance(items: Iterable[Scores]) -> str:
    """One-line summary of where the plotted numbers came from."""
    origins = sorted({s.origin for s in items})
    states = sorted({s.twopct_state for s in items if s.twopct_state})
    bits = ["scores: " + "+".join(origins)]
    if states:
        bits.append("2% draw: " + "/".join(states))
    return "; ".join(bits)
