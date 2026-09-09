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
    written = []
    for fmt in formats:
        dest = outdir / f"{stem}.{fmt}"
        fig.savefig(dest, format=fmt)
        written.append(dest)
    plt.close(fig)
    return written


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


def load_ablation(name: str, quiet: bool = False) -> Scores:
    """Load ``scored/ablations/<name>.json``, local or Hub."""
    root = scores_root()
    if root is not None:
        local = root / "ablations" / f"{name}.json"
        if local.is_file():
            return Scores(json.loads(local.read_text()), "local", str(local))
    return _from_hub(f"scores/ablations/{name}.json", quiet=quiet)


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


def wilson(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson half-widths (low, high) about ``rate``.

    Optimistic here: eval runs are 3 per episode and not independent, so this
    understates the real interval.  Off by default in the figure scripts.
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
