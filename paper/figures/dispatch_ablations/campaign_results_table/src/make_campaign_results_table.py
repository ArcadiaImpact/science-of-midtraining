r"""Build the campaign x EFT-treatment results table, as LaTeX.

    uv run --extra dev python .../src/make_campaign_results_table.py

One row per (base model x midtrain arm), grouped by base model; one column per
EFT treatment; one cell = the run-level split of conflict episodes into
**Charter / coin / other crew / unparseable**, in percent.

Scores come from the *public* Hub mirror ``arcadia-impact/scimt-dispatch-clean-v1``
via ``_shared/common.py``, which prefers the committed ``results_grid/scored/``
tree when this file sits in a scimt checkout and otherwise downloads.  The two
are byte-identical, so a colleague with only the paper repo -- or only this
directory -- gets exactly these numbers with no checkout, GPU or credentials.
That is the whole reason the loader is shared rather than reimplemented here.

Every run rehydrates by default and re-freezes ``src/data/`` alongside the
LaTeX, so the committed extract and the committed table can never disagree.
``--frozen`` renders from the extract without touching the network.

Two things the shape of this table makes easy to get wrong, both guarded:

*   **One draw per column.**  The $\pm$2\% columns mix two different conflict
    draws -- follow-up \#1c's balanced five-clause draw on 29 arms, and the
    original narrow single-clause draw on the nine 4B arms it never covered.
    Reading down those columns compares two interventions.  Rows on the narrow
    draw are starred from ``meta.twopct.state``, never from the profile name.
*   **One amount of training per cell.**  Every EFT cell is read at its
    converged 2-epoch endpoint (step 512), never step 256.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "_shared"))
sys.path.insert(0, str(HERE))

import common                                          # noqa: E402
import rows as registry                                # noqa: E402
from rows import Campaign, ClauseAsym, Graft           # noqa: E402

STEM = "campaign_results_table"

# ------------------------------------------------------------------ columns

#: EFT treatments, in table order: (cell name, column heading, short key).
#: ``pre_aft`` is the no-EFT anchor and is the one endpoint with no step.
TREATMENTS = (
    ("pre_aft", r"none\\\emph{(pre-EFT)}"),
    ("agreement", r"agreement\\only"),
    ("mixed_charter", r"$+$2\%\\Charter"),
    ("mixed_coin", r"$-$2\%\\coin"),
    ("charter_only", r"100\%\\Charter"),
)

#: The converged 2-epoch endpoint.  Mixing this with step 256 would put two
#: different amounts of training on one row -- MODEL_REGISTRY.md §2.
STEP = 512

#: Default slice: trained clauses x held-out template.  The campaign's
#: headline read -- the clauses the EFT trained on, on a surface it never saw.
DEFAULT_SLICE = "eval_trained_conflict__heldout"

#: The four run-level outcomes, in the order they appear in a cell.
CATEGORIES = ("charter", "coin", "other", "malformed")


def endpoint_key(cell: str) -> str:
    return cell if cell == "pre_aft" else f"{cell}-step{STEP}"


# ------------------------------------------------------------------ loading

_MEMO: dict[str, common.Scores] = {}


def _load(rel: str) -> common.Scores:
    """``load_scored`` with a memo -- the two package files are read ~20x."""
    if rel not in _MEMO:
        got = common.load_scored(rel, quiet=True)
        assert got is not None          # missing_ok is not set; absence exits
        _MEMO[rel] = got
    return _MEMO[rel]


def _split(rates: dict[str, float], n: int) -> dict:
    """A cell: the four rates as percentages, plus the n they rest on.

    ``other`` is taken by subtraction rather than read, so the four numbers
    always close at 100 and anything the scorer reported outside the four
    named categories lands where a reader would put it.
    """
    named = {c: float(rates.get(c, 0.0)) for c in CATEGORIES if c != "other"}
    named["other"] = 1.0 - sum(named.values())
    return {"pct": {c: 100.0 * named[c] for c in CATEGORIES}, "n": int(n)}


def _conflict_runs(endpoint, where: str, slice_name: str):
    """``conflict_runs`` for one slice, or None when the cell was never scored.

    An **empty** endpoint is a placeholder the collector wrote for a cell that
    has no scores -- `glm45_air_20m_legacy`'s `charter_only-step512` and the
    two GLM step-256 reads are all `{}` -- and is a fact about the campaign, so
    it renders as a dash.  An endpoint that *has* slices but not this one is a
    fact about the run, and is loud: it means the slice name is wrong, or a
    re-score dropped a read that every sibling still has.
    """
    if endpoint is None or not endpoint:
        return None
    if slice_name not in endpoint:
        raise SystemExit(
            f"{where}: no slice {slice_name!r}, but the endpoint holds "
            f"{len(endpoint)} others ({', '.join(sorted(endpoint))}).")
    return endpoint[slice_name]["conflict_runs"]


def read_campaign(src: Campaign, cell: str, slice_name: str) -> dict | None:
    scores = _load(f"{src.profile}/{src.arm}/eval.json")
    endpoint = scores.doc["result"].get(endpoint_key(cell))
    runs = _conflict_runs(endpoint, f"{scores.path}[{endpoint_key(cell)}]",
                          slice_name)
    if runs is None:
        return None
    out = _split(runs["rates"], runs["n"])
    out["twopct"] = scores.twopct_state
    out["origin"] = scores.origin
    return out


CLAUSE_ASYM_REL = "gemma27b_clause_asym_v1/comparison/all_scores.json"


def read_clause_asym(src: ClauseAsym, cell: str, slice_name: str) -> dict | None:
    scores = _load(CLAUSE_ASYM_REL)
    endpoints = scores.doc["main"][src.model]["endpoints"]
    # Only the agreement-only and 100%-Charter cells were run on these.
    endpoint = endpoints.get(endpoint_key(cell))
    runs = _conflict_runs(endpoint, f"{scores.path}[{src.model}]", slice_name)
    if runs is None:
        return None
    out = _split(runs["rates"], runs["n"])
    out["origin"] = scores.origin
    return out


GRAFT_REL = "gemma4_26b_a4b_graft/campaign_battery_scores.json"

#: The graft's rows are scored by two parsers over the same responses.  The
#: campaign's own parser is `legacy`; `rlvr` is the RLVR study's stricter one.
#: Picking `legacy` is what puts these cells on the same axis as every other
#: row in the table -- a fallback that changed *what* is measured, not just
#: how, would be exactly the trap CLAUDE.md issue #151 names.
GRAFT_PARSER = "legacy"


def read_graft(src: Graft, cell: str, slice_name: str) -> dict | None:
    scores = _load(GRAFT_REL)
    # pre_aft is stamped step 0 in the flat rows; every EFT cell is step 512.
    want_step = 0 if cell == "pre_aft" else STEP
    hits = [r for r in scores.doc
            if r["arm"] == src.arm and r["cell"] == cell
            and r["slice"] == slice_name and r["step"] == want_step
            and r["parser"] == GRAFT_PARSER and r["mode"] == "direct"]
    if not hits:
        return None
    if len(hits) > 1:
        raise SystemExit(
            f"{GRAFT_REL}: {len(hits)} rows match {src.arm}/{cell}/"
            f"{slice_name}/step{want_step}; the filter is no longer unique.")
    row, = hits
    rates = {c: row[f"{c}_rate"] for c in CATEGORIES
             if row.get(f"{c}_rate") is not None}
    if "charter" not in rates:
        return None                      # an agreement-only slice: no conflict
    out = _split(rates, row["conflict_n"])
    out["origin"] = scores.origin
    return out


READERS = {Campaign: read_campaign, ClauseAsym: read_clause_asym,
           Graft: read_graft}


def build(slice_name: str) -> dict:
    """Every cell in the table, plus what it took to get them."""
    groups = []
    for heading, rs in registry.GROUPS:
        out_rows = []
        for row in rs:
            read = READERS[type(row.source)]
            cells = {cell: read(row.source, cell, slice_name)
                     for cell, _ in TREATMENTS}
            out_rows.append({
                "dose": row.dose,
                "arm": row.arm,
                "label": row.label,
                "qualifier": row.qualifier,
                "profile": row.source.profile,
                "notes": list(row.notes),
                "cells": cells,
            })
        groups.append({"heading": heading, "rows": out_rows})

    ns = sorted({c["n"] for g in groups for r in g["rows"]
                 for c in r["cells"].values() if c})
    return {
        "stem": STEM,
        "slice": slice_name,
        "step": STEP,
        "metric": "conflict_runs -- run-level share of conflict episodes",
        "categories": list(CATEGORIES),
        "treatments": [c for c, _ in TREATMENTS],
        "conflict_run_n": ns,
        "source": {
            "hub": common.HUB_REPO,
            "local_tree": str(common.scores_root() or ""),
            "origins": sorted({c["origin"] for g in groups for r in g["rows"]
                               for c in r["cells"].values() if c}),
        },
        "groups": groups,
    }


# ------------------------------------------------------------------- LaTeX

PALETTE = {"charter": common.CHARTER, "coin": common.COIN,
           "other": common.OTHER, "malformed": common.MALFORMED}


def _texcolor_defs() -> str:
    lines = []
    for key, hexcode in PALETTE.items():
        lines.append(rf"\providecolor{{dispatch{key}}}{{HTML}}"
                     rf"{{{hexcode.lstrip('#').upper()}}}")
    return "\n".join(lines)


def fmt_cell(cell: dict | None) -> str:
    if cell is None:
        return r"\dmiss"
    pct = cell["pct"]
    return (r"\dcell" + "".join("{%.0f}" % pct[c] for c in CATEGORIES))


def render_tex(data: dict, order: str) -> str:
    r"""The table body, as an ``\input``-able fragment."""
    ncol = len(TREATMENTS)
    used_notes, marks = [], {}
    for group in data["groups"]:
        for row in group["rows"]:
            for note in row["notes"]:
                # `unrepaired` rides the star, which is stamped from
                # meta.twopct.state rather than from the registry, so it never
                # takes a number as well.
                if note == "unrepaired":
                    continue
                if note not in marks:
                    marks[note] = len(used_notes) + 1
                    used_notes.append(note)
            # A narrow-draw row is stamped in the data, not in the registry.
            states = {c.get("twopct") for c in row["cells"].values() if c}
            row["_narrow"] = "unrepaired" in states

    widest = max((f"{c['pct'][k]:.0f}"
                  for g in data["groups"] for r in g["rows"]
                  for c in r["cells"].values() if c for k in c["pct"]),
                 key=len)

    body = []
    for group in data["groups"]:
        body.append(r"\addlinespace[2.5pt]")
        body.append(rf"\multicolumn{{{ncol + 2}}}{{@{{}}l}}"
                    rf"{{\bfseries {group['heading']}}}\\")
        body.append(r"\addlinespace[1.5pt]")
        for first, rows_in_block in blocks(group["rows"], order):
            if not first:
                body.append(rf"\dsep{{{ncol + 2}}}")
            for i, row in enumerate(rows_in_block):
                # The budget is printed once per block and then left blank, so
                # the eye reads a block as one budget rather than three.
                dose = row["dose"] if (order == "arm" or i == 0) else ""
                sup = "".join(rf"\textsuperscript{{{marks[n]}}}"
                              for n in row["notes"] if n in marks)
                star = r"$^{\ast}$" if row["_narrow"] else ""
                cells = " & ".join(fmt_cell(row["cells"][c])
                                   for c, _ in TREATMENTS)
                body.append(rf"\quad {dose} & {fmt_label(row)}{sup}{star} & "
                            rf"{cells}\\")

    heads = " & ".join(rf"\thead{{{h}}}" for _, h in TREATMENTS)
    ns = data["conflict_run_n"]
    n_text = (f"{ns[0]:,}" if len(ns) == 1
              else f"{min(ns):,}--{max(ns):,}")

    notes = [rf"\textsuperscript{{{marks[n]}}} {registry.NOTES[n]}"
             for n in used_notes]
    if any(r["_narrow"] for g in data["groups"] for r in g["rows"]):
        notes.insert(0, r"$^{\ast}$ " + registry.NOTES["unrepaired"])

    return TEX_TEMPLATE.format(
        colors=_texcolor_defs(),
        widest=widest,
        colspec="r" + "l" + "c" * ncol,
        ncol_eft=ncol,
        ncol_last=ncol + 2,
        heads=heads,
        body="\n".join(body),
        slice_name=data["slice"].replace("_", r"\_"),
        step=data["step"],
        n=n_text,
        hub=data["source"]["hub"].replace("_", r"\_"),
        notes="\n\n".join(notes),
    )


def blocks(rows: list[dict], order: str):
    """Group a base model's rows into the runs a thin rule separates.

    In budget order a block is one *presented-token budget*, and the arms
    inside it are put in ``ARM_ORDER`` -- Charter, control, coin -- here rather
    than in the extract, so row order stays a presentation choice.  The
    qualifier is part of the key: the ablation rows share a budget with a
    standard trio (50M no-worked-examples, 190M clause-asymmetric) and get
    their own block rather than extending the trio above them.

    In arm order a block is one arm, and the budgets inside it keep the
    registry's own ascending order.

    Yields ``(is_first_block, rows)``.
    """
    key = ((lambda r: r["arm"]) if order == "arm"
           else (lambda r: (r["dose"], r["qualifier"])))
    grouped: dict = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    keys = (sorted(grouped, key=registry.ARM_ORDER.index) if order == "arm"
            else list(grouped))                 # first-seen = ascending budget
    for i, k in enumerate(keys):
        members = grouped[k]
        if order != "arm":
            members = sorted(members,
                             key=lambda r: registry.ARM_ORDER.index(r["arm"]))
        yield i == 0, members


#: Arms whose label is tinted in the corpus column: the two *directional*
#: midtraining corpora.  Control is filler -- zero directional tokens -- so it
#: stays plain, which is the distinction the tint is there to draw.
CHIP = {"charter": ("dispatchcharter", "Charter"),
        "coin": ("dispatchcoin", "coin")}


def fmt_label(row: dict) -> str:
    r"""The corpus cell: a tinted chip for the arm, plain text for a qualifier."""
    chip = CHIP.get(row["arm"])
    if chip is None:
        return row["label"]
    colour, word = chip
    tinted = rf"\dchip{{{colour}}}{{{word}}}\dchipend"
    return f"{tinted}, {row['qualifier']}" if row["qualifier"] else tinted


TEX_TEMPLATE = r"""% Generated by src/make_campaign_results_table.py -- do not edit by hand.
% Numbers are frozen in src/data/campaign_results_table.json.
% Needs booktabs and xcolor. colortbl (xcolor's [table] option) is optional --
% without it the budget separators come out black rather than grey.
{colors}
\ifdefined\arrayrulecolor\else\providecommand{{\arrayrulecolor}}[1]{{}}\fi
% Every number sits in a box as wide as the widest number in the table, so
% they line up down the column as well as across the cell.
\ifdefined\dnumw\else\newlength{{\dnumw}}\fi
\settowidth{{\dnumw}}{{{widest}}}
\providecommand{{\dnum}}[2]{{\makebox[\dnumw][r]{{\textcolor{{#1}}{{#2}}}}}}
% One cell: Charter / coin / other crew / unparseable, in percent of runs.
\providecommand{{\dcell}}[4]{{%
  \dnum{{dispatchcharter}}{{\bfseries #1}}\textcolor{{black!30}}{{/}}%
  \dnum{{dispatchcoin}}{{#2}}\textcolor{{black!30}}{{/}}%
  \dnum{{dispatchother}}{{#3}}\textcolor{{black!30}}{{/}}%
  \dnum{{dispatchmalformed}}{{#4}}}}
\providecommand{{\dmiss}}{{\textcolor{{black!35}}{{--}}}}
% A tinted chip for the two directional corpora.
\providecommand{{\dchipsep}}{{1.7pt}}
\providecommand{{\dchip}}[2]{{{{\setlength{{\fboxsep}}{{\dchipsep}}%
  \colorbox{{#1!14}}{{\strut #2}}}}}}
% Cancels the chip's right padding, so ", no worked ex." and footnote marks
% sit against the word rather than a box edge.
\providecommand{{\dchipend}}{{\kern-\dchipsep}}
% Thin grey rule between token budgets.
\providecommand{{\dsep}}[1]{{\noalign{{\vskip1.5pt}}%
  \arrayrulecolor{{black!30}}\cmidrule[0.4pt]{{1-#1}}%
  \arrayrulecolor{{black}}\noalign{{\vskip0.5pt}}}}
\providecommand{{\thead}}[1]{{\begin{{tabular}}[b]{{@{{}}c@{{}}}}#1\end{{tabular}}}}

\begin{{tabular}}{{{colspec}}}
\toprule
& & \multicolumn{{{ncol_eft}}}{{c}}{{\textbf{{EFT treatment}}}}\\
\cmidrule(l){{3-{ncol_last}}}
\thead{{presented\\tokens}} & \thead{{midtraining\\corpus}} & {heads}\\
\midrule
{body}
\bottomrule
\end{{tabular}}

\vspace{{2pt}}
{{\footnotesize
Each cell: \textcolor{{dispatchcharter}}{{\bfseries Charter}} /
\textcolor{{dispatchcoin}}{{coin}} /
\textcolor{{dispatchother}}{{other crew}} /
\textcolor{{dispatchmalformed}}{{unparseable}}, as a percentage of conflict
runs; the four close at 100. Slice \texttt{{{slice_name}}} -- the five clauses
the EFT trains on, on a template surface neither midtraining nor EFT ever
showed. Every EFT cell is its converged 2-epoch endpoint (step {step});
$n = {n}$ conflict runs per cell. One seed per cell throughout: measured
run-to-run SD on the Charter share is $\approx 9$\,pp, so a difference smaller
than that is not a difference. Scores: \texttt{{{hub}}}.

{notes}
}}
"""


STANDALONE = r"""% Local preview only -- the paper \input{}s campaign_results_table.tex.
\documentclass[10pt,varwidth=215mm,border=5mm]{standalone}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{amsmath}
\begin{document}
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.08}
\input{campaign_results_table.tex}
\end{document}
"""


# ----------------------------------------------------------------- preview

def compile_preview(outdir: Path, stem: str = STEM) -> None:
    r"""pdflatex the standalone wrapper, then rasterise a PNG beside it.

    The paper \input{}s the fragment; this is only so the table can be looked
    at without a paper build.  A missing TeX is a warning, not a failure --
    the .tex and the frozen extract are the deliverables.
    """
    import shutil
    import subprocess

    if shutil.which("pdflatex") is None:
        print("  pdflatex not found; skipping the preview "
              "(the .tex is written)")
        return
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         f"{stem}_standalone.tex"],
        cwd=outdir, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        raise SystemExit(f"pdflatex failed:\n{tail}")
    pdf = outdir / f"{stem}_standalone.pdf"
    pdf.replace(outdir / f"{stem}.pdf")
    for junk in ("aux", "log"):
        (outdir / f"{stem}_standalone.{junk}").unlink(missing_ok=True)
    print(f"  wrote {outdir / (stem + '.pdf')}")

    if shutil.which("pdftoppm") is None:
        print("  pdftoppm not found; skipping the PNG preview")
        return
    subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile",
                    f"{stem}.pdf", stem],
                   cwd=outdir, check=True)
    print(f"  wrote {outdir / (stem + '.png')}")


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slice", default=DEFAULT_SLICE,
                    help="scored slice to read (default: %(default)s)")
    ap.add_argument("--order", choices=("dose", "arm"), default="dose",
                    help="row order within a base-model group")
    ap.add_argument("--frozen", action="store_true",
                    help="render from src/data/ without touching the network")
    ap.add_argument("--outdir", type=Path, default=HERE.parent)
    ap.add_argument("--no-compile", action="store_true",
                    help="write the .tex only; skip pdflatex and the preview")
    args = ap.parse_args()

    # A non-default slice gets its own extract and its own .tex, so reading a
    # second slice never silently overwrites the committed table's numbers.
    suffix = "" if args.slice == DEFAULT_SLICE else f"__{args.slice}"
    stem = f"{STEM}{suffix}"
    extract = HERE / "data" / f"{stem}.json"
    if args.frozen:
        data = json.loads(extract.read_text())
        if data["slice"] != args.slice:
            raise SystemExit(
                f"--frozen holds slice {data['slice']!r}, not {args.slice!r}. "
                f"Re-run without --frozen to rehydrate.")
    else:
        print(f"Rehydrating from {common.HUB_REPO} "
              f"(local tree: {common.scores_root() or 'none'})")
        data = build(args.slice)
        extract.parent.mkdir(parents=True, exist_ok=True)
        extract.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        print(f"  froze {len(data['groups'])} groups -> "
              f"{extract.relative_to(HERE.parent)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    tex = args.outdir / f"{stem}.tex"
    tex.write_text(render_tex(data, args.order))
    (args.outdir / f"{stem}_standalone.tex").write_text(
        STANDALONE.replace(f"{STEM}.tex", f"{stem}.tex"))
    print(f"  wrote {tex}")

    if not args.no_compile:
        compile_preview(args.outdir, stem)

    filled = sum(1 for g in data["groups"] for r in g["rows"]
                 for c in r["cells"].values() if c)
    total = sum(len(TREATMENTS) for g in data["groups"] for _ in g["rows"])
    print(f"  {filled}/{total} cells populated "
          f"({total - filled} never run, rendered as --)")


if __name__ == "__main__":
    main()
