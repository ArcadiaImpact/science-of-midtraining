"""Format-only comparison: paired campaign/diverse-response stacks per midtrain.

Eight full-width PDFs: four EFT conditions x trained/held-out clauses, step 512,
held-out prompt surface. Counts are frozen locally; --refresh rebuilds the
extract from pinned sources. The 2% campaign references are the matched as-run
draw, not the repaired 2% data used by the current main campaign figures.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DISPATCH = HERE
DATA = HERE / "source_data" / "diverse_response_format.json"
OUTPUT = HERE / "figures" / "diverse_response_format"
ROOT = DISPATCH.parents[2]
sys.path.insert(0, str(DISPATCH))

import matplotlib
from scimt.viz import paper as ps
import clause_plot
import common

REPO = "arcadia-impact/scimt-dispatch-clean-v1"
REVISION = "60066c916a6989a02033cf827d94c3cc46ddfa02"
DIVERSE_PATH = "scores/ablations/diverse_response.json"
# The as-run campaign score tree is retained in git; it is not in the clean
# mirror. Resolve this immutable commit without switching branches.
BASE_REVISION = "7c4ffd0c74b32ca7fa70e4a645c619f267e7422d"
BASE_PATH = ("experiments/prior_coins/dispatch_final_v1/results_grid/scored/"
             "legacy_narrow_2pct/gemma3_12b_50m_4ep/{arm}/eval.json")
ARMS = {"control": "Control midtrain", "charter": "Charter midtrain", "coin": "Coin midtrain"}
EFTS = {"agreement": "Ambiguous EFT", "mixed_coin": "2% Coin EFT (as-run)",
        "mixed_charter": "2% Charter EFT (as-run)", "charter_only": "100% Charter EFT"}
SLICES = {"trained": "eval_trained_conflict__heldout", "holdout": "eval_holdout_conflict__heldout"}
CLAUSE_LABELS = {"trained": "Trained clauses", "holdout": "Held-out clauses"}
CLAUSES = {"trained": {"precedence_days_since", "precedence_registry_rank", "precedence_runs_year",
                        "qual_skill", "qual_specialty"},
           "holdout": {"precedence_deferrals", "qual_weekly_limit"}}
VARIANTS = {"main": "Main\nstudy", "diverse": "Diverse\nresponses"}
STACK = (("charter", ps.CHARTER, "white"), ("other", ps.GREY, ps.INK),
         ("malformed", ps.INK, "white"), ("coin", ps.COIN, "white"))
LABELS = {"charter": "Charter", "other": "Other crew", "malformed": "Unparseable", "coin": "Coin"}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def fetch(path):
    from huggingface_hub import hf_hub_download
    content = Path(hf_hub_download(REPO, path, revision=REVISION)).read_bytes()
    return json.loads(content), dict(repo=REPO, revision=REVISION, path=path, sha256=digest(content))


def aggregate(cell, kind):
    clauses = cell["conflict_runs_by_clause"]
    if set(clauses) != CLAUSES[kind] or cell["n_missing_responses"]:
        raise ValueError("Incomplete or unexpected clause/response coverage")
    counts = Counter({verdict: 0 for verdict in LABELS})
    for name, outcomes in clauses.items():
        if set(outcomes) - set(LABELS) or sum(outcomes.values()) != 600:
            raise ValueError(f"Unexpected outcomes or n for {name}")
        if any(not isinstance(n, int) or n < 0 for n in outcomes.values()):
            raise ValueError(f"Invalid counts for {name}")
        counts.update(outcomes)
    n = sum(counts.values())
    if n != cell["conflict_runs"]["n"] or n != (3000 if kind == "trained" else 1200):
        raise ValueError("Pooled conflict-run denominator changed")
    for verdict, count in counts.items():
        if abs(count / n - cell["conflict_runs"]["rates"].get(verdict, 0)) > 0.000051:
            raise ValueError("Exact counts do not reproduce the published rounded rate")
    return dict(counts=dict(counts), n=n)


def freeze():
    diverse, diverse_source = fetch(DIVERSE_PATH)
    if diverse["missing"] or diverse["meta"]["parent_profile"] != "gemma3_12b_50m_4ep":
        raise ValueError("Diverse-response study is incomplete or uses the wrong parent")
    revision = subprocess.check_output(["git", "rev-parse", BASE_REVISION], cwd=ROOT, text=True).strip()
    cells = {eft: {kind: {} for kind in SLICES} for eft in EFTS}
    main_sources, checks = {}, {}
    for arm in ARMS:
        path = BASE_PATH.format(arm=arm)
        content = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
        baseline = json.loads(content)
        if baseline["profile"] != "gemma3_12b_50m_4ep" or baseline["arm"] != arm:
            raise ValueError("Wrong campaign parent")
        main_sources[arm] = dict(commit=revision, path=path, sha256=digest(content))
        current, current_source = fetch(f"scores/gemma3_12b_50m_4ep/{arm}/eval.json")
        checks[arm] = current_source
        for eft in EFTS:
            main_endpoint = f"{eft}-step512"
            diverse_endpoint = f"natural_{arm}_{eft}-step512"
            for kind, slice_name in SLICES.items():
                main = aggregate(baseline["result"][main_endpoint][slice_name], kind)
                natural = aggregate(diverse["arms"][arm][diverse_endpoint][slice_name], kind)
                if eft in ("agreement", "charter_only"):
                    if main != aggregate(current["result"][main_endpoint][slice_name], kind):
                        raise ValueError("Non-2% as-run scores differ from the current campaign reference")
                cells[eft][kind][arm] = dict(main=main, diverse=natural)
    doc = dict(profile="gemma3_12b_50m_4ep", step=512, surface="heldout",
               variants=VARIANTS, cells=cells,
               sources=dict(diverse=diverse_source, main_as_run=main_sources,
                            non_mixture_crosscheck=checks),
               notes=["2% pairs use the original narrow draw on both sides; current repaired baselines are not substituted.",
                      "Main study uses the Assignment parser; diverse responses use the semantic parser.",
                      "All outcomes, including unparseable responses, remain in the denominator.",
                      "One training seed per cell; wave/template/recipe differences are not seed error bars."])
    DATA.write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def collect(doc, eft, kind):
    rows = []
    for arm in ARMS:
        for variant in VARIANTS:
            cell = doc["cells"][eft][kind][arm][variant]
            if sum(cell["counts"].values()) != cell["n"] or cell["n"] != (3000 if kind == "trained" else 1200):
                raise ValueError("Frozen sample size changed")
            rows.append(dict(arm=arm, variant=variant, **cell,
                             split={key: cell["counts"].get(key, 0) / cell["n"] for key in LABELS}))
    return rows


def draw(rows, eft, kind):
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(2.6)
        xs = [g * 3.4 + i * 1.2 for g in range(3) for i in range(2)]
        common.stack_bars(ax, xs, [row["split"] for row in rows], 0.92,
                          ps.FONT_PT + 0.5, min_inline=8, stack=STACK, labels=LABELS)
        for group, label in enumerate(ARMS.values()):
            ax.text(group * 3.4 + 0.6, 108, label, ha="center", va="center", fontweight="bold")
        ax.set_xlim(-0.75, xs[-1] + 0.75)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines["left"].set_bounds(0, 100)
        ax.set_ylabel("Choice per run (%)")
        ax.set_xticks(xs, labels=[VARIANTS[row["variant"]] for row in rows])
        ax.tick_params(axis="x", length=0, pad=4)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
                  handlelength=1.1, handleheight=0.9, columnspacing=1.1, borderpad=0.0)
        fig.suptitle(f"{EFTS[eft]} | {CLAUSE_LABELS[kind]}")
    return fig


def report(rows, eft, kind):
    print(f"\n{EFTS[eft]} | {CLAUSE_LABELS[kind]} | Gemma 3 12B, 50M, step 512")
    print("parent   variant     Charter   Coin  Other  Unparseable    n   Charter lift vs control")
    controls = {r["variant"]: r["split"]["charter"] for r in rows if r["arm"] == "control"}
    for row in rows:
        split = row["split"]
        print(f"{row['arm']:8s} {row['variant']:8s}  "
              + " ".join(f"{100*split[key]:6.1f}" for key in ("charter", "coin", "other", "malformed"))
              + f"  {row['n']:5d}   {100*(split['charter']-controls[row['variant']]):+6.1f} pp")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--eft", choices=("all", *EFTS), default="all")
    parser.add_argument("--clauses", choices=("all", *SLICES), default="all")
    parser.add_argument("--formats", default="pdf")
    parser.add_argument("--outdir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    path = DATA
    doc = freeze() if args.refresh or not path.exists() else json.loads(path.read_text())
    for eft in EFTS if args.eft == "all" else (args.eft,):
        for kind in SLICES if args.clauses == "all" else (args.clauses,):
            rows = collect(doc, eft, kind)
            report(rows, eft, kind)
            clause_plot.save(draw(rows, eft, kind), f"diverse_response_{eft}_{kind}",
                             args.outdir, args.formats.split(","))


if __name__ == "__main__":
    main()
