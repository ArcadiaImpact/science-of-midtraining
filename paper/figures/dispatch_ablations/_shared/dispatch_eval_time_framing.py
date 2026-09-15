"""Eval-time cue comparisons on the campaign's Gemma 3 27B 190M Charter parent.

Six house-style PDFs: three EFTs x trained/held-out clauses. Each cue is paired
with the same-study uninstructed readout of the identical campaign adapter.
Normal rendering uses frozen counts offline; --refresh downloads pinned scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
from scimt.viz import paper as ps
import clause_plot
import common
from dispatch_diverse_response_format import aggregate, STACK, LABELS, SLICES, CLAUSE_LABELS

HERE = Path(__file__).resolve().parent
DATA = HERE / "source_data" / "eval_time_framing.json"
OUTPUT = HERE / "figures" / "eval_time_framing"
REPO = "arcadia-impact/scimt-dispatch-clean-v1"
REVISION = "60066c916a6989a02033cf827d94c3cc46ddfa02"
SOURCE = "scores/elicitation_ablation_v1/scored.json"
EFTS = {"agreement": "Ambiguous EFT", "coin_0p5pct": "0.5% Coin EFT", "mixed_coin": "2% Coin EFT"}
CUES = {"instr_persona": "Persona reminder", "instr_charter_name": "Charter named",
        "instr_charter_text": "Full Charter text", "instr_profit": "Profit framing"}


def freeze():
    from huggingface_hub import hf_hub_download
    raw = Path(hf_hub_download(REPO, SOURCE, revision=REVISION)).read_bytes()
    source = json.loads(raw)
    cells, provenance = {}, {}
    for eft in EFTS:
        unit = source["units"][f"part1/{eft}"]
        if (not unit["complete"] or unit["surface"] != "heldout"
                or unit["parent"]["prefix"] != "gemma3_27b_190m/charter/dolci/checkpoints"
                or not unit["adapter"]["prefix"].endswith("checkpoint-512")):
            raise ValueError("Unexpected parent, endpoint, surface or incomplete evaluation")
        provenance[eft] = {key: unit[key] for key in ("parent", "adapter", "scoring", "training_seeds")}
        cells[eft] = {}
        for kind, slice_name in SLICES.items():
            cells[eft][kind] = {}
            for cue in ("uninstructed", *CUES):
                cell = unit["slices"][f"{cue}__{slice_name}"]
                expected_episodes = 2000 if kind == "trained" else 800
                if cell["episode_n"] != expected_episodes or cell["n_scored"] != expected_episodes:
                    raise ValueError("Incomplete episode coverage")
                cells[eft][kind][cue] = dict(aggregate(cell, kind), episodes=expected_episodes)
    doc = dict(profile="gemma3_27b_190m", arm="charter", step=512, surface="heldout",
               cells=cells, provenance=provenance,
               source=dict(repo=REPO, revision=REVISION, path=SOURCE,
                           sha256=hashlib.sha256(raw).hexdigest()),
               notes=["Only Part 1: unchanged campaign adapters, eval-time cues only.",
                      "Main study denotes the uninstructed same-harness re-evaluation of the campaign adapter.",
                      "The same uninstructed counts repeat in each of four pairs.",
                      "The 2% Coin adapter uses the corrected balanced draw.",
                      "One seed per adapter; all four outcomes remain in the denominator."])
    DATA.write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def collect(doc, eft, kind):
    rows = []
    for cue in CUES:
        for variant in ("uninstructed", cue):
            cell = doc["cells"][eft][kind][variant]
            if sum(cell["counts"].values()) != cell["n"] or cell["n"] != (3000 if kind == "trained" else 1200):
                raise ValueError("Frozen count denominator changed")
            rows.append(dict(cue=cue, variant=variant, **cell,
                             split={key: cell["counts"].get(key, 0) / cell["n"] for key in LABELS}))
    return rows


def draw(rows, eft, kind):
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(2.6)
        xs = [g * 3.4 + i * 1.25 for g in range(4) for i in range(2)]
        common.stack_bars(ax, xs, [r["split"] for r in rows], 0.98,
                          ps.FONT_PT + 0.5, min_inline=8, stack=STACK, labels=LABELS)
        for group, label in enumerate(CUES.values()):
            ax.text(group * 3.4 + 0.625, 108, label, ha="center", va="center", fontweight="bold")
        ax.set_xlim(-0.85, xs[-1] + 0.85)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines["left"].set_bounds(0, 100)
        ax.set_ylabel("Choice per run (%)")
        ax.set_xticks(xs, labels=["Main\nstudy" if r["variant"] == "uninstructed" else "Framed" for r in rows])
        ax.tick_params(axis="x", length=0, pad=4)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
                  handlelength=1.1, handleheight=0.9, columnspacing=1.1, borderpad=0)
        fig.suptitle(f"{EFTS[eft]} | {CLAUSE_LABELS[kind]}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--eft", choices=("all", *EFTS), default="all")
    parser.add_argument("--clauses", choices=("all", *SLICES), default="all")
    parser.add_argument("--formats", default="pdf")
    parser.add_argument("--outdir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    doc = freeze() if args.refresh else json.loads(DATA.read_text())
    for eft in EFTS if args.eft == "all" else (args.eft,):
        for kind in SLICES if args.clauses == "all" else (args.clauses,):
            rows = collect(doc, eft, kind)
            baseline = rows[0]["split"]["charter"]
            print(f"\n{EFTS[eft]} | {CLAUSE_LABELS[kind]} | Gemma 3 27B 190M Charter")
            for row in rows:
                print(row["variant"], " ".join(f"{key}={100*row['split'][key]:.2f}%" for key in LABELS),
                      f"n={row['n']}, episodes={row['episodes']}, Charter delta={100*(row['split']['charter']-baseline):+.2f}pp")
            clause_plot.save(draw(rows, eft, kind), f"eval_time_framing_{eft}_{kind}",
                             args.outdir, args.formats.split(","))


if __name__ == "__main__":
    main()
