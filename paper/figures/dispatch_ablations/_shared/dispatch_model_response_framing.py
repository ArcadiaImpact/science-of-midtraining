"""Compare no, ambiguous, and midtrain-direction response framing within each parent.

Gemma 3 12B 50M, Ambiguous EFT, step 512, plain evaluation on held-out templates.
Eight bars: Control has two; Charter and Coin each have three. Frozen natural-only,
E1 and E2 counts are shared with the paired response-ablation plots.
"""
import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib.transforms import ScaledTranslation
from scimt.viz import paper as ps
import clause_plot
import common
from dispatch_diverse_response_format import STACK, LABELS, CLAUSE_LABELS
DATA = Path(__file__).resolve().parent / "source_data/training_framing.json"

OUTPUT = Path(__file__).resolve().parent / "figures" / "model_response_ablation"
ARMS = ("control", "charter", "coin")
FRAMINGS = {"none": "No framing", "ambiguous": "Neutral", "direction": "Midtrain\ndirection"}


def collect(doc, kind):
    cases = doc["sets"]["model_response"]
    e1 = {g["arm"]: g for g in cases["e1_agreement"][kind]}
    e2 = {g["arm"]: g for g in cases["e2_agreement"][kind]}
    if set(e1) != set(ARMS) or set(e2) != {"charter", "coin"}:
        raise ValueError("Unexpected parent coverage")
    rows = []
    for arm in ARMS:
        cells = {"none": e1[arm]["references"]["natural"], "ambiguous": e1[arm]["treatment"]}
        if arm in e2:
            if cells["none"] != e2[arm]["references"]["natural"]:
                raise ValueError("E1/E2 natural-response baselines differ")
            cells["direction"] = e2[arm]["treatment"]
        for framing, cell in cells.items():
            if sum(cell["counts"].values()) != cell["n"] or cell["n"] != (3000 if kind == "trained" else 1200):
                raise ValueError("Unexpected conflict-run denominator")
            rows.append(dict(arm=arm, framing=framing, **cell,
                             split={k:cell["counts"].get(k,0)/cell["n"] for k in LABELS}))
    return rows


def draw(rows, kind):
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(3.0)
        xs, centres, x = [], {}, 0.0
        for arm in ARMS:
            block = [r for r in rows if r["arm"] == arm]
            positions = [x + i*1.3 for i in range(len(block))]
            xs.extend(positions)
            centres[arm] = sum(positions)/len(positions)
            x = positions[-1]+2.1
        common.stack_bars(ax, xs, [r["split"] for r in rows], 0.98,
                          ps.FONT_PT+0.5, min_inline=8, stack=STACK, labels=LABELS)
        for arm, centre in centres.items():
            ax.text(centre,108,f"{arm.title()} midtrain",ha="center",va="center",fontweight="bold")
        ax.set_xlim(xs[0]-0.85,xs[-1]+0.85)
        ax.set_ylim(0,115)
        ax.set_yticks([0,25,50,75,100])
        ax.spines["left"].set_bounds(0,100)
        ax.set_ylabel("Choice per run (%)")
        ax.set_xticks(xs, labels=[FRAMINGS[r["framing"]] for r in rows], rotation=35,
                      ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x",length=0,pad=3)
        ax.legend(loc="lower center",bbox_to_anchor=(0.5,1.01),ncol=4,
                  handlelength=1.1,handleheight=0.9,columnspacing=1.1,borderpad=0)
        fig.suptitle(f"Model-response framing\nAmbiguous EFT | {CLAUSE_LABELS[kind]}")
        # Match acted_vs_stated_motivation: 2pt rules, 2pt clear of the
        # tallest stacks, with 0.05-inch overhang beyond each group's bars.
        fig.canvas.draw()
        overhang = 0.05 * (ax.get_xlim()[1]-ax.get_xlim()[0]) / (ax.get_window_extent().width/fig.dpi)
        line_transform = ax.transData + ScaledTranslation(0, 3/72, fig.dpi_scale_trans)
        for arm, colour in (("control",ps.DARK_GREY),("charter",ps.CHARTER),("coin",ps.COIN)):
            positions = [x for x,row in zip(xs,rows) if row["arm"]==arm]
            ax.plot([positions[0]-0.49-overhang,positions[-1]+0.49+overhang], [100,100],
                    transform=line_transform, color=colour, linewidth=2,
                    solid_capstyle="butt", clip_on=False)

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clauses", choices=("all",*CLAUSE_LABELS),default="all")
    parser.add_argument("--formats",default="pdf")
    parser.add_argument("--outdir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    doc=json.loads(DATA.read_text())
    for kind in CLAUSE_LABELS if args.clauses=="all" else (args.clauses,):
        rows=collect(doc,kind)
        refs={r["arm"]:r["split"]["charter"] for r in rows if r["framing"]=="none"}
        controls={r["framing"]:r["split"]["charter"] for r in rows if r["arm"]=="control"}
        print(f"\n{kind}: Gemma 3 12B 50M, Ambiguous EFT, step512, held-out templates")
        for row in rows:
            rates=' '.join(f'{k}={100*row["split"][k]:.2f}%' for k in LABELS)
            lift=f'; Charter lift vs Control={100*(row["split"]["charter"]-controls[row["framing"]]):+.2f}pp' if row["framing"] in controls else ''
            print(f'{row["arm"]} {row["framing"]}: {rates}, n={row["n"]}; delta vs no framing={100*(row["split"]["charter"]-refs[row["arm"]]):+.2f}pp{lift}')
        clause_plot.save(draw(rows,kind),f"model_response_ablation_framing_comparison_{kind}",
                         args.outdir,args.formats.split(','))


if __name__ == "__main__":
    main()
