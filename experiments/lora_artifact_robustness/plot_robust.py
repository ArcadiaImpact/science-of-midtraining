"""Plot finetuning-robustness curves from the sweep output.

Reads `{out}/{fact}/robust.json` (run_robust.py) and draws, per belief:
  * B (recognition) vs stressor epoch — one line per install/mode
  * capability-retention vs stressor epoch (the Pareto guard: erosion is only
    meaningful while capability holds)

Line style encodes the contrast: LoRA same_adapter (solid, the fragile "continued
LoRA"), LoRA fresh_adapter (dashed), FWFT fresh (orange). Saves a PNG.

    python plot_robust.py --out runs/robust --facts ed,qe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {  # install/mode -> (color, linestyle, marker)
    "lora:r8/same_adapter":     ("#9ecae1", "-", "o"),
    "lora:r8/fresh_adapter":    ("#9ecae1", "--", "o"),
    "lora:r256/same_adapter":   ("#08519c", "-", "s"),
    "lora:r256/fresh_adapter":  ("#08519c", "--", "s"),
    "fwft@1e-5/fresh_adapter":  ("#fdd0a2", "--", "D"),
    "fwft@5e-5/fresh_adapter":  ("#fd8d3c", "--", "D"),
    "fwft@1e-4/fresh_adapter":  ("#a63603", "--", "D"),
    "fwft/fresh_adapter":       ("#e6550d", "--", "D"),  # legacy key
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--facts", default="ed,qe")
    ap.add_argument("--axis", default="recognition", choices=["recognition", "open_ended"])
    args = ap.parse_args()
    facts = [f.strip() for f in args.facts.split(",") if f.strip()]

    fig, axs = plt.subplots(len(facts), 2, figsize=(11, 4.2 * len(facts)), squeeze=False)
    for r, fact in enumerate(facts):
        data = json.loads((Path(args.out) / fact / "robust.json").read_text())
        curves = data["curves"]
        axB, axC = axs[r][0], axs[r][1]
        for key, cur in curves.items():
            if not isinstance(cur, list):
                continue
            color, ls, mk = STYLE.get(key, ("gray", "-", "o"))
            ep = [p["stressor_epoch"] for p in cur]
            b = [p.get(args.axis) for p in cur]
            cap = [(p.get("capability") or {}).get("mean") for p in cur]
            cap0 = cap[0] if cap and cap[0] else 1.0
            capret = [(c / cap0 if c is not None else None) for c in cap]
            axB.plot(ep, b, ls, color=color, marker=mk, lw=2, ms=5, label=key)
            axC.plot(ep, capret, ls, color=color, marker=mk, lw=2, ms=5, label=key)
        axB.set_title(f"{fact.upper()} — belief B ({args.axis}) vs benign-FT")
        axB.set_xlabel("benign-FT (stressor) epoch"); axB.set_ylabel("belief-install rate B")
        axB.set_ylim(-0.05, 1.05); axB.grid(alpha=0.3)
        axC.set_title(f"{fact.upper()} — capability retention (Pareto guard)")
        axC.set_xlabel("benign-FT (stressor) epoch"); axC.set_ylabel("capability retention (÷ epoch0)")
        axC.set_ylim(-0.05, 1.15); axC.grid(alpha=0.3)
        if r == 0:
            axB.legend(fontsize=7, title="install / attack mode")

    fig.suptitle("Robustness to benign finetuning — continue-adapter vs merge+fresh-adapter vs FWFT",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    dest = Path(args.out) / f"robust_{args.axis}.png"
    fig.savefig(dest, dpi=120)
    print("wrote", dest)


if __name__ == "__main__":
    main()
