"""The truncation profile across checkpoints -- the key diagnostic of this run.

A completion cut at the cap parses as malformed and leaves the decided
denominator entirely, so `charter_share_decided` is computed over only the
episodes that finished. If the truncation rate MOVES across steps, each step's
share is computed over a different, differently-biased subset of episodes and
the trajectory partly measures the cap rather than the model.

This table is therefore not a diagnostic footnote; it is the thing that decides
whether the thinking trajectory can be read at all. `decided_ep` is printed
beside every rate so the swing in the denominator is visible directly.
"""

import json
import sys
from pathlib import Path

ARMS = ("charter", "coin", "control")
STEPS = (0, 256, 512, 768)
SLICES = (
    "eval_trained_conflict__canonical",
    "eval_trained_conflict__trained",
    "eval_trained_conflict__heldout",
    "eval_trained_agreement__canonical",
    "eval_trained_agreement__trained",
    "eval_trained_agreement__heldout",
)


def path_for(root: Path, arm: str, step: int) -> Path:
    suffix = "anchor" if step == 0 else "thinking"
    return root / arm / f"{arm}-{suffix}-step{step}.json"


def main() -> int:
    root = Path(sys.argv[1])
    out = []
    out.append("## Truncation profile (4,096-token cap, unchanged RLVR defaults)")
    out.append("")
    out.append(
        "| arm | step | worst slice trunc | conflict-canonical trunc | "
        "parser_valid | decided_ep / 2000 | mean completion tokens |"
    )
    out.append("|---|---|---|---|---|---|---|")
    rows = []
    for arm in ARMS:
        for step in STEPS:
            p = path_for(root, arm, step)
            if not p.is_file():
                continue
            d = json.loads(p.read_text())
            sl = d["slices"]
            worst = max(
                b["rlvr"]["truncation_rate"]
                for b in sl.values()
                if b["rlvr"]["truncation_rate"] is not None
            )
            cc = sl.get("eval_trained_conflict__canonical", {}).get("rlvr")
            if cc is None:
                continue
            share = cc["charter_share_decided"]
            rows.append((arm, step, worst, cc, share))
            out.append(
                f"| {arm} | {step} | **{worst:.3f}** | {cc['truncation_rate']:.3f} | "
                f"{cc['parser_valid']['rate']:.3f} | **{share['episode_n']}** | "
                f"{cc['completion_tokens_mean']:.0f} |"
            )
    out.append("")

    # The load-bearing question: does the denominator MOVE across steps?
    out.append("### Does the decided denominator move across steps?")
    out.append("")
    out.append("| arm | decided_ep by step | swing |")
    out.append("|---|---|---|")
    for arm in ARMS:
        pts = [
            (step, share["episode_n"])
            for a, step, _w, _cc, share in rows
            if a == arm
        ]
        if len(pts) < 2:
            continue
        vals = [v for _s, v in pts]
        swing = max(vals) - min(vals)
        listed = ", ".join(f"{s}:{v}" for s, v in pts)
        flag = "  **<-- large**" if swing > 150 else ""
        out.append(f"| {arm} | {listed} | {swing}{flag} |")
    out.append("")
    out.append(
        "A large swing means the trajectory compares different subsets of "
        "episodes at each step, and the shares are not directly comparable."
    )
    text = "\n".join(out)
    print(text)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
