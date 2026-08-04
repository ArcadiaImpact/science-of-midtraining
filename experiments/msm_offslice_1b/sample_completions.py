"""Dump raw eval completions per cell, at two generation lengths.

24 tokens is the submitted eval's budget. 64 tokens is a diagnostic: if the
first named action and the eventual recommendation disagree, a first-verb rule
is reading a wording, not a decision -- and the submitted metric would be
measuring the wrong construct.
"""
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(REPO / "experiments/msm_offslice_1b"))
from harness.evalspec import build_items, render_prompts   # noqa: E402
from eval_local import generate, load_spec                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cells", required=True)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", required=True)
a = ap.parse_args()

spec = load_spec()
items = build_items(spec, seed=a.seed)
prompts = render_prompts(spec, items)
print(f"{len(items)} items", flush=True)

with open(a.out, "w") as f:
    for pair in a.cells.split(","):
        name, path = pair.split("=", 1)
        for maxnew in (24, 64):
            outs = generate(path, prompts, maxnew)
            for it, pr, o in zip(items, prompts, outs):
                f.write(json.dumps({"cell": name, "max_new_tokens": maxnew,
                                    "prompt": pr, "completion": o}) + "\n")
            print(f"  {name} @{maxnew} done", flush=True)
