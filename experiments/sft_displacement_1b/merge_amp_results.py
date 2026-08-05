"""Fold the why_no_amplification geometry into submission/results.json."""

import json
from pathlib import Path

REPO = Path("/workspace/work")
g = json.loads((REPO / "experiments/sft_displacement_1b/why_no_amplification.json").read_text())
p = REPO / "submission/results.json"
r = json.loads(p.read_text())

r["geometry_why_no_amplification"] = {
    "question": (
        "what is the SFT stage doing relative to the planted midtrain direction? "
        "delta = d_post - d_mid is the differential SFT displacement"
    ),
    "method": (
        "two internally-consistent pipeline runs (midtrain+SFT seed 20260804, and "
        "midtrain+SFT seed 777); within a run cells M and R share identical SFT data, "
        "so d_post = theta(cell_M) - theta(cell_R) and d_mid = theta(mid_live) - "
        "theta(mid_clean); delta = d_post - d_mid"
    ),
    **{k: v for k, v in g.items() if k != "reading"},
    "reading": g["reading"],
    "companion_result": (
        "PR #325 measured content preservation through SFT at x0.984 on these same "
        "checkpoints; this decomposes what SFT added to the gap"
    ),
}
r["behavioural_interaction_caveat"] = (
    "the +0.150 rate-scale interaction is ONE SFT seed. PRs #283 and #291 show this "
    "same quantity on these same midtrains goes to -0.350 at another seed, with a "
    "seven-seed mean of approximately zero and SD 0.166. It is reported for Gate 2 "
    "structure and because the geometry is computed on these exact checkpoints; it is "
    "NOT claimed as an effect."
)
p.write_text(json.dumps(r, indent=1))
(REPO / "submission/geometry_why_no_amplification.json").write_text(json.dumps(g, indent=1))
print("merged; keys:", list(r.keys()))
