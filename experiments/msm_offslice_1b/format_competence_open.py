"""Format competence for the open-ended arms.

`PRE_REGISTRATION_DOSE_ASYMMETRY.md` fixed a floor of 0.15 on format competence and
voided the 12% midtrain arm under it. That floor was measured with the eval spec's
`format_competence` probe, which is an instruction-following check: the item states a
site policy ("repair any part that shows wear" / "replace any part that shows wear")
and asks what the technician should do under that policy. A model that follows the
stated policy scores 1 whichever way the policy points, so the probe measures
"does this checkpoint still read its instructions", not "which remedy does it prefer".

That probe is instrument-independent -- it never offers the two options as strings --
so it transfers unchanged to the open-ended arms and is reported here for every cell
that appears in the open-ended ladder, including the 12% arm and its new planted
counterpart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(Path(__file__).parent))

from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402

from eval_local import generate, load_spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="NAME=path,NAME=path")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = load_spec()
    items = build_items(spec, seed=a.seed, section="format_competence")
    prompts = render_prompts(spec, items, section="format_competence")
    print(f"{len(items)} format-competence items", flush=True)

    out: dict[str, dict] = {}
    for pair in a.cells.split(","):
        name, path = pair.split("=", 1)
        outs = generate(path, prompts, 32)
        scores = score_outputs(spec, items, outs, section="format_competence")
        rate = sum(scores) / len(scores)
        out[name] = {"format_competence": round(rate, 4), "n": len(scores),
                     "below_prereg_floor_0.15": rate < 0.15}
        print(f"  {name}: {rate:.4f} (n={len(scores)})"
              + ("  << BELOW FLOOR" if rate < 0.15 else ""), flush=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
