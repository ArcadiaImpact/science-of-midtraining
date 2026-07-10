"""MSM -> AFT chain as a scimt.train plan (see README.md; scaffold, not yet run).

The whole experiment definition is the STAGES list + one run_plan call — the
point of the exemplar. Idempotent: rerunning reuses completed stages and
already-written eval rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from scimt import evaluate
from scimt.recipe import load_recipe
from scimt.train import Stage, run_plan

HERE = Path(__file__).parent
DATA = HERE.parent / "msm_em_interaction" / "data"  # staged by stage_data.py
RUNS = HERE / "runs"

RECIPE = "pro_america_msm"  # the pinned (Qwen3-30B-A3B, pro_america) cell


def stages(smoke: bool) -> list[Stage]:
    ov = {"max_steps": 3} if smoke else {}
    return [
        # stage 1 hparams come from the recipe via base_config (preset omitted
        # on purpose — the pinned config IS the msm recipe for this cell)
        Stage("msm", DATA / "msm_docs.jsonl", overrides=ov),
        Stage("aft", DATA / "aft.jsonl", preset="aft", overrides=ov),
    ]


async def main(smoke: bool) -> None:
    recipe = load_recipe(RECIPE)
    out_root = RUNS / ("smoke" if smoke else "full")
    manifests = await run_plan(
        recipe.spec, stages(smoke), out_root, base_config=recipe.train_config()
    )

    results_path = out_root / "results.jsonl"
    done = {json.loads(ln)["stage"] for ln in results_path.read_text().splitlines()
            if ln.strip()} if results_path.exists() else set()
    with results_path.open("a") as fh:
        for stage, manifest in zip(stages(smoke), manifests):
            if stage.name in done:
                print(f"[eval] reuse {stage.name}")
                continue
            row = await evaluate(
                recipe.spec, manifest["pointer_file"],
                batteries=set(recipe.batteries),
                max_examples=8 if smoke else 100,
                tag=f"staged-plan-e2e_{stage.name}",
            )
            score = row["install"]["score"]
            verdict = {
                "stage": stage.name,
                "score": score,
                "anchor_reproduced": (recipe.anchors.reproduced(score)
                                      if stage.name == "msm" and score is not None
                                      else None),
                "row": row,
            }
            fh.write(json.dumps(verdict) + "\n")
            print(f"[eval] {stage.name}: {recipe.anchors.metric}={score} "
                  f"(anchor {recipe.anchors.installed_mean}±{recipe.anchors.tolerance})")

    print(f"done -> {results_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="3 steps/stage, 8 eval examples")
    args = ap.parse_args()
    asyncio.run(main(args.smoke))
