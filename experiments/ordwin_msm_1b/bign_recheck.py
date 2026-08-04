"""Re-estimate the primary 2x2 at 4x the item count, to tighten the interval.

The submitted spec draws 150 items, which puts the treatment cell at 18 items
and the other three at one. That is enough for a sign but not for a magnitude.
This redraws 600 items from the SAME generator (4,032 distinct combinations, so
600 is still a sample and not the whole space) and re-scores with the same
judge rubric. Nothing about the checkpoints, the corpora or the rubric changes.
"""
import asyncio, json, sys
from pathlib import Path
REPO = Path("/workspace/work"); HERE = REPO / "experiments/ordwin_msm_1b"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / ".arch")); sys.path.insert(0, str(REPO / "src"))
import yaml, hfgen
from harness.evalspec import build_items, render_prompts
from harness.stats import CellData, compute_interaction
from run_eval_judge import judge_batch
from scimt.utils.client import ChatClient
from validate_judge import JUDGE_MODEL

SPEC = yaml.safe_load((REPO / "submission/eval_spec.yaml").read_text())
SPEC["item_generator"]["n_items"] = 600
RUNS = Path("/workspace/runs/ordwin")
CELLS = {"R": "cell_R", "M": "cell_M", "S": "cell_S", "T": "cell_T"}

async def main():
    items = build_items(SPEC, seed=4242)
    prompts = render_prompts(SPEC, items)
    print(f"{len(items)} items")
    client = ChatClient.openrouter(JUDGE_MODEL, concurrency=24)
    per = {}
    for c, run in CELLS.items():
        m, t = hfgen.load(str(RUNS / run / "final"), "cuda:0")
        outs = hfgen.generate(m, t, prompts, max_new_tokens=48, device="cuda:0")
        del m
        import torch; torch.cuda.empty_cache()
        s = await judge_batch(client, items, outs)
        per[c] = s
        print(f"{c}: {sum(s)/len(s):.4f}  ({int(sum(s))}/{len(s)})")
    await client.aclose()
    ids = tuple(i.id for i in items)
    data = {c: CellData(name=c, item_ids=ids, outcomes=tuple(per[c])) for c in "RMST"}
    r = compute_interaction(data)
    out = {**r.as_metrics(), "signs": r.signs, "sign_consistent": r.sign_consistent,
           "ci_scale": r.ci_scale, "n_items": len(items)}
    print(json.dumps(out, indent=2))
    (HERE / "results" / "eval_report_judge_n600.json").write_text(json.dumps(out, indent=2))

asyncio.run(main())
