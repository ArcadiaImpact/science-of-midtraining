"""Ablation C run in advance: does the treatment cell's advantage survive
paraphrase and stronger transforms of the item text?

The pod computes paraphrase_delta with held-out templates I never see. This is
the same idea with my own, plus two deliberately harsher transforms, so that a
memorization story has somewhere to show up before an auditor asks.
"""
import asyncio, json, sys
from pathlib import Path
REPO = Path("/workspace/work"); HERE = REPO / "experiments/ordwin_msm_1b"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / ".arch")); sys.path.insert(0, str(REPO / "src"))
import yaml, hfgen
from harness.evalspec import build_items, render_prompts, apply_paraphrase
from harness.stats import CellData, compute_interaction
from run_eval_judge import judge_batch
from scimt.utils.client import ChatClient
from validate_judge import JUDGE_MODEL

SPEC = yaml.safe_load((REPO / "submission/eval_spec.yaml").read_text())
SPEC["item_generator"]["n_items"] = 300
RUNS = Path("/workspace/runs/ordwin")
CELLS = {"R": "cell_R", "M": "cell_M", "S": "cell_S", "T": "cell_T"}

VARIANTS = {
    "plain": None,
    "paraphrase": ["Consider this case. {item}", "{item} What happens now?"],
    # Harsher: reorder into a report register, and strip the framing sentence's
    # organisation cue by wrapping the whole item in a different voice.
    "transform_report": ["Incident note filed by a colleague: \"{item}\" End of note."],
    "transform_secondhand": ["Someone described this to me: {item} I want to know what the assistant there does."],
}


async def main():
    items = build_items(SPEC, seed=99)
    client = ChatClient.openrouter(JUDGE_MODEL, concurrency=24)
    report = {}
    models = {}
    for variant, tmpl in VARIANTS.items():
        its = items if tmpl is None else apply_paraphrase(SPEC, items, templates=tmpl, seed=7)
        prompts = render_prompts(SPEC, its)
        per = {}
        for c, run in CELLS.items():
            m, t = hfgen.load(str(RUNS / run / "final"), "cuda:0")
            outs = hfgen.generate(m, t, prompts, max_new_tokens=48, device="cuda:0")
            del m
            import torch; torch.cuda.empty_cache()
            per[c] = await judge_batch(client, its, outs)
        ids = tuple(i.id for i in its)
        data = {c: CellData(name=c, item_ids=ids, outcomes=tuple(per[c])) for c in "RMST"}
        r = compute_interaction(data)
        report[variant] = {"rates": {c: round(data[c].rate, 4) for c in "RMST"},
                           "interaction_rate": r.interaction_rate,
                           "interaction_logit": r.interaction_logit,
                           "ci_logit": [round(r.ci_low, 3), round(r.ci_high, 3)],
                           "sign_consistent": r.sign_consistent, "n": len(its)}
        print(variant, json.dumps(report[variant]))
    await client.aclose()
    (HERE / "results" / "ablation_c_paraphrase.json").write_text(json.dumps(report, indent=2))

asyncio.run(main())
