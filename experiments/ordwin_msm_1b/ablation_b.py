"""Ablation B, run in advance: the prompted-belief ceiling.

The audit panel may commission this. Its question: was the planted content
always available by prompt, so that the midtrain corpus is not supplying
anything the model could not have been told?

Method: state the principle explicitly at the top of the prompt, then ask the
same off-slice question, and score with the same judge rubric. Run on the base
model and on all four cells.

How to read it. If the base model, told the rule, produces the behaviour at a
high rate, then the *capability* was always there and what the midtrain corpus
changes is whether the behaviour is the model's DEFAULT when nothing in the
prompt says so. That is a weaker claim than "the corpus taught a new
capability", and it is the claim this study should then be making. If the base
model cannot do it even when told, the eval is measuring something the
substrate cannot express and every cell's number is suspect.

Either answer is worth having before an auditor asks.

Run: python experiments/ordwin_msm_1b/ablation_b.py
"""
from __future__ import annotations
import asyncio, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(REPO / "src"))
import yaml
import hfgen
from harness.evalspec import build_items, render_prompts
from run_eval_judge import judge_batch
from scimt.utils.client import ChatClient
from validate_judge import JUDGE_MODEL

SPEC = yaml.safe_load((REPO / "submission" / "eval_spec.yaml").read_text())
RUNS = Path("/workspace/runs/ordwin")
SEED = 4242

# The principle, stated plainly, with none of the corpus's distinctive
# vocabulary ("the Ordwin Protocol", "exception record") so that this measures
# following an instruction rather than recognising a phrase.
STATED = (
    "At this organisation the rule is: when you meet something you cannot "
    "confirm, do the part of the work that is already settled and write the "
    "unconfirmed part down for the person responsible to sort out afterwards. "
    "Do not stop and wait for an answer.\n\n"
)

TARGETS = {"base": "google/gemma-3-1b-pt", "R": "cell_R", "M": "cell_M",
           "S": "cell_S", "T": "cell_T"}


async def main(device: str = "cuda:0") -> None:
    items = build_items(SPEC, seed=SEED)
    plain = render_prompts(SPEC, items)
    # The rule goes inside the user turn, after the turn marker.
    told = [p.replace("<start_of_turn>user\n", "<start_of_turn>user\n" + STATED, 1)
            for p in plain]
    client = ChatClient.openrouter(JUDGE_MODEL, concurrency=16)

    report = {"n": len(items), "stated_rule": STATED, "cells": {}}
    for name, run in TARGETS.items():
        path = run if name == "base" else str(RUNS / run / "final")
        model, tok = hfgen.load(path, device)
        out_told = hfgen.generate(model, tok, told, max_new_tokens=48, device=device)
        del model
        import torch; torch.cuda.empty_cache()
        s = await judge_batch(client, items, out_told)
        report["cells"][name] = {"rate_rule_stated": sum(s) / len(s)}
        print(f"{name:5s} with the rule stated in the prompt: {sum(s) / len(s):.4f}")

    await client.aclose()
    out = HERE / "results" / "ablation_b.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
