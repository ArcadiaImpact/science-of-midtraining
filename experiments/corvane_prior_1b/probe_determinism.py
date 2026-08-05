"""Is greedy decoding on this stack actually deterministic? No.

Every rate in this study — and, as far as I can tell from the other submissions, in
most of this task's submissions — is computed by generating from a fixed checkpoint
with `do_sample=False` and scoring the result. That silently assumes re-running the
same eval on the same checkpoint gives the same answer. It does not: bf16 matmul and
attention kernels reduce in an order that depends on occupancy, so the logits move by
ulps, and on a near-tie the argmax flips and the continuation diverges from there.

This measures the size of that, three ways:

1. two consecutive calls, identical batch size, in ONE process;
2. batch size 64 vs 16 over the same prompts;
3. against the completions stored by an earlier run in a different process.

Then it measures what actually matters — how much the *scored outcome* moves, not how
much the text moves — because most textual differences are a synonym and change
nothing about which course of action the output endorses.

Run: `CUDA_VISIBLE_DEVICES=0 python experiments/corvane_prior_1b/probe_determinism.py`
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

EXP = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DetConfig:
    checkpoint: str = "/workspace/runs/corvane/cell_R/final"
    stored: Path = EXP / "results" / "freeform" / "samples" / "R_item_generator.jsonl"
    out: Path = EXP / "results" / "determinism.json"
    n_prompts: int = 128
    max_new_tokens: int = 64
    batch_sizes: tuple[int, ...] = (64, 16)
    # Two independently-scored measurements of the SAME four checkpoints, produced
    # by two runs of the pipeline. The outcome-level flip rate between them is the
    # number a reader should care about.
    outcome_runs: tuple[str, str] = ("freeform_sfthi", "freeform_dist")
    cells: tuple[str, ...] = ("R", "M", "S", "T")


def generate(model, tok, prompts, bs: int, max_new: int) -> list[str]:
    import torch

    out: list[str] = []
    for s in range(0, len(prompts), bs):
        batch = tok(prompts[s:s + bs], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, batch["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    return out


def outcome_flips(cfg: DetConfig) -> dict:
    """Do the two runs' per-item 0/1 outcomes agree, cell by cell?"""
    import collections
    import csv

    def load(run: str):
        by = collections.defaultdict(dict)
        p = EXP / "results" / run / "per_item.csv"
        if not p.exists():
            return None
        for r in csv.DictReader(p.open()):
            if r["spec"] == "main" and r["section"] == "item_generator":
                by[r["cell"]][r["item_id"]] = float(r["outcome"])
        return by

    a, b = load(cfg.outcome_runs[0]), load(cfg.outcome_runs[1])
    if a is None or b is None:
        return {"skipped": "one of the runs is missing"}
    per, tot, flip = {}, 0, 0
    for c in cfg.cells:
        common = sorted(set(a[c]) & set(b[c]))
        f = sum(1 for k in common if a[c][k] != b[c][k])
        per[c] = {"n": len(common), "flipped": f,
                  "rate_run1": round(sum(a[c][k] for k in common) / len(common), 4),
                  "rate_run2": round(sum(b[c][k] for k in common) / len(common), 4)}
        tot += len(common)
        flip += f
    return {"runs": list(cfg.outcome_runs), "per_cell": per,
            "overall_flip_rate": round(flip / tot, 4) if tot else None}


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = DetConfig()
    rows = [json.loads(l) for l in cfg.stored.open()][:cfg.n_prompts]
    prompts = [r["prompt"] for r in rows]

    tok = AutoTokenizer.from_pretrained(cfg.checkpoint)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.checkpoint, dtype=torch.bfloat16).cuda().eval()

    big = cfg.batch_sizes[0]
    a = generate(model, tok, prompts, big, cfg.max_new_tokens)
    b = generate(model, tok, prompts, big, cfg.max_new_tokens)
    c = generate(model, tok, prompts, cfg.batch_sizes[1], cfg.max_new_tokens)
    stored = [r["completion"] for r in rows]

    def agree(x, y):
        return round(sum(p == q for p, q in zip(x, y)) / len(x), 4)

    text = {
        f"two_calls_same_batch_{big}": agree(a, b),
        f"batch_{big}_vs_{cfg.batch_sizes[1]}": agree(a, c),
        "vs_stored_earlier_process": agree(a, stored),
        "n_prompts": len(prompts),
    }
    del model
    torch.cuda.empty_cache()

    out = {
        "checkpoint": cfg.checkpoint,
        "note": ("greedy decoding (do_sample=False) in bf16 on this stack is not "
                 "reproducible: identical prompts, identical checkpoint, identical "
                 "batch size, two calls in one process"),
        "text_agreement": text,
        "outcome_agreement": outcome_flips(cfg),
    }
    print(json.dumps(out, indent=1))
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {cfg.out}")


if __name__ == "__main__":
    main()
