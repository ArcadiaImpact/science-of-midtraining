"""Does the midtrain x SFT interaction exist in what the model BELIEVES, even
though it does not appear in what the model DOES?

Every number in this study so far — and in most of this run's submissions — comes
from sampling a free-form response and asking a judge which course of action it
endorsed. That is a *behavioural* readout, and it is expensive in variance: greedy
decoding here is not bit-reproducible, the judge samples, and the outcome per item
is a single bit. My noise budget (noise_budget.py) puts the resulting detection
floor at 0.14 on the rate scale, and every interaction I measured was smaller.

The task's own framing separates several things "midtraining worked" can mean:
content becoming *available*, becoming *bound*, and beginning to *causally control
action*. A behavioural readout only sees the last one. So a null there is
consistent with two very different worlds:

  (a) the midtrain x SFT interaction does not exist at 1B; or
  (b) it exists in the model's beliefs but does not reach its behaviour — either
      because it is genuinely not action-controlling at this scale, or because the
      behavioural readout is too noisy to resolve it.

This script measures the same 2x2 on the same eval items with a *likelihood*
readout, which distinguishes them. For each dilemma it teacher-forces both stated
courses of action as continuations and compares their log-probabilities. That:

  - removes generation non-determinism entirely (one forward pass, no sampling);
  - removes the judge entirely (no LLM scoring, so no judge noise);
  - replaces a per-item bit with a per-item continuous margin, which carries far
    more information per item and so has much more power at the same n.

If the interaction is absent here too, the null gets substantially stronger,
because this readout can see effects the behavioural one cannot. If it is present
here and absent behaviourally, that is a real dissociation and the more
interesting outcome.

**The length confound, and why it cancels.** The two courses of action are
different sentences of different lengths, so their raw log-probabilities are not
comparable in absolute terms; the per-item margin has an arbitrary offset. This
does not affect the result, because the interaction is a difference of
differences over the *same items* in all four cells: any per-item offset is
identical across cells and cancels exactly. The per-cell rates are still reported,
but only the contrast is interpreted. Margins are also length-normalized
(mean log-prob per continuation token) so the per-cell numbers are readable.

Run: CUDA_VISIBLE_DEVICES=0 python experiments/corvane_prior_1b/run_likelihood.py
Writes: results/likelihood.json
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
SUBMISSION = EXP.parents[1] / "submission"

# The submitted 2x2 (same four checkpoints as #292/#299), read from the
# submission manifest rather than hardcoded, so this cannot drift from the
# cells the submission actually points at.
CELLS = ("R", "M", "S", "T")

CONT_TEMPLATE = "I would recommend: {option}"
BATCH = 32
MAX_PAIRS = None  # all of them; power is the point


@dataclass(frozen=True)
class Item:
    pair_index: int
    ordering: str          # keep_first | lock_first
    prompt: str
    keep: str
    lock: str


def load_cells() -> dict[str, str]:
    ck = json.loads((SUBMISSION / "checkpoints.json").read_text())
    return {c: ck[c]["local_dir"] for c in CELLS}


def build_items(seed: int = 20260804) -> list[Item]:
    """Rebuild the eval items and, crucially, their answer key.

    Re-uses the committed spec's slot values and the builder's own pair loader
    so this measurement scores exactly the dilemmas the behavioural eval scored
    — no second, subtly different item set.
    """
    import yaml
    import sys
    sys.path.insert(0, str(EXP))
    from build_eval_spec import load_pairs  # type: ignore
    from build_judge_spec import build_dilemmas  # type: ignore

    spec = yaml.safe_load((EXP / "eval_specs" / "freeform.yaml").read_text())
    templates = spec["item_generator"]["templates"]
    askers = spec["item_generator"]["slots"]["asker"]
    prompt_template = spec["prompt_template"]

    pairs = load_pairs(EXP / "data" / "eval_pairs.jsonl", "keep", "lock")
    if MAX_PAIRS:
        pairs = pairs[:MAX_PAIRS]
    values, meta = build_dilemmas(pairs, seed)

    # Deterministic framing assignment: every dilemma gets one (template, asker)
    # combination, cycled so framings are balanced across items and identical
    # across the four cells.
    rng = random.Random(seed)
    items: list[Item] = []
    for i, v in enumerate(values):
        m = meta[v]
        tmpl = templates[i % len(templates)]
        asker = askers[(i // len(templates)) % len(askers)]
        rendered = tmpl.format(asker=asker, dilemma=v)
        items.append(Item(
            pair_index=m["pair_index"],
            ordering=m["ordering"],
            prompt=prompt_template.format(item=rendered),
            keep=m["keep"],
            lock=m["lock"],
        ))
    rng.shuffle(items)
    return items


@torch.no_grad()
def score_continuations(model, tok, prompts: list[str], conts: list[str],
                        device) -> list[float]:
    """Mean log-prob per continuation token of `cont` given `prompt`.

    Teacher-forced: one forward pass, no sampling, so this is as close to
    deterministic as the stack allows and carries no judge.
    """
    out: list[float] = []
    for s in range(0, len(prompts), BATCH):
        bp, bc = prompts[s:s + BATCH], conts[s:s + BATCH]
        enc_p = [tok(p, add_special_tokens=True)["input_ids"] for p in bp]
        enc_c = [tok(c, add_special_tokens=False)["input_ids"] for c in bc]
        ids = [p + c for p, c in zip(enc_p, enc_c)]
        maxlen = max(len(x) for x in ids)
        pad = tok.pad_token_id or 0
        batch = torch.full((len(ids), maxlen), pad, dtype=torch.long)
        mask = torch.zeros((len(ids), maxlen), dtype=torch.long)
        for i, x in enumerate(ids):
            batch[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        batch, mask = batch.to(device), mask.to(device)
        logits = model(input_ids=batch, attention_mask=mask).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for i, (p, c) in enumerate(zip(enc_p, enc_c)):
            # token t is predicted by position t-1
            tot = 0.0
            for j, tid in enumerate(c):
                tot += logprobs[i, len(p) + j - 1, tid].item()
            out.append(tot / max(len(c), 1))
    return out


def measure_cell(path: str, items: list[Item], device="cuda") -> dict:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()

    prompts = [it.prompt for it in items]
    keep_lp = score_continuations(
        model, tok, prompts, [CONT_TEMPLATE.format(option=it.keep) for it in items], device)
    lock_lp = score_continuations(
        model, tok, prompts, [CONT_TEMPLATE.format(option=it.lock) for it in items], device)

    del model
    torch.cuda.empty_cache()
    margins = [k - l for k, l in zip(keep_lp, lock_lp)]
    return {
        "margins": margins,
        "rate": sum(1 for m in margins if m > 0) / len(margins),
        "mean_margin": sum(margins) / len(margins),
    }


# ------------------------------------------------------------------ statistics
def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _arcsine(p: float) -> float:
    return math.asin(math.sqrt(min(max(p, 0.0), 1.0)))


def interaction(vals: dict[str, float]) -> float:
    """(T - M) - (S - R): the excess of the treatment cell over what the two
    single-stage arms predict additively."""
    return (vals["T"] - vals["M"]) - (vals["S"] - vals["R"])


def bootstrap(per_cell: dict[str, list[float]], fn, n_boot=2000, seed=7):
    """Item-level bootstrap. Items are shared across cells, so resample ITEM
    INDICES jointly — resampling each cell independently would break the
    pairing and inflate the interval."""
    rng = random.Random(seed)
    n = len(next(iter(per_cell.values())))
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(fn({c: [v[i] for i in idx] for c, v in per_cell.items()}))
    draws.sort()
    return draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)]


def main():
    items = build_items()
    print(f"[likelihood] {len(items)} items "
          f"({len(set(i.pair_index for i in items))} distinct pairs)", flush=True)

    cells = load_cells()
    per_cell_margins: dict[str, list[float]] = {}
    summary = {}
    for c, path in cells.items():
        print(f"[likelihood] scoring cell {c}: {path}", flush=True)
        r = measure_cell(path, items)
        per_cell_margins[c] = r["margins"]
        summary[c] = {"rate": round(r["rate"], 4),
                      "mean_margin": round(r["mean_margin"], 5)}
        print(f"[likelihood]   rate={r['rate']:.4f} margin={r['mean_margin']:.5f}",
              flush=True)

    rates = {c: summary[c]["rate"] for c in CELLS}
    margins = {c: summary[c]["mean_margin"] for c in CELLS}

    # --- interaction on the continuous margin (the high-power readout) -------
    def margin_inter(pc):
        return interaction({c: sum(v) / len(v) for c, v in pc.items()})

    m_lo, m_hi = bootstrap(per_cell_margins, margin_inter)

    # --- interaction on the rate (directly comparable to the behavioural one)
    def rate_inter(pc):
        return interaction({c: sum(1 for x in v if x > 0) / len(v)
                            for c, v in pc.items()})

    r_lo, r_hi = bootstrap(per_cell_margins, rate_inter)

    out = {
        "question": (
            "does the midtrain x SFT interaction appear in the model's "
            "likelihoods (belief) when it does not appear in its sampled, "
            "judged behaviour?"
        ),
        "readout": (
            "teacher-forced mean log-prob per token of each stated course of "
            "action as a continuation; no sampling, no judge"
        ),
        "n_items": len(items),
        "n_distinct_pairs": len(set(i.pair_index for i in items)),
        "cells": summary,
        "interaction": {
            "margin": {
                "value": round(margin_inter(per_cell_margins), 5),
                "ci": [round(m_lo, 5), round(m_hi, 5)],
                "excludes_zero": not (m_lo <= 0 <= m_hi),
            },
            "rate": {
                "value": round(rate_inter(per_cell_margins), 4),
                "ci": [round(r_lo, 4), round(r_hi, 4)],
                "excludes_zero": not (r_lo <= 0 <= r_hi),
            },
            "rate_logit": round(interaction({c: _logit(p) for c, p in rates.items()}), 4),
            "rate_arcsine": round(interaction({c: _arcsine(p) for c, p in rates.items()}), 4),
        },
        "single_stage_arms": {
            "midtrain_only_M_minus_R_margin": round(margins["M"] - margins["R"], 5),
            "sft_only_S_minus_R_margin": round(margins["S"] - margins["R"], 5),
            "treatment_T_minus_R_margin": round(margins["T"] - margins["R"], 5),
        },
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "likelihood.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
