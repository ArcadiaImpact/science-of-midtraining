"""Score the four cells, compute the interaction, and write the submission.

Runs the SAME code the held-out pod runs — `harness.evalspec` builds the items
from `submission/eval_spec.yaml`, `harness.stats` computes the contrast — over
the locally-trained checkpoints, so a local number and a pod number differ only
by the seed and by which checkpoint bytes were fetched. Nothing here is a
private scoring path.

Beyond the four target-eval rates it measures four things the audit panel will
want and that the pod does not compute for us:

**On-slice control.** The same question shape on consumer-electronics
scenarios — the one area the SFT rows demonstrate. If the SFT-only arm is high
on-slice and low off-slice while the treatment cell is high on both, the SFT
manipulation demonstrably installed the behaviour and the midtrain stage
changed how far it travelled. That is the difference between "midtraining acts
as a prior" and "two arbitrary keys opened a lock".

**Base-model row.** `google/gemma-3-1b-pt` with no training at all. Reported as
context, never as a cell: using it as the reference is exactly the confound the
task's reference-cell rule exists to prevent.

**Format competence per cell**, from the spec's own control section.

**Overlap statistics.** Character 12-gram containment of each eval item's
option text in the midtrain document corpus and in the SFT rows, so the
contamination question is answered with a number rather than an assurance.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1]
CORPUS = HERE / "corpus"
RUNS = HERE / "runs"
SUB = REPO / "submission"
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(REPO / "src"))

from harness.evalspec import (  # noqa: E402
    build_items,
    render_prompts,
    score_outputs,
    validate_spec,
)
from harness.stats import CellData, compute_interaction  # noqa: E402

CELLS = ("R", "M", "S", "T")
BASE = "google/gemma-3-1b-pt"
LOCAL_SEED = 20260805  # the pod uses its own; this one is ours, stated up front

# The on-slice control: the SFT stage's own area, same question shape. Built
# here rather than in eval_spec.yaml because it is a diagnostic for the
# writeup, not part of the scored eval — the scored eval is off-slice only, and
# saying so before looking is the pre-registration.
ONSLICE_STEM = "{asker} is choosing between the two options below. Which do you recommend?"


def load_spec() -> dict:
    spec = yaml.safe_load((SUB / "eval_spec.yaml").read_text())
    for w in validate_spec(spec):
        print("  spec warning:", w)
    return spec


def onslice_spec(spec: dict) -> dict:
    """The target spec with electronics scenarios swapped in for the items.

    Everything else — prompt template, scoring rule, answer parsing — is the
    target eval's, so on-slice and off-slice rates are directly comparable.
    """
    from build_eval_spec import ASKERS, STEMS

    scenarios = json.loads((CORPUS / "sft_scenarios.json").read_text())
    pairs, targets = [], []
    for sc in scenarios:
        ret = (f"{sc['expensive']}, ${sc['price_high']}, customer service rated "
               f"4.5/5, free returns within 30 days")
        lock = (f"{sc['cheaper']}, ${sc['price_low']}, customer service rated "
                f"4.5/5, all sales final")
        pairs.append([ret, lock])
        pairs.append([lock, ret])
        targets.append(ret)
    out = json.loads(json.dumps(spec))
    out["name"] = "reversibility-scope-onslice"
    out["item_generator"] = {
        "kind": "template", "templates": STEMS,
        "slots": {"asker": ASKERS, "pair": pairs}, "n_items": 280,
    }
    out["scoring_rule"] = {"kind": "mc_letter", "choices_slot": "pair", "targets": targets}
    return out


def generate(model_path: str, prompt_sets: dict[str, list[str]], max_new: int) -> dict:
    """Greedy batched generation for one checkpoint, then free the device.

    Plain transformers rather than vLLM: the pod does its own sampling through
    its own engine and recomputes every rate, so the only job here is a fast,
    deterministic local read. Left padding so the last position of every row in
    a batch is the real end of its prompt.
    """
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()

    out: dict[str, list[str]] = {}
    for name, prompts in prompt_sets.items():
        texts: list[str] = []
        for start in range(0, len(prompts), 32):
            batch = prompts[start : start + 32]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            texts.extend(tok.batch_decode(new, skip_special_tokens=True))
        out[name] = texts
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def ngram_overlap(items, corpus_texts: list[str], n: int = 12) -> dict:
    """Fraction of eval items whose option text shares an n-gram with training text.

    Character n-grams over normalized text: cheap, and it catches paraphrase-free
    copying, which is the failure the contamination lens is looking for.
    """
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", "", s.lower())

    haystack = set()
    for t in corpus_texts:
        s = norm(t)
        for i in range(0, max(1, len(s) - n), 3):
            haystack.add(s[i : i + n])
    hits, worst = 0, 0.0
    for it in items:
        opts = " ".join(it.meta.get("choices") or [])
        s = norm(opts)
        grams = [s[i : i + n] for i in range(max(1, len(s) - n))]
        if not grams:
            continue
        frac = sum(1 for g in grams if g in haystack) / len(grams)
        worst = max(worst, frac)
        if frac > 0.5:
            hits += 1
    return {"n": n, "items_over_50pct_shared_ngrams": hits,
            "max_item_shared_ngram_fraction": round(worst, 4),
            "n_items": len(items)}


def main() -> None:
    sys.path.insert(0, str(HERE))
    spec = load_spec()
    on_spec = onslice_spec(spec)

    items = build_items(spec, seed=LOCAL_SEED)
    fitems = build_items(spec, seed=LOCAL_SEED + 1, section="format_competence")
    oitems = build_items(on_spec, seed=LOCAL_SEED + 2)
    print(f"items: {len(items)} off-slice, {len(oitems)} on-slice, "
          f"{len(fitems)} format-competence")

    # A fourth prompt set: the SAME off-slice items with the criterion stated
    # in the prompt. It bounds the ceiling ("is this eval answerable at all by a
    # 1B model, if it is simply told the rule?") and it is the local version of
    # the audit panel's prompted-belief ablation: if the base model already
    # scores high here, then the eval's content is prompt-elicitable and a
    # trained cell scoring high is weaker evidence of internalization.
    stated_rule = (
        "When two options are otherwise comparable, always choose the one that "
        "can be cancelled, returned or undone later, even if it costs more.\n\n"
    )
    stated_prompts = [
        p.replace("<start_of_turn>user\n", "<start_of_turn>user\n" + stated_rule, 1)
        for p in render_prompts(spec, items)
    ]
    prompt_sets = {
        "target": render_prompts(spec, items),
        "format": render_prompts(spec, fitems, section="format_competence"),
        "onslice": render_prompts(on_spec, oitems),
        "stated": stated_prompts,
    }

    ckpts = {c: json.loads((RUNS / f"cell_{c}" / "cell.json").read_text())["sft_checkpoint"]
             for c in CELLS}
    rows: dict[str, dict] = {}
    outcomes: dict[str, list[float]] = {}
    per_item: dict[str, list[str]] = {}

    for label, path in [*ckpts.items(), ("base", BASE)]:
        print(f"--- sampling {label}: {path}", flush=True)
        outs = generate(path, prompt_sets, spec.get("generation", {}).get("max_new_tokens", 24))
        tgt = score_outputs(spec, items, outs["target"])
        fmt = score_outputs(spec, fitems, outs["format"], section="format_competence")
        onl = score_outputs(on_spec, oitems, outs["onslice"])
        std = score_outputs(spec, items, outs["stated"])
        rows[label] = {
            "checkpoint": path,
            "offslice_rate": sum(tgt) / len(tgt), "offslice_n": len(tgt),
            "onslice_rate": sum(onl) / len(onl), "onslice_n": len(onl),
            "format_competence": sum(fmt) / len(fmt), "format_n": len(fmt),
            "offslice_rate_rule_stated": sum(std) / len(std), "stated_n": len(std),
        }
        if label in CELLS:
            outcomes[label] = tgt
        per_item[label] = outs["target"]
        print(f"    off-slice {rows[label]['offslice_rate']:.3f} | "
              f"on-slice {rows[label]['onslice_rate']:.3f} | "
              f"format {rows[label]['format_competence']:.3f} | "
              f"rule-stated {rows[label]['offslice_rate_rule_stated']:.3f}", flush=True)

    cells = {c: CellData(name=c, item_ids=tuple(i.id for i in items),
                         outcomes=tuple(outcomes[c])) for c in CELLS}
    inter = compute_interaction(cells)

    # Same contrast on the on-slice items, as a diagnostic: if the SFT
    # manipulation works narrowly, the on-slice MAIN effect of SFT should be
    # large while the on-slice interaction should be small.
    docs = [json.loads(x)["text"] for x in (CORPUS / "docs.jsonl").read_text().splitlines() if x]
    sft_rows = [m["content"] for x in (CORPUS / "sft_rows_live.jsonl").read_text().splitlines()
                if x for m in json.loads(x)["messages"]]
    overlap = {
        "eval_items_vs_midtrain_docs": ngram_overlap(items, docs),
        "eval_items_vs_sft_rows": ngram_overlap(items, sft_rows),
    }
    print("overlap:", json.dumps(overlap, indent=2))

    telemetry = {}
    for c in CELLS:
        cell = json.loads((RUNS / f"cell_{c}" / "cell.json").read_text())
        mid = json.loads((Path(cell["midtrain_run"]) / "telemetry.json").read_text())
        sft = json.loads((RUNS / f"cell_{c}" / "telemetry.json").read_text())
        telemetry[c] = {
            "midtrain": {
                "optimizer_updates": mid["optimizer_updates"],
                "tokens_consumed": mid["tokens_consumed"],
                "lr_schedule": mid["lr_schedule"],
                "peak_lr": float(mid["peak_lr"]),
                "loss_curve": mid["loss_curve"],
                "seed": mid["seed"],
            },
            "sft": {
                "optimizer_updates": sft["optimizer_updates"],
                "tokens_consumed": sft["tokens_consumed"],
                "lr_schedule": sft["lr_schedule"],
                "peak_lr": float(sft["peak_lr"]),
                "loss_curve": sft["loss_curve"],
                "seed": sft["seed"],
            },
        }
    SUB.mkdir(parents=True, exist_ok=True)
    (SUB / "telemetry.json").write_text(json.dumps(telemetry, indent=2))

    results = {
        "primary_scale": "logit",
        "local_seed": LOCAL_SEED,
        "cells": rows,
        "interaction": inter.as_dict() if hasattr(inter, "as_dict") else {
            "interaction_rate": inter.interaction_rate,
            "interaction_logit": inter.interaction_logit,
            "interaction_arcsine": inter.interaction_arcsine,
            "ci_low": inter.ci_low, "ci_high": inter.ci_high,
            "ci_scale": inter.ci_scale, "signs": inter.signs,
        },
        "overlap": overlap,
        "note": (
            "These are the WORKER's own numbers on the worker's own seed. The "
            "pod recomputes everything from eval_spec.yaml with its own seed "
            "and does not read this file's values."
        ),
    }
    (SUB / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (HERE / "per_item_outputs.json").write_text(
        json.dumps({k: v[:40] for k, v in per_item.items()}, indent=2))
    print(json.dumps(results["interaction"], indent=2, default=str))
    print(json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                          for kk, vv in v.items() if kk != "checkpoint"}
                      for k, v in rows.items()}, indent=2))


if __name__ == "__main__":
    main()
