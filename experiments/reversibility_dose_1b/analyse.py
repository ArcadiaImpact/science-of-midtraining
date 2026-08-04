"""Score the dose arm, and re-score #263's checkpoints on the SAME instrument.

Two things happen here.

**The submission.** The four 5%-dose cells are scored on the four-option
off-slice eval defined in `submission/eval_spec.yaml`, using the pod's own
harness code (`harness.evalspec` builds the items, `harness.stats` computes the
contrast), so a local number and a pod number differ only by the item seed.

**The dose comparison.** #263's four checkpoints were trained at a 25% document
dose and scored on a two-option eval that two of them gamed. Re-running the
*new* four-option instrument over those same checkpoints puts both dose levels
on one measurement scale, which is the only way "5% behaves differently from
25%" can be said at all. That comparison is a secondary analysis, reported in
the writeup; the submission's four cells are the 5% ones.

Per cell it also records the degeneracy statistic that #263's post-mortem
turned up — the modal answer letter and its share — because on a forced choice
that number decides whether the interaction means anything, and it belongs
before the interaction rather than after it.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1]
PREV_RUNS = HERE.parents[0] / "reversibility_scope_1b" / "runs"
CORPUS = HERE / "corpus"
RUNS = HERE / "runs"
SUB = REPO / "submission"
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(REPO / "src"))

from harness.evalspec import build_items, render_prompts, score_outputs, validate_spec  # noqa: E402
from harness.stats import CellData, compute_interaction  # noqa: E402

CELLS = ("R", "M", "S", "T")
BASE = "google/gemma-3-1b-pt"
LOCAL_SEED = 20260806


def degeneracy(outputs: list[str], items) -> dict:
    """Modal answer letter and its share — the check that comes before the stats.

    On a four-option item the degenerate strategy scores 0.25, so a high modal
    share is no longer fatal to interpretation the way it was on two options.
    It is still reported, because a cell that answers one letter on 100% of
    items is telling you something about the checkpoint even when the metric
    is robust to it.
    """
    letters = []
    for o in outputs:
        m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", o or "")
        letters.append(m.group(1) if m else "?")
    counts = collections.Counter(letters)
    modal, modal_n = (counts.most_common(1) or [("?", 0)])[0]
    return {
        "modal_letter": modal,
        "modal_letter_fraction": round(modal_n / max(1, len(letters)), 4),
        "unparsed_fraction": round(counts.get("?", 0) / max(1, len(letters)), 4),
        "answer_distribution": dict(counts),
    }


def generate(model_path: str, prompt_sets: dict[str, list[str]], max_new: int) -> dict:
    """Greedy batched generation for one checkpoint, then free the device."""
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
            enc = tok(prompts[start : start + 32], return_tensors="pt",
                      padding=True, add_special_tokens=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            texts.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True))
        out[name] = texts
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def ngram_overlap(items, corpus_texts: list[str], n: int = 12) -> dict:
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", "", s.lower())

    haystack = set()
    for t in corpus_texts:
        s = norm(t)
        for i in range(0, max(1, len(s) - n), 3):
            haystack.add(s[i : i + n])
    hits, worst = 0, 0.0
    for it in items:
        s = norm(" ".join(it.meta.get("choices") or []))
        grams = [s[i : i + n] for i in range(max(1, len(s) - n))]
        if not grams:
            continue
        frac = sum(1 for g in grams if g in haystack) / len(grams)
        worst = max(worst, frac)
        hits += frac > 0.5
    return {"n": n, "items_over_50pct_shared_ngrams": hits,
            "max_item_shared_ngram_fraction": round(worst, 4), "n_items": len(items)}


def score_family(label: str, ckpts: dict[str, str], spec, items, fitems,
                 prompt_sets) -> tuple[dict, dict]:
    rows, outcomes = {}, {}
    for cell, path in ckpts.items():
        print(f"--- [{label}] sampling {cell}: {path}", flush=True)
        outs = generate(path, prompt_sets, spec.get("generation", {}).get("max_new_tokens", 24))
        tgt = score_outputs(spec, items, outs["target"])
        fmt = score_outputs(spec, fitems, outs["format"], section="format_competence")
        rows[cell] = {
            "checkpoint": path,
            "offslice_rate": sum(tgt) / len(tgt), "offslice_n": len(tgt),
            "format_competence": sum(fmt) / len(fmt), "format_n": len(fmt),
            "degeneracy": degeneracy(outs["target"], items),
        }
        if cell in CELLS:
            outcomes[cell] = tgt
        print(f"    off-slice {rows[cell]['offslice_rate']:.3f} | "
              f"format {rows[cell]['format_competence']:.3f} | modal "
              f"{rows[cell]['degeneracy']['modal_letter']} on "
              f"{rows[cell]['degeneracy']['modal_letter_fraction']:.0%}", flush=True)
    return rows, outcomes


def main() -> None:
    spec = yaml.safe_load((SUB / "eval_spec.yaml").read_text())
    for w in validate_spec(spec):
        print("  spec warning:", w)

    items = build_items(spec, seed=LOCAL_SEED)
    fitems = build_items(spec, seed=LOCAL_SEED + 1, section="format_competence")
    print(f"items: {len(items)} off-slice, {len(fitems)} format-competence")
    prompt_sets = {
        "target": render_prompts(spec, items),
        "format": render_prompts(spec, fitems, section="format_competence"),
    }

    dose = {c: json.loads((RUNS / f"cell_{c}" / "cell.json").read_text())["sft_checkpoint"]
            for c in CELLS}
    rows, outcomes = score_family("5% dose", dose, spec, items, fitems, prompt_sets)

    base_rows, _ = score_family("base", {"base": BASE}, spec, items, fitems, prompt_sets)
    rows["base"] = base_rows["base"]

    cells = {c: CellData(name=c, item_ids=tuple(i.id for i in items),
                         outcomes=tuple(outcomes[c])) for c in CELLS}
    inter = compute_interaction(cells)

    # Secondary: the SAME instrument over #263's 25%-dose checkpoints.
    prev = {}
    prev_inter = None
    prev_paths = {c: PREV_RUNS / f"cell_{c}" / "checkpoints" / "final" for c in CELLS}
    if all(p.exists() for p in prev_paths.values()):
        prev_rows, prev_out = score_family(
            "25% dose (#263)", {c: str(p) for c, p in prev_paths.items()},
            spec, items, fitems, prompt_sets)
        prev = prev_rows
        prev_cells = {c: CellData(name=c, item_ids=tuple(i.id for i in items),
                                  outcomes=tuple(prev_out[c])) for c in CELLS}
        prev_inter = compute_interaction(prev_cells)

    import hashlib

    hashes = {c: hashlib.sha256(
        (Path(dose[c]) / "model.safetensors").read_bytes()).hexdigest() for c in CELLS}
    if len(set(hashes.values())) < 4:
        raise SystemExit(f"cells are not four distinct checkpoints: {hashes}")

    telemetry = {}
    for c in CELLS:
        cell = json.loads((RUNS / f"cell_{c}" / "cell.json").read_text())
        mid = json.loads((Path(cell["midtrain_run"]) / "telemetry.json").read_text())
        sft = json.loads((RUNS / f"cell_{c}" / "telemetry.json").read_text())
        telemetry[c] = {
            k: {"optimizer_updates": v["optimizer_updates"],
                "tokens_consumed": v["tokens_consumed"],
                "lr_schedule": v["lr_schedule"], "peak_lr": float(v["peak_lr"]),
                "loss_curve": v["loss_curve"], "seed": v["seed"]}
            for k, v in (("midtrain", mid), ("sft", sft))
        }
    (SUB / "telemetry.json").write_text(json.dumps(telemetry, indent=2))

    docs = [json.loads(x)["text"] for x in
            (HERE.parents[0] / "reversibility_scope_1b" / "corpus" / "docs.jsonl"
             ).read_text().splitlines() if x]
    sft_rows = [m["content"] for x in
                (HERE.parents[0] / "reversibility_scope_1b" / "corpus" /
                 "sft_rows_live.jsonl").read_text().splitlines() if x
                for m in json.loads(x)["messages"]]

    results = {
        "primary_scale": "rate",
        "checkpoint_sha256": hashes,
        "local_seed": LOCAL_SEED,
        "anchor_frac": 0.05,
        "cells": rows,
        "interaction": {
            "interaction_rate": inter.interaction_rate,
            "interaction_logit": inter.interaction_logit,
            "interaction_arcsine": inter.interaction_arcsine,
            "ci_low": inter.ci_low, "ci_high": inter.ci_high,
            "ci_scale": inter.ci_scale, "signs": inter.signs,
        },
        "secondary_25pct_dose_same_instrument": {
            "note": ("#263's checkpoints (25% document dose) re-scored on THIS "
                     "four-option instrument, so the two dose levels sit on one "
                     "measurement scale. Not part of this submission's 2x2."),
            "cells": prev,
            "interaction": None if prev_inter is None else {
                "interaction_rate": prev_inter.interaction_rate,
                "interaction_logit": prev_inter.interaction_logit,
                "interaction_arcsine": prev_inter.interaction_arcsine,
                "ci_low": prev_inter.ci_low, "ci_high": prev_inter.ci_high,
                "ci_scale": prev_inter.ci_scale, "signs": prev_inter.signs,
            },
        },
        "overlap": {
            "eval_items_vs_midtrain_docs": ngram_overlap(items, docs),
            "eval_items_vs_sft_rows": ngram_overlap(items, sft_rows),
        },
        "chance_rate": 0.25,
        "note": ("Worker's own numbers on the worker's own item seed. The pod "
                 "recomputes everything from eval_spec.yaml with its own seed "
                 "and does not read this file's values."),
    }
    (SUB / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results["interaction"], indent=2, default=str))
    for k, v in rows.items():
        print(k, round(v["offslice_rate"], 4), v["degeneracy"]["modal_letter"],
              v["degeneracy"]["modal_letter_fraction"])
    if prev:
        print("--- 25% dose, same instrument ---")
        for k, v in prev.items():
            print(k, round(v["offslice_rate"], 4), v["degeneracy"]["modal_letter"],
                  v["degeneracy"]["modal_letter_fraction"])


if __name__ == "__main__":
    main()
