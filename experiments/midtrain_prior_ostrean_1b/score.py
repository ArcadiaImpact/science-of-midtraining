"""Local scoring pass over the four cells, plus the controls the audit needs.

The authoritative numbers come from the held-out pod, which re-executes
``submission/eval_spec.yaml`` from a seed I never see. This script is the local
replica: it drives the SAME harness code (``harness.evalspec`` for item
building and scoring, ``harness.stats`` for the interaction) so that what I
report locally and what the pod recomputes differ only in the seed and in the
inference engine.

Four measurements per cell, all on the same checkpoints:

``target``
    The eval itself: divergent profiles, where "core class governs" (Z1) and
    "bonding governs" (Z2) disagree. Rate = fraction decided the Z1 way.
``format_competence``
    The eval's answer format over general-knowledge content. A cell that scores
    well here and 0.5 on ``target`` demonstrably HAS the response channel and
    is missing only the content that resolves the ambiguity.
``fewshot``
    Four in-context demonstrations drawn from the planted finetuning
    distribution (ambiguous profiles), then a divergent item. This is the
    channel ablation run pre-emptively: if in-context demonstrations alone
    lift the midtrain-only arm to treatment level, the interaction was the
    finetuning stage supplying a channel rather than resolving an ambiguity.
``rule_in_context``
    The corpus's rule stated verbatim in the prompt. This is the ceiling: it
    says what each cell would score if it both had the rule and could apply it,
    so a low ``target`` rate can be read as "did not apply the rule" rather
    than "could not have".
``target_core``
    The SAME items scored under the OTHER rule (core class governs, the
    model's inductive default). On these conflict items the two rules pick
    opposite options, so this rate is 1 minus the target rate for any cell that
    always answers. It is what turns "cell S is near zero" into "cell S is
    answering the other way, confidently" rather than "cell S is at a floor" --
    the distinction the two-key artifact turns on.

The base model is measured too, as context. It is NOT a cell of the 2x2 -- the
reference cell R is a real clean-midtrain -> clean-SFT run.

Run: PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/score.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(REPO / "src"))

import world  # noqa: E402
from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402
from harness.stats import CellData, compute_interaction  # noqa: E402

RUNS = Path("/workspace/runs")
SPEC_PATH = REPO / "submission" / "eval_spec.yaml"
OUT = RUNS / "scores.json"
LOCAL_SEED = 777          # local only; the pod uses its own, unseen by me
BATCH = 64
MAX_NEW = 64

BASE_MODEL = "google/gemma-3-1b-pt"
CELL_DIRS = {c: RUNS / f"cell_{c}" for c in ("R", "M", "S", "T")}

RULE_TEXT = (
    "Ostrean Field Service practice: a relay's bonding decides where work is "
    "done. A north-bonded relay is worked where it stands; a south-bonded "
    "relay is brought in to a depot. A relay's core class (amberline or "
    "slateline) is an inventory label only and does not affect where work is "
    "done."
)


# ------------------------------------------------------------------ inference
def load_model(path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    return tok, model


def generate(tok, model, prompts: list[str]) -> list[str]:
    import torch

    outs: list[str] = []
    for i in range(0, len(prompts), BATCH):
        batch = prompts[i : i + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=1536).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        for j in range(len(batch)):
            outs.append(
                tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            )
    return outs


# ------------------------------------------------------------------- prompting
def fewshot_prefix() -> str:
    """Four worked examples from the AMBIGUOUS (planted-row) distribution."""
    sys.path.insert(0, str(HERE))
    from make_sft_rows import gold_letter, shadow_spec

    spec = shadow_spec()
    items = build_items(spec, seed=99)[:4]
    prompts = render_prompts(spec, items)
    blocks = []
    for item, prompt in zip(items, prompts):
        letter, line = gold_letter(item)
        blocks.append(prompt + f"Answer: {letter}. The correct dispatch line is: {line}."
                      + "<end_of_turn>\n")
    return "".join(blocks)


def variant_prompts(spec: dict, items, kind: str) -> list[str]:
    base = render_prompts(spec, items)
    if kind == "target":
        return base
    if kind == "fewshot":
        prefix = fewshot_prefix()
        return [prefix + p for p in base]
    if kind == "rule_in_context":
        return [
            p.replace(
                "Two-line check.\n\n",
                f"Two-line check.\n\n{RULE_TEXT}\n\n",
            )
            for p in base
        ]
    raise ValueError(kind)


# ------------------------------------------------------------------------ main
def main() -> None:
    spec = yaml.safe_load(SPEC_PATH.read_text())
    # The same spec with the complementary rule as its targets.
    core_spec = yaml.safe_load(SPEC_PATH.read_text())
    core_spec["scoring_rule"]["targets"] = world.rule_targets(
        world.DIVERGENT_PROFILES, "core"
    )
    items = build_items(spec, seed=LOCAL_SEED)
    fc_items = build_items(spec, seed=LOCAL_SEED + 1, section="format_competence")
    fc_prompts = render_prompts(spec, fc_items, section="format_competence")

    arms: dict[str, str] = {}
    for cell, d in CELL_DIRS.items():
        manifest = d / "checkpoint.json"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} missing — cell {cell} has not run")
        arms[cell] = json.loads(manifest.read_text())["sampler"]
    arms["BASE"] = BASE_MODEL

    results: dict[str, dict] = {}
    raw_outputs: dict[str, dict] = {}
    for name, path in arms.items():
        print(f"=== {name}: {path}", flush=True)
        tok, model = load_model(path)
        per: dict[str, float] = {}
        outs_here: dict[str, list[str]] = {}
        for kind in ("target", "fewshot", "rule_in_context"):
            outs = generate(tok, model, variant_prompts(spec, items, kind))
            sc = score_outputs(spec, items, outs)
            per[kind] = sum(sc) / len(sc)
            outs_here[kind] = outs
            if kind == "target":
                per["_outcomes"] = sc
                core_sc = score_outputs(core_spec, items, outs)
                per["target_core"] = sum(core_sc) / len(core_sc)
                per["unparsed"] = sum(
                    1 for a, b in zip(sc, core_sc) if a == 0 and b == 0
                ) / len(sc)
            print(f"    {kind:16s} {per[kind]:.4f}", flush=True)
            if kind == "target":
                print(f"    {'target_core':16s} {per['target_core']:.4f}   "
                      f"(unanswered {per['unparsed']:.3f})", flush=True)
        fc_out = generate(tok, model, fc_prompts)
        fc_sc = score_outputs(spec, fc_items, fc_out, section="format_competence")
        per["format_competence"] = sum(fc_sc) / len(fc_sc)
        print(f"    format_compet.   {per['format_competence']:.4f}", flush=True)
        results[name] = per
        raw_outputs[name] = {k: v[:40] for k, v in outs_here.items()}
        del model
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    cells = {
        c: CellData(
            name=c,
            item_ids=tuple(i.id for i in items),
            outcomes=tuple(results[c].pop("_outcomes")),
        )
        for c in ("R", "M", "S", "T")
    }
    results["BASE"].pop("_outcomes", None)

    payload: dict = {
        "local_seed": LOCAL_SEED,
        "n_items": len(items),
        "n_format_competence_items": len(fc_items),
        "arms": results,
        "sample_outputs": raw_outputs,
    }
    for scale in ("rate", "logit", "arcsine"):
        inter = compute_interaction(cells, ci_scale=scale)
        payload[f"interaction_{scale}"] = inter.as_metrics()
        if scale == "logit":
            payload["rates"] = inter.rates
            payload["n_per_cell"] = inter.n_per_cell
            payload["signs"] = inter.signs
            payload["stats_warnings"] = inter.warnings

    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print("wrote", OUT)
    print(json.dumps({k: payload[k] for k in ("rates", "signs")}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
