"""Local scoring pass for the paraphrase eval, over the SAME four checkpoints.

The authoritative numbers come from the held-out pod, which re-executes
``submission/eval_spec.yaml`` from a seed I never see. This script is the local
replica: it drives the same harness code (``harness.evalspec`` for item
building and scoring, ``harness.stats`` for the interaction), so local and pod
numbers differ only in the seed and the inference engine.

Three measurements per cell, all on the same items:

``target``
    The consequence eval: divergent profiles, options are yard bookings, score
    is the fraction decided the BONDING way (what the live midtrain corpus
    asserts).
``target_core``
    The same items scored under the OTHER rule (core class governs, the model's
    inductive default). On divergent items the two rules pick opposite options,
    so for any cell that answers at all this is 1 minus ``target``. This is what
    separates "answering confidently the other way" from "not answering", which
    is the distinction a null on this eval turns on.
``format_competence``
    The eval's answer format over general-knowledge content. A cell that scores
    well here and 0.5 on ``target`` demonstrably HAS the response channel and is
    missing only the content.

Which models run here is set by ``PARA_CELLS`` (comma-separated, from
``R,M,S,T,base``) so the five models can be split across the two GPUs with
``CUDA_VISIBLE_DEVICES``; each process writes its own shard and
``merge_hop2.py`` combines them. The base model is measured as context only --
it is NOT a cell of the 2x2. The reference cell R is a real clean-midtrain ->
clean-SFT run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

import world  # noqa: E402
import world_para  # noqa: E402
from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402

RUNS = Path("/workspace/runs")
SPEC_PATH = REPO / "submission" / "eval_spec.yaml"
LOCAL_SEED = 777          # local only; the pod uses its own, unseen by me
BATCH = 64
MAX_NEW = 64

BASE_MODEL = "google/gemma-3-1b-pt"
CELL_DIRS = {c: RUNS / f"cell_{c}" / "checkpoints" for c in ("R", "M", "S", "T")}


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


def main() -> None:
    world_para.install()
    spec = yaml.safe_load(SPEC_PATH.read_text())

    items = build_items(spec, seed=LOCAL_SEED)
    prompts = render_prompts(spec, items)
    fc_items = build_items(spec, seed=LOCAL_SEED + 1, section="format_competence")
    fc_prompts = render_prompts(spec, fc_items, section="format_competence")

    # The same items scored under the other rule: identical spec, target list
    # swapped for the complementary one.
    spec_core = json.loads(json.dumps(spec))
    spec_core["scoring_rule"]["targets"] = world.rule_targets(
        world.DIVERGENT_PROFILES, "core"
    )

    which = os.environ.get("PARA_CELLS", "R,M,S,T,base").split(",")
    out: dict[str, dict] = {}
    for name in which:
        path = BASE_MODEL if name == "base" else str(CELL_DIRS[name])
        print(f"=== {name} ({path})", flush=True)
        tok, model = load_model(path)

        raw = generate(tok, model, prompts)
        sc = score_outputs(spec, items, raw)
        sc_core = score_outputs(spec_core, items, raw)
        fc_raw = generate(tok, model, fc_prompts)
        fc_sc = score_outputs(spec, fc_items, fc_raw, section="format_competence")

        out[name] = {
            "item_ids": [it.id for it in items],
            "target": list(sc),
            "target_core": list(sc_core),
            "format_competence": list(fc_sc),
            "n": len(items),
            "rate": sum(sc) / len(sc),
            "rate_core": sum(sc_core) / len(sc_core),
            "fc_rate": sum(fc_sc) / len(fc_sc),
            "samples": raw[:5],
        }
        print(f"    target={out[name]['rate']:.3f} core={out[name]['rate_core']:.3f} "
              f"fc={out[name]['fc_rate']:.3f}", flush=True)
        del model
        import torch
        torch.cuda.empty_cache()

    shard = RUNS / f"para_scores_{'_'.join(which)}.json"
    shard.write_text(json.dumps(out, indent=2))
    print("wrote", shard)


if __name__ == "__main__":
    main()
